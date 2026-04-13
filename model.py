from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch
from dataclasses import dataclass


@dataclass
class Tacotron2Config:
    n_mels = 80

    # encoder
    vocab_size = 76
    enc_hidden_dim = 512
    enc_n_convs = 3
    enc_conv_kernel = 5
    enc_dropout_p = 0.5

    # attn
    attn_hidden_dim = 128 
    attn_loc_conv_dim = 32
    attn_loc_conv_kernel = 31

    # decoder
    dec_hidden_dim = 1024
    dec_prenet_hidden_dim = 256
    dec_prenet_depth = 2
    dec_prenet_dropout_p = 0.5
    
    dec_postnet_n_convs = 5
    dec_postnet_conv_kernel = 5
    dec_postnet_conv_dim = 512
    dec_postnet_dropout_p = 0.5


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.text_emb = nn.Embedding(config.vocab_size, config.enc_hidden_dim)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        config.enc_hidden_dim,
                        config.enc_hidden_dim,
                        config.enc_conv_kernel,
                        padding="same",
                    ),
                    nn.BatchNorm1d(config.enc_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.enc_dropout_p),
                )
                for _ in range(config.enc_n_convs)
            ]
        )

        self.lstm = nn.LSTM(
            config.enc_hidden_dim,
            config.enc_hidden_dim // 2,
            bidirectional=True,
        )

    def forward(self, text_tokens, text_tokens_lengths):
        x = self.text_emb(text_tokens).transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = x.transpose(1, 2)
        x = pack_padded_sequence(
            x, text_tokens_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        return x


class LocalSensitiveAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.enc_proj = nn.Linear(config.enc_hidden_dim, config.attn_hidden_dim)
        self.dec_proj = nn.Linear(config.dec_hidden_dim, config.attn_hidden_dim, bias=False)
        self.loc_conv = nn.Conv1d(2, config.attn_loc_conv_dim, config.attn_loc_conv_kernel, padding="same")
        self.loc_proj = nn.Linear(config.attn_loc_conv_dim, config.attn_hidden_dim, bias=False)
        self.final_proj = nn.Linear(config.attn_hidden_dim, 1, bias=False)

        self.reset()

    def reset(self):
        self.h_cache = None

    def forward(self, decoder_hidden, encoder_outputs, attn_weights_cat, mask=None):
        if self.h_cache is None:
            self.h_cache = self.enc_proj(encoder_outputs)

        h = self.h_cache
        s = self.dec_proj(decoder_hidden.unsqueeze(1))
        f = self.loc_proj(self.loc_conv(attn_weights_cat).transpose(1, 2))

        logits = self.final_proj(torch.tanh(h + s + f))
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        weights = F.softmax(logits.squeeze(-1), dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, weights


class Prenet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        dims = [config.n_mels] + [
            config.dec_prenet_hidden_dim
        ] * config.dec_prenet_depth

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1], bias=False),
                    nn.ReLU(),
                )
                for i in range(len(dims) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.dropout(x, self.config.dec_prenet_dropout_p, training=True)

        return x


class Postnet(nn.Module):
    def __init__(self, config):
        super().__init__()

        dims = [config.n_mels] + [config.dec_postnet_conv_dim] * (config.dec_postnet_n_convs - 1) + [config.n_mels]
        last_i = len(dims) - 2

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dims[i], dims[i+1], config.dec_postnet_conv_kernel, padding="same"),
                nn.BatchNorm1d(config.dec_postnet_conv_dim) if i != last_i else nn.BatchNorm1d(config.n_mels),
                nn.Tanh() if i != last_i else nn.Identity(),
                nn.Dropout(config.dec_postnet_dropout_p)
            )
            for i in range(len(dims)-1)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.prenet = Prenet(config)

        self.attn = LocalSensitiveAttention(config)

        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(
                    config.dec_prenet_hidden_dim + config.enc_hidden_dim,
                    config.dec_hidden_dim,
                ),
                nn.LSTMCell(config.dec_hidden_dim, config.dec_hidden_dim),
            ]
        )

        self.mel_proj = nn.Linear(
            config.dec_hidden_dim + config.enc_hidden_dim, config.n_mels
        )
        self.stop_proj = nn.Linear(config.dec_hidden_dim + config.enc_hidden_dim, 1)

        self.postnet = Postnet(config)

    def _init_decoder(self, encoder_outputs, encoder_mask=None):
        B, T, E = encoder_outputs.shape
        device = encoder_outputs.device

        self.h = [
            torch.zeros(B, self.config.dec_hidden_dim, device=device) for _ in range(2)
        ]
        self.c = [
            torch.zeros(B, self.config.dec_hidden_dim, device=device) for _ in range(2)
        ]

        self.attn.reset()
        self.cum_attn_weight = torch.zeros(B, T, device=device)
        self.attn_weights = torch.zeros(B, T, device=device)
        self.attn_context = torch.zeros(B, self.config.enc_hidden_dim, device=device)

        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask

    def decoder_step(self, mel_step):

        attn_weights_cat = torch.cat(
            [self.attn_weights.unsqueeze(1), self.cum_attn_weight.unsqueeze(1)], dim=1
        )
        attn_context, attn_weights = self.attn(
            self.h[1],
            self.encoder_outputs,
            attn_weights_cat,
            mask=self.encoder_mask,
        )
        self.attn_context = attn_context
        self.attn_weights = attn_weights
        self.cum_attn_weight += attn_weights

        rnn_input = torch.cat([mel_step, self.attn_context], dim=-1)

        self.h[0], self.c[0] = self.rnn[0](rnn_input, (self.h[0], self.c[0]))
        self.h[1], self.c[1] = self.rnn[1](self.h[0], (self.h[1], self.c[1]))
        rnn_out = self.h[1]

        proj_input = torch.cat([rnn_out, self.attn_context], dim=-1)
        mel_out = self.mel_proj(proj_input)
        stop_out = self.stop_proj(proj_input)

        return mel_out, stop_out, attn_weights

    def forward(self, encoder_outputs, encoder_mask, mels, mels_mask):
        self._init_decoder(encoder_outputs, encoder_mask)

        B, T_dec, _ = mels.shape

        bos = torch.zeros(B, 1, self.config.n_mels, device=mels.device)
        dec_input = torch.cat([bos, mels[:, :-1]], dim=1)
        dec_input = self.prenet(dec_input)

        mel_outs, stop_outs, attentions = [], [], []
        for t in range(T_dec):
            mel_t, stop_t, attn_t = self.decoder_step(dec_input[:, t])
            mel_outs.append(mel_t)
            stop_outs.append(stop_t)
            attentions.append(attn_t)
        
        mel_outs = torch.stack(mel_outs, dim=1)
        stop_outs = torch.stack(stop_outs, dim=1).squeeze(-1)
        attentions = torch.stack(attentions, dim=1)

        mel_outs = mel_outs.masked_fill(~mels_mask.unsqueeze(-1), 0.0)
        mel_post = mel_outs + self.postnet(mel_outs.transpose(1, 2)).transpose(1, 2)
        mel_post = mel_post.masked_fill(~mels_mask.unsqueeze(-1), 0.0)

        return mel_outs, mel_post, stop_outs, attentions


    @torch.inference_mode()
    def inference(self, encoder_outputs, encoder_mask=None, max_steps=1000, stop_threshold=0.5):
        self._init_decoder(encoder_outputs, encoder_mask)

        B = encoder_outputs.size(0)
        device = encoder_outputs.device
        mel_input = torch.zeros(B, self.config.n_mels, device=device)

        mels, stops, attns = [], [], []
        for _ in range(max_steps):
            prenet_out = self.prenet(mel_input)
            mel_t, stop_t, attn_t = self.decoder_step(prenet_out)

            mels.append(mel_t)
            stops.append(stop_t)
            attns.append(attn_t)

            if torch.sigmoid(stop_t).max().item() > stop_threshold:
                break
            mel_input = mel_t

        mel_outs = torch.stack(mels, dim=1)
        stop_logits = torch.stack(stops, dim=1).squeeze(-1)
        attentions = torch.stack(attns, dim=1)
        mel_post = mel_outs + self.postnet(mel_outs.transpose(1, 2)).transpose(1, 2)

        return mel_outs, mel_post, stop_logits, attentions


class Tacotron2(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self, batch):
        text = batch["text"]
        text_lengths = batch["text_lengths"]
        mel = batch["mel"]
        mel_mask = batch["mel_mask"]

        enc_out = self.encoder(text, text_lengths)

        T_enc = enc_out.size(1)
        text_mask = (
            torch.arange(T_enc, device=enc_out.device)[None, :]
            < text_lengths.to(enc_out.device)[:, None]
        )  # (B, T_enc)
        encoder_mask = text_mask.unsqueeze(-1)  

        mel, mel_post, stop_logits, attentions = self.decoder(
            enc_out, encoder_mask, mel, mel_mask
        )

        return {
            "mel": mel,
            "mel_post": mel_post,
            "stop_logits": stop_logits,
            "attentions": attentions,
        }

    @torch.inference_mode()
    def inference(self, text_tokens, text_lengths, max_steps=1000, stop_threshold=0.5):
        enc_out = self.encoder(text_tokens, text_lengths)

        T_enc = enc_out.size(1)
        text_mask = (
            torch.arange(T_enc, device=enc_out.device)[None, :]
            < text_lengths.to(enc_out.device)[:, None]
        )
        encoder_mask = text_mask.unsqueeze(-1)

        mel, mel_post, stop_logits, attentions = self.decoder.inference(
            enc_out, encoder_mask, max_steps=max_steps, stop_threshold=stop_threshold,
        )
        return {
            "mel": mel,
            "mel_post": mel_post,
            "stop_logits": stop_logits,
            "attentions": attentions,
        }


if __name__ == "__main__":
    config = Tacotron2Config()
    encoder = Encoder(config)

    text_tokens = torch.tensor(
        [
            [13,54,44,1,49,60,1,63,41,59,1,41,1,53,41,60,60,45,58,1,55,46,1,43,55,61,58,59,45,1,60,48,41,58,45,7,],
            [    13,    54,    44,    1,    49,    60,    1,    63,    41,    59,    1,    41,    1,    53,    41,    60,    60,    45,    58,    1,    55,    46,    1,    43,    55,    61,    58,    59,    45,    1,    60,    0,    0,    0,    0,    0,],
            [ 13, 54, 44, 1, 49, 60, 1, 63, 41, 59, 1, 41, 1, 53, 41, 60, 60, 45, 58, 1, 55, 46, 1, 43, 55, 61, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
        ]
    )
    text_tokens_lengts = torch.tensor([36, 31, 27])
    print(text_tokens.shape)

    out = encoder(text_tokens, text_tokens_lengts)
    print(out.shape)

    B, T_enc, _ = out.shape
    attn = LocalSensitiveAttention(config)

    decoder_hidden = torch.randn(B, config.dec_hidden_dim)
    prev_weights = torch.zeros(B, T_enc)
    cum_weights = torch.zeros(B, T_enc)
    attn_weights_cat = torch.stack([prev_weights, cum_weights], dim=1)  # (B, 2, T)

    # mask: True = valid token, False = padding
    mask = (
        torch.arange(T_enc)[None, :] < text_tokens_lengts[:, None]
    ).unsqueeze(-1)  # (B, T, 1) — broadcasts over logits' last dim

    context, weights = attn(decoder_hidden, out, attn_weights_cat, mask=mask)

    print("context:", context.shape, "expected:", (B, config.enc_hidden_dim))
    print("weights:", weights.shape, "expected:", (B, T_enc))
