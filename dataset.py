import torch
import pandas as pd
import lightning as L
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


CHARS = " !\"'(),-.:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzàâèéêü’“”"

MEL_MIN, MEL_MAX = -11.0, 1.0


def normalize_mel(mel):
    mel = mel.clamp(MEL_MIN, MEL_MAX)
    return 8.0 * (mel - MEL_MIN) / (MEL_MAX - MEL_MIN) - 4.0


def denormalize_mel(mel_norm):
    return (mel_norm + 4.0) * (MEL_MAX - MEL_MIN) / 8.0 + MEL_MIN


class CharTokenizer:
    def __init__(self, chars=CHARS):
        self.pad_token = 0
        self.chars = chars
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars) + 1  # +1 for pad

    def encode(self, text):
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices):
        return "".join(
            self.idx_to_char.get(i, "") for i in indices if i != self.pad_token
        )


class LJDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.tokenizer = CharTokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mel_path = row["filepath"].replace("wavs", "mels").replace(".wav", ".pt")
        mel = torch.load(mel_path, weights_only=True)
        mel = normalize_mel(mel)
        mel = mel.transpose(0, 1)

        tokens = torch.tensor(self.tokenizer.encode(row["text_norm"]), dtype=torch.long)

        return {"text": tokens, "mel": mel}


def ljcollate(batch):
    text_lengths = torch.tensor([item["text"].size(0) for item in batch])
    mel_lengths = torch.tensor([item["mel"].size(0) for item in batch])

    text_padded = pad_sequence(
        [item["text"] for item in batch], batch_first=True, padding_value=0
    )
    mel_padded = pad_sequence(
        [item["mel"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )  # (B, max_time, n_mels)

    T_mel = mel_padded.size(1)
    mel_mask = torch.arange(T_mel).unsqueeze(0) < mel_lengths.unsqueeze(1)  # (B, T_mel)

    stop_target = torch.zeros(len(batch), T_mel)
    for i, L in enumerate(mel_lengths.tolist()):
        stop_target[i, L - 1 :] = 1.0

    return {
        "text": text_padded,
        "text_lengths": text_lengths,
        "mel": mel_padded,
        "mel_lengths": mel_lengths,
        "mel_mask": mel_mask,
        "stop_target": stop_target,
    }


class LJDataModule(L.LightningDataModule):
    def __init__(self, train_csv, val_csv, batch_size=16, num_workers=4):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = CharTokenizer()

    def setup(self, stage=None):
        self.train_ds = LJDataset(self.train_csv)
        self.val_ds = LJDataset(self.val_csv)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=ljcollate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=ljcollate,
        )


if __name__ == "__main__":
    lj = LJDataset("data/LJSpeech-1.1/val.csv")

    item = lj[0]
    for k, v in item.items():
        print(k, v.shape)

    mel = item["mel"]
    print(mel.min(), mel.max())
