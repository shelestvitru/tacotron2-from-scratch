import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from dataset import LJDataModule
from model import Tacotron2, Tacotron2Config


class Tacotron2Module(L.LightningModule):
    def __init__(self, config: Tacotron2Config, lr: float = 1e-3, weight_decay: float = 1e-6):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = Tacotron2(config)
        self.lr = lr
        self.weight_decay = weight_decay

    def _compute_loss(self, batch, pred):
        mel_target = batch["mel"]                   # (B, T_mel, n_mels)
        mel_mask = batch["mel_mask"].unsqueeze(-1)  # (B, T_mel, 1)
        stop_target = batch["stop_target"]          # (B, T_mel)

        n_valid = mel_mask.sum() * self.config.n_mels

        mel_l1 = (F.l1_loss(pred["mel"], mel_target, reduction="none") * mel_mask).sum() / n_valid
        post_l1 = (F.l1_loss(pred["mel_post"], mel_target, reduction="none") * mel_mask).sum() / n_valid
        stop_loss = F.binary_cross_entropy_with_logits(pred["stop_logits"], stop_target)

        total = mel_l1 + post_l1 + stop_loss
        return total, {"mel_l1": mel_l1, "post_l1": post_l1, "stop": stop_loss, "total": total}

    def training_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss, parts = self._compute_loss(batch, pred)
        self.log_dict({f"train/{k}": v for k, v in parts.items()}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(batch)
        loss, parts = self._compute_loss(batch, pred)
        self.log_dict({f"val/{k}": v for k, v in parts.items()}, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="data/LJSpeech-1.1/train.csv")
    p.add_argument("--val_csv", default="data/LJSpeech-1.1/val.csv")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--ckpt_dir", default="checkpoints/tacotron2")
    p.add_argument("--resume", default=None, help="path to .ckpt to resume from")
    args = p.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    config = Tacotron2Config()
    module = Tacotron2Module(config, lr=args.lr)

    dm = LJDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename="epoch={epoch:03d}-val_total={val/total:.4f}",
        auto_insert_metric_name=False,
        monitor="val/total",
        mode="min",
        save_top_k=-1,          # keep all
        every_n_epochs=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[ckpt_cb, lr_cb],
        gradient_clip_val=args.grad_clip,
        log_every_n_steps=25,
        default_root_dir=args.ckpt_dir,
    )

    trainer.fit(module, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
