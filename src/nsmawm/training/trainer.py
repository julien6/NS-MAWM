"""PyTorch Lightning trainer for NS-MAWM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from nsmawm.models.nsmawm import NSMAWM
from nsmawm.metrics.rvr import compute_rvr


@dataclass
class TrainOutput:
    loss: float
    loss_symb: float


class NSMAWMLightningModule(pl.LightningModule):
    def __init__(self, model: NSMAWM, learning_rate: float = 1e-3, weight_decay: float = 0.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, loss_symb = self.model.compute_loss(batch["obs"], batch["act"], batch["next_obs"])
        self.log("train_loss", loss)
        self.log("train_loss_symb", loss_symb)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        output = self.model.forward(batch["obs"], batch["act"], apply_projection=True)
        loss = torch.mean((output.prediction - batch["next_obs"]) ** 2)
        rvr = compute_rvr(output.prediction, output.omega_d, output.mask)
        self.log("val_loss", loss)
        self.log("val_rvr", rvr)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


def fit(
    model: NSMAWM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    max_epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    accelerator: str = "cpu",
    devices: int = 1,
    log_every_n_steps: int = 10,
    enable_checkpointing: bool = True,
) -> pl.Trainer:
    module = NSMAWMLightningModule(model, learning_rate=learning_rate, weight_decay=weight_decay)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=enable_checkpointing,
    )
    trainer.fit(module, train_loader, val_loader)
    return trainer
