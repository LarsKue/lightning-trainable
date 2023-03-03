from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import TensorDataset, Dataset

from lightning_trainable import Trainable, TrainableHParams


def test_instantiate():
    hparams = TrainableHParams(max_epochs=10, batch_size=32)
    Trainable(hparams)


def test_simple_model():
    class SimpleTrainable(Trainable):
        def __init__(self, hparams: TrainableHParams | dict,
                     log_dir: Path | str = "lightning_logs",
                     train_data: Dataset = None,
                     val_data: Dataset = None,
                     test_data: Dataset = None
                     ):
            super().__init__(hparams, log_dir, train_data, val_data, test_data)
            self.param = torch.nn.Parameter(torch.randn(8, 1))

        def compute_metrics(self, batch, batch_idx) -> dict:
            return {
                "loss": ((batch[0] @ self.param) ** 2).mean()
            }

    hparams = TrainableHParams(
        accelerator="cpu",
        max_epochs=10,
        batch_size=32,
        lr_scheduler="1cycle"
    )
    model = SimpleTrainable(hparams, train_data=TensorDataset(torch.randn(128, 8)))
    model.fit()
