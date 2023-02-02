"""
MIT License

Copyright (c) 2023 Lars Erik Kühmichel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import pytorch_lightning as lightning
from pytorch_lightning.profiler import Profiler
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader, Dataset

from pathlib import Path

from .hparams import HParams


class TrainableHParams(HParams):
    accelerator: str = "gpu"
    devices: int = 1
    max_epochs: int | None
    optimizer: str = "adam"
    learning_rate: float | int = 1e-3
    weight_decay: float | int = 0
    batch_size: int
    accumulate_batches: int | None = None
    track_grad_norm: int | None = 2
    gradient_clip: float | int | None = None
    profiler: str | Profiler | None = None


class Trainable(lightning.LightningModule):
    def __init__(self, hparams: TrainableHParams | dict, train_data: Dataset = None, val_data: Dataset = None, test_data: Dataset = None):
        super().__init__()
        if not isinstance(hparams, TrainableHParams):
            hparams = TrainableHParams(**hparams)

        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def loss(self, batch, batch_idx) -> torch.Tensor:
        """ Compute the loss on the given batch """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("validation_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.loss(batch, batch_idx)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        """
        Configure optimizers for Lightning
        """
        lr = self.hparams.learning_rate
        weight_decay = self.hparams.weight_decay
        match self.hparams.optimizer.lower():
            case "adam":
                optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
            case "rmsprop":
                optimizer = torch.optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
            case optimizer:
                raise NotImplementedError(f"Unsupported Optimizer: {optimizer}")

        return optimizer

    def configure_callbacks(self):
        """
        Configure and return train callbacks for Lightning
        """
        return [
            lightning.callbacks.ModelCheckpoint(monitor="validation_loss", save_last=True, every_n_epochs=25, save_top_k=5),
            lightning.callbacks.LearningRateMonitor(),
        ]

    def train_dataloader(self):
        """
        Configure and return the train dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

    def val_dataloader(self):
        """
        Configure and return the validation dataloader
        """
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def test_dataloader(self):
        """
        Configure and return the test dataloader
        """
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

    def configure_trainer(self, save_dir: str | Path = None, **trainer_kwargs):
        """
        Configure and return the Trainer used to train this module
        """
        if save_dir is None:
            logger = True
        else:
            save_path = Path(save_dir)
            version = save_path.name
            experiment_name = save_path.parent.name
            save_dir = save_path.parent.parent
            logger = TensorBoardLogger(save_dir=save_dir, name=experiment_name, version=version)

        return lightning.Trainer(
            accelerator=self.hparams.accelerator.lower(),
            logger=logger,
            devices=self.hparams.devices,
            max_epochs=self.hparams.max_epochs,
            gradient_clip_val=self.hparams.gradient_clip,
            accumulate_grad_batches=self.hparams.accumulate_batches,
            track_grad_norm=self.hparams.track_grad_norm,
            profiler=self.hparams.profiler,
            benchmark=True,
            **trainer_kwargs,
        )

    @torch.enable_grad()
    def fit(self, **trainer_kwargs):
        trainer = self.configure_trainer(**trainer_kwargs)
        trainer.fit(self)

        return trainer.validate(self)[0]["validation_loss"]
