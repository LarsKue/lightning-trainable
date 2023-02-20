
from .hparams import HParams

import pytorch_lightning as lightning
from pytorch_lightning.profiler import Profiler
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from pathlib import Path

from .utils import EpochProgressBar


class TrainableHParams(HParams):
    # name of the loss, your `compute_metrics` should return a dict with this name in its keys
    loss: str = "loss"

    accelerator: str = "gpu"
    devices: int = 1
    max_epochs: int | None
    max_steps: int = -1
    optimizer: str = "adam"
    learning_rate: float | int = 1e-3
    weight_decay: float | int = 0
    batch_size: int
    accumulate_batches: int | None = None
    track_grad_norm: int | None = 2
    gradient_clip: float | int | None = None
    profiler: str | Profiler | None = None


class Trainable(lightning.LightningModule):
    def __init__(
            self,
            hparams: TrainableHParams | dict,
            log_dir: Path | str = "lightning_logs",
            train_data: Dataset = None,
            val_data: Dataset = None,
            test_data: Dataset = None
    ):
        super().__init__()
        if not isinstance(hparams, TrainableHParams):
            hparams = TrainableHParams(**hparams)
        self.save_hyperparameters(hparams)

        self.log_dir = Path(log_dir)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def compute_metrics(self, batch, batch_idx) -> dict:
        """ Compute any relevant metrics, including the loss, on the given batch """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, batch_idx)
        if self.hparams.loss not in metrics:
            raise RuntimeError(f"You must return the loss '{self.hparams.loss}' from `compute_metrics`.")

        for key, value in metrics.items():
            self.log(f"training/{key}", value)

        return metrics[self.hparams.loss]

    def validation_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, batch_idx)
        for key, value in metrics.items():
            self.log(f"validation/{key}", value)

    def test_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, batch_idx)
        for key, value in metrics.items():
            self.log(f"test/{key}", value)

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
            lightning.callbacks.ModelCheckpoint(
                monitor=f"validation/{self.hparams.loss}",
                save_last=True,
                every_n_epochs=25,
                save_top_k=5
            ),
            lightning.callbacks.LearningRateMonitor(),
        ]

    def train_dataloader(self):
        """
        Configure and return the train dataloader
        """
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=not isinstance(self.train_data, IterableDataset),
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

    def configure_logger(self, **kwargs):
        """
        Configure and return the Logger to be used by the Lightning.Trainer
        """
        return TensorBoardLogger(
            save_dir=self.log_dir,
            **kwargs
        )

    def configure_trainer(self, logger_kwargs: dict = None, trainer_kwargs: dict = None):
        """
        Configure and return the Trainer used to train this module
        """
        if logger_kwargs is None:
            logger_kwargs = dict()
        if trainer_kwargs is None:
            trainer_kwargs = dict()

        return lightning.Trainer(
            accelerator=self.hparams.accelerator.lower(),
            logger=self.configure_logger(**logger_kwargs),
            devices=self.hparams.devices,
            max_epochs=self.hparams.max_epochs,
            max_steps=self.hparams.max_steps,
            gradient_clip_val=self.hparams.gradient_clip,
            accumulate_grad_batches=self.hparams.accumulate_batches,
            track_grad_norm=self.hparams.track_grad_norm,
            profiler=self.hparams.profiler,
            benchmark=True,
            callbacks=EpochProgressBar(),
            **trainer_kwargs,
        )

    @torch.enable_grad()
    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None) -> dict:
        """ Fit the module to data and return validation metrics """
        if logger_kwargs is None:
            logger_kwargs = dict()
        if trainer_kwargs is None:
            trainer_kwargs = dict()

        trainer = self.configure_trainer(logger_kwargs, trainer_kwargs)
        trainer.fit(self)

        return trainer.validate(self)[0]
