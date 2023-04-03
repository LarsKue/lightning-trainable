import os

import lightning

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ProgressBar
from lightning.pytorch.loggers import Logger, TensorBoardLogger

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from lightning_trainable import utils
from lightning_trainable.callbacks import EpochProgressBar

from .trainable_hparams import TrainableHParams


class SkipBatch(Exception):
    pass


class Trainable(lightning.LightningModule):
    hparams_type = TrainableHParams
    hparams: TrainableHParams

    def __init__(
            self,
            hparams: TrainableHParams | dict,
            train_data: Dataset = None,
            val_data: Dataset = None,
            test_data: Dataset = None
    ):
        super().__init__()
        if not isinstance(hparams, self.hparams_type):
            hparams = self.hparams_type(**hparams)
        self.save_hyperparameters(hparams)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def __init_subclass__(cls, **kwargs):
        cls.hparams_type = cls.__annotations__.get("hparams", TrainableHParams)

    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Compute any relevant metrics, including the loss, on the given batch.
        You should return a dict in the style of {metric_name: metric_value} from this method,
        where metric_value is scalar. The loss as defined by your hparams must also be a key
        of this dictionary.

        You *must* override this method and you *must* return the loss as defined by your hparams
        within the dictionary, if you want to use :func:`trainable.Trainable.fit`

        @param batch: The batch to compute metrics on. Usually a Tensor or a tuple of Tensors.
        @param batch_idx: Index of this batch.
        @return: Dictionary containing metrics to log and the loss to perform backpropagation on.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        try:
            metrics = self.compute_metrics(batch, batch_idx)
        except SkipBatch:
            return None

        if metrics is None:
            raise RuntimeError("If you want to skip a batch, raise `SkipBatch` instead of returning None.")

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

    def configure_lr_schedulers(self, optimizer):
        """
        Configure the LR Scheduler as defined in HParams.
        By default, we only use a single LR Scheduler, attached to a single optimizer.
        You can use a ChainedScheduler if you need multiple LR Schedulers throughout training,
        or override this method if you need different schedulers for different parameters.

        @param optimizer: The optimizer to attach the scheduler to.
        @return: The LR Scheduler object.
        """
        match self.hparams.lr_scheduler:
            case str() as name:
                match name.lower():
                    case "onecyclelr":
                        kwargs = dict(
                            max_lr=optimizer.defaults["lr"],
                            epochs=self.hparams.max_epochs,
                            steps_per_epoch=len(self.train_dataloader())
                        )
                        interval = "step"
                    case _:
                        kwargs = dict()
                        interval = "step"
                scheduler = utils.get_scheduler(name)(optimizer, **kwargs)
                return dict(
                    scheduler=scheduler,
                    interval=interval,
                )
            case dict() as kwargs:
                name = kwargs.pop("name")
                interval = "step"
                if "interval" in kwargs:
                    interval = kwargs.pop("interval")
                scheduler = utils.get_scheduler(name)(optimizer, **kwargs)
                return dict(
                    scheduler=scheduler,
                    interval=interval,
                )
            case type(torch.optim.lr_scheduler.LRScheduler) as Scheduler:
                kwargs = dict()
                interval = "step"
                scheduler = Scheduler(optimizer, **kwargs)
                return dict(
                    scheduler=scheduler,
                    interval=interval,
                )
            case (torch.optim.lr_scheduler.LRScheduler() | torch.optim.lr_scheduler.ReduceLROnPlateau()) as scheduler:
                return dict(
                    scheduler=scheduler,
                    interval="step",
                )
            case None:
                # do not use a scheduler
                return None
            case other:
                raise NotImplementedError(f"Unrecognized Scheduler: {other}")

    def configure_optimizers(self):
        """
        Configure Optimizer and LR Scheduler objects as defined in HParams.
        By default, we only use a single optimizer and an optional LR Scheduler.
        If you need multiple optimizers, override this method.

        @return: A dictionary containing the optimizer and lr_scheduler.
        """
        kwargs = dict()

        match self.hparams.optimizer:
            case str() as name:
                optimizer = utils.get_optimizer(name)(self.parameters(), **kwargs)
            case dict() as kwargs:
                name = kwargs.pop("name")
                optimizer = utils.get_optimizer(name)(self.parameters(), **kwargs)
            case type(torch.optim.Optimizer) as Optimizer:
                optimizer = Optimizer(self.parameters(), **kwargs)
            case torch.optim.Optimizer() as optimizer:
                pass
            case None:
                return None
            case other:
                raise NotImplementedError(f"Unrecognized Optimizer: {other}")

        lr_scheduler = self.configure_lr_schedulers(optimizer)

        if lr_scheduler is None:
            return optimizer

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def configure_callbacks(self) -> list:
        """
        Configure train callbacks used by the Lightning Trainer in module fitting.
        We provide some useful defaults here, but you may opt to override this method if you want different
        callbacks. Callbacks defined here override those provided directly to the Lightning Trainer object.

        @return: A list of train callbacks.
        """
        if self.val_data is None:
            monitor = f"training/{self.hparams.loss}"
        else:
            monitor = f"validation/{self.hparams.loss}"
        return [
            ModelCheckpoint(
                monitor=monitor,
                save_last=True,
                every_n_epochs=25,
                save_top_k=5
            ),
            LearningRateMonitor(),
            EpochProgressBar(),
        ]

    def train_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Train DataLoader for Lightning. Uses the dataset you passed as train_data.

        @return: The DataLoader Object.
        """
        if self.train_data is None:
            return []
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.batch_size,
            shuffle=not isinstance(self.train_data, IterableDataset),
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Validation DataLoader for Lightning. Uses the dataset you passed as val_data.

        @return: The DataLoader Object.
        """
        if self.val_data is None:
            return []
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader | list[DataLoader]:
        """
        Configures the Test DataLoader for Lightning. Uses the dataset you passed as test_data.

        @return: The DataLoader Object.
        """
        if self.test_data is None:
            return []
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            pin_memory=self.hparams.pin_memory,
            num_workers=self.hparams.num_workers,
        )

    def configure_logger(self, save_dir=os.getcwd(), **kwargs) -> Logger:
        """
        Instantiate the Logger used by the Trainer in module fitting.
        By default, we use a TensorBoardLogger, but you can use any other logger of your choice.

        @param root_dir: The root directory in which all your experiments with
            different names and versions will be stored.
        @param kwargs: Keyword-Arguments to the Logger.
        @return: The Logger object.
        """
        return TensorBoardLogger(
            save_dir=save_dir,
            default_hp_metric=False,
            **kwargs
        )

    def configure_trainer(self, logger_kwargs: dict = None, trainer_kwargs: dict = None) -> lightning.Trainer:
        """
        Instantiate the Lightning Trainer used to train this module.

        @param logger_kwargs: Keyword-Arguments to the Logger.
            See also :func:`~trainable.Trainable.configure_logger`.
        @param trainer_kwargs: Keyword-Arguments to the Lightning Trainer.
            See also :func:`~trainable.Trainable.configure_trainer`.
        @return: The Lightning Trainer object.
        """
        if logger_kwargs is None:
            logger_kwargs = dict()
        if trainer_kwargs is None:
            trainer_kwargs = dict()

        if "enable_progress_bar" not in trainer_kwargs and any(isinstance(callback, ProgressBar) for callback in self.configure_callbacks()):
            trainer_kwargs["enable_progress_bar"] = False

        return lightning.Trainer(
            accelerator=self.hparams.accelerator.lower(),
            logger=self.configure_logger(**logger_kwargs),
            devices=self.hparams.devices,
            max_epochs=self.hparams.max_epochs,
            max_steps=self.hparams.max_steps,
            gradient_clip_val=self.hparams.gradient_clip,
            accumulate_grad_batches=self.hparams.accumulate_batches,
            profiler=self.hparams.profiler,
            benchmark=True,
            **trainer_kwargs,
        )

    def on_before_optimizer_step(self, optimizer):
        # who doesn't love breaking changes in underlying libraries
        match self.hparams.track_grad_norm:
            case int() as norm_type:
                grad_norm = lightning.pytorch.utilities.grad_norm(self, norm_type=norm_type)
                self.log_dict(grad_norm)
            case None:
                pass
            case other:
                raise NotImplementedError(f"Unrecognized grad norm: {other}")

    @torch.enable_grad()
    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None) -> dict:
        """
        Instantiate a Lightning Trainer and use it to fit the module to data.

        @param logger_kwargs: Keyword-Arguments to the Logger.
            See also :func:`~trainable.Trainable.configure_logger`.
        @param trainer_kwargs: Keyword-Arguments to the Lightning Trainer.
            See also :func:`~trainable.Trainable.configure_trainer`.
        @return: Validation Metrics as defined in :func:`~trainable.Trainable.compute_metrics`.
        """
        if logger_kwargs is None:
            logger_kwargs = dict()
        if trainer_kwargs is None:
            trainer_kwargs = dict()

        trainer = self.configure_trainer(logger_kwargs, trainer_kwargs)
        metrics_list = trainer.validate(self)
        if metrics_list is not None and len(metrics_list) > 0:
            metrics = metrics_list[0]
        else:
            metrics = {}
        trainer.logger.log_hyperparams(self.hparams, metrics)
        trainer.fit(self)

        return {
            key: value.item()
            for key, value in trainer.callback_metrics.items()
            if any(key.startswith(key) for key in ["training/", "validation/"])
        }
