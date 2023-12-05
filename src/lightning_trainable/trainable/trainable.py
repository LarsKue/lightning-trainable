import lightning
import os
import torch
import warnings

from copy import deepcopy
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import Logger, TensorBoardLogger
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from lightning_trainable import utils
from lightning_trainable.callbacks import LogHParamsCallback
from .trainable_hparams import TrainableHParams

from . import lr_schedulers
from . import optimizers


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
        self.save_hyperparameters(hparams.as_dict())
        # workaround for https://github.com/Lightning-AI/lightning/issues/17889
        self._hparams_name = "hparams"

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def __init_subclass__(cls, **kwargs):
        hparams_type = cls.__annotations__.get("hparams")
        if hparams_type is not None:
            # only overwrite hparams_type if it is defined by the child class
            cls.hparams_type = hparams_type

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
            self.log(f"training/{key}", value,
                     prog_bar=key == self.hparams.loss)

        return metrics[self.hparams.loss]

    def validation_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, batch_idx)
        for key, value in metrics.items():
            self.log(f"validation/{key}", value,
                     prog_bar=key == self.hparams.loss)

    def test_step(self, batch, batch_idx):
        metrics = self.compute_metrics(batch, batch_idx)
        for key, value in metrics.items():
            self.log(f"test/{key}", value,
                     prog_bar=key == self.hparams.loss)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for this model, based on the HParams.
        This method is called automatically by the Lightning Trainer in module fitting.

        By default, we use one optimizer and zero or one learning rate scheduler.
        If you want to use multiple optimizers or learning rate schedulers, you must override this method.

        @return: A dictionary containing the optimizer and learning rate scheduler, if any.
        """
        optimizer = optimizers.configure(self)
        lr_scheduler = lr_schedulers.configure(self, optimizer)

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
        checkpoint_kwargs = deepcopy(self.hparams.model_checkpoint)
        if self.val_data is None:
            monitor = f"training/{self.hparams.loss}"
        else:
            monitor = f"validation/{self.hparams.loss}"
        if "monitor" in checkpoint_kwargs and checkpoint_kwargs["monitor"] == "auto":
            checkpoint_kwargs["monitor"] = monitor
        callbacks = [
            ModelCheckpoint(**checkpoint_kwargs),
            LearningRateMonitor(),
            LogHParamsCallback(),
        ]
        if self.hparams.early_stopping is not None:
            if self.hparams.early_stopping["monitor"] == "auto":
                self.hparams.early_stopping["monitor"] = monitor
            callbacks.append(EarlyStopping(**self.hparams.early_stopping))
        return callbacks

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
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
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
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
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
            pin_memory=auto_pin_memory(self.hparams.pin_memory, self.hparams.accelerator),
            num_workers=self.hparams.num_workers,
        )

    def configure_logger(self, logger_name: str = "TensorBoardLogger", **logger_kwargs) -> Logger:
        """
        Instantiate the Logger used by the Trainer in module fitting.
        By default, we use a TensorBoardLogger, but you can use any other logger of your choice.

        @param logger_name: The name of the logger to use. Defaults to TensorBoardLogger.
        @param logger_kwargs: Keyword-Arguments to the Logger. Set `logger_name` to use a different logger than TensorBoardLogger.
        @return: The Logger object.
        """
        logger_kwargs = logger_kwargs or {}
        logger_kwargs.setdefault("save_dir", os.getcwd())
        logger_class = utils.get_logger(logger_name)
        if issubclass(logger_class, TensorBoardLogger):
            logger_kwargs.setdefault("default_hp_metric", False)

        return logger_class(**logger_kwargs)

    def configure_trainer(self, logger_kwargs: dict = None, trainer_kwargs: dict = None) -> lightning.Trainer:
        """
        Instantiate the Lightning Trainer used to train this module.

        @param logger_kwargs: Keyword-Arguments to the Logger.
            See also :func:`~trainable.Trainable.configure_logger`.
        @param trainer_kwargs: Keyword-Arguments to the Lightning Trainer.
            See also :func:`~trainable.Trainable.configure_trainer`.
        @return: The Lightning Trainer object.
        """
        logger_kwargs = logger_kwargs or {}
        trainer_kwargs = trainer_kwargs or {}

        trainer_kwargs.setdefault("accelerator", self.hparams.accelerator.lower())
        trainer_kwargs.setdefault("accumulate_grad_batches", self.hparams.accumulate_batches)
        trainer_kwargs.setdefault("benchmark", True)
        trainer_kwargs.setdefault("devices", self.hparams.devices)
        trainer_kwargs.setdefault("gradient_clip_val", self.hparams.gradient_clip)
        trainer_kwargs.setdefault("logger", self.configure_logger(**logger_kwargs))
        trainer_kwargs.setdefault("max_epochs", self.hparams.max_epochs)
        trainer_kwargs.setdefault("max_steps", self.hparams.max_steps)
        trainer_kwargs.setdefault("profiler", self.hparams.profiler)

        return lightning.Trainer(**trainer_kwargs)

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
    def fit(self, logger_kwargs: dict = None, trainer_kwargs: dict = None, fit_kwargs: dict = None) -> dict:
        """
        Instantiate a Lightning Trainer and use it to fit the module to data.

        @param logger_kwargs: Keyword-Arguments to the Logger.
            See also :func:`~trainable.Trainable.configure_logger`.
        @param trainer_kwargs: Keyword-Arguments to the Lightning Trainer.
            See also :func:`~trainable.Trainable.configure_trainer`.
        @param fit_kwargs: Keyword-Arguments to the Trainer's fit method.
        @return: Validation Metrics as defined in :func:`~trainable.Trainable.compute_metrics`.
        """
        logger_kwargs = logger_kwargs or {}
        trainer_kwargs = trainer_kwargs or {}
        fit_kwargs = fit_kwargs or {}

        trainer = self.configure_trainer(logger_kwargs, trainer_kwargs)

        trainer.fit(self, **fit_kwargs)

        return {
            key: value.item()
            for key, value in trainer.callback_metrics.items()
            if any(key.startswith(k) for k in ["training/", "validation/"])
        }

    @torch.no_grad()
    def validate(self, logger_kwargs: dict = None, trainer_kwargs: dict = None, validate_kwargs: dict = None):
        logger_kwargs = logger_kwargs or {}
        trainer_kwargs = trainer_kwargs or {}
        validate_kwargs = validate_kwargs or {}

        trainer = self.configure_trainer(logger_kwargs, trainer_kwargs)

        return trainer.validate(self, **validate_kwargs)

    @torch.enable_grad()
    def fit_fast(self, device="cuda"):
        """
        Perform a fast fit using only a simple torch loop.
        This is useful for prototyping, especially with small models.
        Note that this uses minimal features, so LRSchedulers, logging,
        lightning callbacks and other features are not available.

        Performs no validation loop. Returns the final training loss.

        Use at your own risk. You should switch to the full fit() function once you are done prototyping.
        """
        self.train()
        self.to(device)

        maybe_optimizer = self.configure_optimizers()
        if isinstance(maybe_optimizer, dict):
            optimizer = maybe_optimizer["optimizer"]
        elif isinstance(maybe_optimizer, torch.optim.Optimizer):
            optimizer = maybe_optimizer
        else:
            raise RuntimeError("Invalid optimizer")

        dataloader = self.train_dataloader()

        loss = None
        for _epoch in tqdm(range(self.hparams.max_epochs)):
            for batch_idx, batch in enumerate(dataloader):
                batch = self.transfer_batch_to_device(batch, self.device, 0)

                optimizer.zero_grad()
                loss = self.training_step(batch, batch_idx)
                loss.backward()
                optimizer.step()

        return loss

    @classmethod
    def load_best_checkpoint(cls, root: str | Path = "lightning_logs", version: int = "last", metric: str = "validation/loss", dataloader_idx: int = 0):
        root = Path(root)

        if not root.is_dir():
            raise ValueError(f"Checkpoint root directory '{root}' does not exist")

        # get existing version number or error
        version = utils.io.find_version(root, version)

        logs_folder = root / f"version_{version}"
        checkpoint_folder = root / f"version_{version}" / "checkpoints"

        checkpoints = list(checkpoint_folder.glob("*.ckpt"))
        if len(checkpoints) == 0:
            raise FileNotFoundError(f"No checkpoints in '{checkpoint_folder}'")

        best_model = None
        best_metric = None

        # TODO: check via logs instead? (is cheaper, but not portable between loggers)
        for cp in checkpoints:
            model = cls.load_from_checkpoint(cp)
            metrics = model.validate(trainer_kwargs=dict(logger=None))
            metrics = metrics[dataloader_idx]
            if metric not in metrics:
                raise RuntimeError(f"Could not find metric '{metric}' in validation metrics.")

            if best_metric is None or metrics[metric] < best_metric:
                print("Found new best model:", cp, metrics[metric])
                best_metric = metrics[metric]
                best_model = model
            else:
                print("New model was worse:", cp, metrics[metric])

        return best_model


def auto_pin_memory(pin_memory: bool | None, accelerator: str):
    if pin_memory is None:
        return accelerator != "cpu"
    return pin_memory
