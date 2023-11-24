
import lightning_trainable.trainable as trainable

from torch.optim import Optimizer
from lightning_trainable.utils import get_scheduler

from . import defaults


def configure(model: "trainable.Trainable", optimizer: Optimizer) -> dict | None:
    """
    Return a config for a learning rate scheduler based on the model's hparams
    @param model: Trainable model
    @param optimizer: The optimizer to use with the scheduler
    @return: The configured learning rate scheduler
    """
    if optimizer is None and model.hparams.lr_scheduler is not None:
        raise ValueError(f"Cannot use a learning rate scheduler without an optimizer. "
                         f"Did you forget to set lr_scheduler=None?")

    match model.hparams.lr_scheduler:
        case str() as name:
            # the user specified only the scheduler name
            # get the default config for that scheduler
            config = defaults.get_config(name)
            kwargs = defaults.get_kwargs(name, model, optimizer)
        case dict() as config:
            # the user specified an additional config for the scheduler
            # fill in the missing values with the default config
            config = config.copy()
            name = config.pop("name")

            # combine user-defined kwargs with defaults
            kwargs = defaults.get_kwargs(name, model, optimizer)
            if "kwargs" in config:
                kwargs = kwargs | config.pop("kwargs")

            # combine user-defined config options with defaults
            config = defaults.get_config(name) | config

        case None:
            # do not use a learning rate scheduler
            return None
        case other:
            raise NotImplementedError(f"Unrecognized Scheduler: '{other}'")

    scheduler = get_scheduler(name)(optimizer=optimizer, **kwargs)

    return dict(
        scheduler=scheduler,
        **config,
    )
