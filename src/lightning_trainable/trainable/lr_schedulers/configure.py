
from torch.optim.lr_scheduler import LRScheduler

from lightning_trainable.utils import get_scheduler

from . import defaults


def with_kwargs(model, optimizer, **kwargs) -> LRScheduler:
    """
    Get a learning rate scheduler with the given kwargs
    Insert default values for missing kwargs
    @param model: Trainable model
    @param optimizer: The optimizer to use with the scheduler
    @param kwargs: Keyword arguments for the scheduler
    @return: The configured learning rate scheduler
    """
    name = kwargs.pop("name")
    default_kwargs = defaults.get_defaults(name, model, optimizer)
    kwargs = default_kwargs | kwargs

    external_kwargs = {
        key: kwargs.pop(key)
        for key in ["interval", "frequency", "monitor", "strict"]
        if key in kwargs
    }

    return dict(
        scheduler=get_scheduler(name)(optimizer, **kwargs),
        **external_kwargs
    )


def configure(model, optimizer) -> LRScheduler | None:
    """
    Configure a learning rate scheduler from the model's hparams
    @param model: Trainable model
    @param optimizer: The optimizer to use with the scheduler
    @return: The configured learning rate scheduler
    """
    match model.hparams.lr_scheduler:
        case str() as name:
            return with_kwargs(model, optimizer, name=name)
        case dict() as kwargs:
            return with_kwargs(model, optimizer, **kwargs.copy())
        case None:
            # do not use a learning rate scheduler
            return None
        case other:
            raise NotImplementedError(f"Unrecognized Scheduler: '{other}'")
