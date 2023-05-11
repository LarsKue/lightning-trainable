
from torch.optim import Optimizer

from lightning_trainable.utils import get_optimizer

from . import defaults


def with_kwargs(model, **kwargs) -> Optimizer:
    """
    Get an optimizer with the given kwargs
    Insert default values for missing kwargs
    @param model: Trainable model
    @param kwargs: Keyword arguments for the optimizer
    @return: The configured optimizer
    """
    name = kwargs.pop("name")
    default_kwargs = defaults.get_defaults(name, model)
    kwargs = default_kwargs | kwargs
    return get_optimizer(name)(**kwargs)


def configure(model) -> Optimizer | None:
    """
    Configure an optimizer from the model's hparams
    @param model: Trainable model
    @return: The configured optimizer
    """
    match model.hparams.optimizer:
        case str() as name:
            return with_kwargs(model, name=name)
        case dict() as kwargs:
            return with_kwargs(model, **kwargs.copy())
        case None:
            # do not use an optimizer
            return None
        case other:
            raise NotImplementedError(f"Unrecognized Optimizer: '{other}'")
