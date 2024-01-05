
import lightning_trainable.trainable as trainable

from torch.optim import Optimizer
from lightning_trainable.utils import get_optimizer

from . import defaults


def configure(model: "trainable.Trainable") -> Optimizer | None:
    """
    Configure an optimizer from the model's hparams
    @param model: Trainable model
    @return: The configured optimizer, if any
    """
    match model.hparams.optimizer:
        case str() as name:
            kwargs = defaults.get_kwargs(name, model)
        case dict() as config:
            config = config.copy()
            name = config.pop("name")
            kwargs = config.pop("kwargs", dict())
            kwargs = defaults.get_kwargs(name, model) | kwargs
        case None:
            # do not use an optimizer
            return None
        case other:
            raise NotImplementedError(f"Unrecognized Optimizer: '{other}'")

    return get_optimizer(name)(**kwargs)
