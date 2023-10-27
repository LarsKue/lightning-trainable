
import torch.nn as nn
from lightning_trainable.hparams import HParams


class HParamsModule(nn.Module):
    """
    An nn.Module that accepts HParams
    """
    hparams: HParams

    def __init__(self, hparams: HParams | dict):
        super().__init__()
        if not isinstance(hparams, self.hparams_type):
            hparams = self.hparams_type(**hparams)

        self.hparams = hparams

    def __init_subclass__(cls, **kwargs):
        hparams_type = cls.__annotations__.get("hparams")
        if hparams_type is not None:
            # only overwrite hparams_type if it is defined by the child class
            cls.hparams_type = hparams_type
