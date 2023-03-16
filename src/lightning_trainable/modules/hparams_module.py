
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
        cls.hparams_type = cls.__annotations__["hparams"]
