
from lightning_trainable.hparams import HParams

from .block_hparams import ConvolutionalBlockHParams


class ConvolutionalNetworkHParams(HParams):
    block_hparams: list[ConvolutionalBlockHParams]
