import torch
import torch.nn as nn


from ..hparams_module import HParamsModule
from ..sequential_mixin import SequentialMixin

from .block import ConvolutionalBlock
from .hparams import ConvolutionalNetworkHParams


class ConvolutionalNetwork(SequentialMixin, HParamsModule):
    """
    Implements a series of pooled convolutional blocks.
    """
    hparams: ConvolutionalNetworkHParams

    def __init__(self, hparams):
        super().__init__(hparams)
        self.network = self.configure_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_network(self):
        blocks = [ConvolutionalBlock(hparams) for hparams in self.hparams.block_hparams]

        return nn.Sequential(*blocks)
