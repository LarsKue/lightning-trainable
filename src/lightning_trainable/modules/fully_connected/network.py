import torch
import torch.nn as nn

from lightning_trainable.utils import get_activation

from ..hparams_module import HParamsModule
from ..sequential_mixin import SequentialMixin
from .hparams import FullyConnectedNetworkHParams


class FullyConnectedNetwork(SequentialMixin, HParamsModule):
    """
    Implements a series of fully connected layers, each followed by an activation function.
    """
    hparams: FullyConnectedNetworkHParams

    def __init__(self, hparams):
        super().__init__(hparams)

        self.network = self.configure_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_network(self):
        layers = []
        widths = [self.hparams.input_dims, *self.hparams.layer_widths, self.hparams.output_dims]

        if self.hparams.input_dims == "lazy":
            widths = widths[1:]
            layers = [nn.LazyLinear(widths[0]), get_activation(self.hparams.activation)(inplace=True)]

        for (w1, w2) in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(get_activation(self.hparams.activation)(inplace=True))

        # remove last activation
        layers = layers[:-1]

        return nn.Sequential(*layers)
