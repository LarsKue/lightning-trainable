
from torch import Tensor

import torch.nn as nn

from lightning_trainable.utils import get_activation

from itertools import chain, zip_longest

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

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_network(self):
        widths = [self.hparams.input_dims, *self.hparams.layer_widths, self.hparams.output_dims]

        windows = list(zip(widths[:-1], widths[1:]))

        linears = [self.linear(in_features, out_features) for (in_features, out_features) in windows]

        w = windows if self.hparams.last_layer_activation else windows[:-1]
        activations = [self.activation() for _ in w]
        w = windows if self.hparams.last_layer_dropout else windows[:-1]
        dropouts = [self.dropout() for _ in w]
        w = windows if self.hparams.last_layer_norm else windows[:-1]
        norms = [self.norm(out_features) for (_, out_features) in w]

        layers = zip_longest(linears, activations, dropouts, norms)
        layers = chain.from_iterable(layers)
        layers = filter(lambda x: x is not None, layers)
        layers = list(layers)

        return nn.Sequential(*layers)

    def linear(self, in_features, out_features) -> nn.Module:
        match in_features:
            case "lazy":
                return nn.LazyLinear(out_features)
            case int() as in_features:
                return nn.Linear(in_features, out_features)
            case other:
                raise NotImplementedError(f"Unrecognized input_dims value: '{other}'")

    def activation(self) -> nn.Module:
        return get_activation(self.hparams.activation)(inplace=True)

    def dropout(self) -> nn.Module | None:
        if self.hparams.dropout > 0:
            return nn.Dropout(self.hparams.dropout)

    def norm(self, num_features) -> nn.Module | None:
        match self.hparams.norm:
            case "none":
                return None
            case "batch":
                return nn.BatchNorm1d(num_features)
            case "layer":
                return nn.LayerNorm(num_features)
            case other:
                raise NotImplementedError(f"Unrecognized norm value: '{other}'")
