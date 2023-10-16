
from torch import Tensor

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

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

    def configure_network(self):
        widths = self.hparams.layer_widths

        # input layer
        input_linear = self.configure_linear(self.hparams.input_dims, widths[0])
        input_activation = self.configure_activation()
        layers = [input_linear, input_activation]

        # hidden layers
        for (in_features, out_features) in zip(widths[:-1], widths[1:]):
            dropout = self.configure_dropout()
            norm = self.configure_norm(in_features)

            activation = self.configure_activation()
            linear = self.configure_linear(in_features, out_features)

            if dropout is not None:
                layers.append(dropout)

            if norm is not None:
                layers.append(norm)

            layers.append(linear)
            layers.append(activation)

        # output layer
        output_linear = self.configure_linear(widths[-1], self.hparams.output_dims)
        layers.append(output_linear)

        return nn.Sequential(*layers)

    def configure_linear(self, in_features, out_features) -> nn.Module:
        match in_features:
            case "lazy":
                return nn.LazyLinear(out_features)
            case int() as in_features:
                return nn.Linear(in_features, out_features)
            case other:
                raise NotImplementedError(f"Unrecognized input_dims value: '{other}'")

    def configure_activation(self) -> nn.Module:
        return get_activation(self.hparams.activation)(inplace=True)

    def configure_dropout(self) -> nn.Module | None:
        if self.hparams.dropout > 0:
            return nn.Dropout(self.hparams.dropout)

    def configure_norm(self, num_features) -> nn.Module | None:
        match self.hparams.norm:
            case "none":
                return None
            case "batch":
                return nn.BatchNorm1d(num_features)
            case "layer":
                return nn.LayerNorm(num_features)
            case other:
                raise NotImplementedError(f"Unrecognized norm value: '{other}'")
