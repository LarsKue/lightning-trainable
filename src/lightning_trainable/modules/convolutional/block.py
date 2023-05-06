import torch
import torch.nn as nn

from itertools import chain

from lightning_trainable.modules import HParamsModule
from lightning_trainable.utils import get_activation

from .block_hparams import ConvolutionalBlockHParams


class ConvolutionalBlock(HParamsModule):
    """
    Implements a series of convolutions, each followed by an activation function.
    """

    hparams: ConvolutionalBlockHParams

    def __init__(self, hparams: dict | ConvolutionalBlockHParams):
        super().__init__(hparams)

        self.network = self.configure_network()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_network(self):
        # construct convolutions
        convolutions = []
        cck = zip(self.hparams.channels[:-1], self.hparams.channels[1:], self.hparams.kernel_sizes)
        for in_channels, out_channels, kernel_size in cck:
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=self.hparams.padding,
                dilation=self.hparams.dilation,
                groups=self.hparams.groups,
                bias=self.hparams.bias,
                padding_mode=self.hparams.padding_mode,
            )

            convolutions.append(conv)

        # construct activations
        activations = []
        for _ in range(len(self.hparams.channels) - 1):
            activations.append(get_activation(self.hparams.activation)(inplace=True))

        # zip longest
        layers = list(chain.from_iterable(zip(convolutions, activations)))

        # add pooling layer if requested
        if self.hparams.pool:
            match self.hparams.pool_direction:
                case "up":
                    if self.hparams.pool_position == "first":
                        channels = self.hparams.channels[0]
                    else:
                        channels = self.hparams.channels[-1]

                    pool = nn.ConvTranspose2d(channels, channels, 2, 2)
                case "down":
                    pool = nn.MaxPool2d(2, 2)
                case _:
                    raise NotImplementedError(f"Unrecognized pool direction '{self.hparams.pool_direction}'.")

            match self.hparams.pool_position:
                case "first":
                    layers.insert(0, pool)
                case "last":
                    layers.append(pool)
                case _:
                    raise NotImplementedError(f"Unrecognized pool position '{self.hparams.pool_position}'.")

        return nn.Sequential(*layers)
