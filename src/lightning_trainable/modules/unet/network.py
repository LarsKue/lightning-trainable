import torch
import torch.nn as nn

import math

from lightning_trainable.hparams import AttributeDict

from ..convolutional import ConvolutionalBlock
from ..fully_connected import FullyConnectedNetwork
from ..hparams_module import HParamsModule

from .hparams import UNetHParams
from .skip_connection import SkipConnection
from .temporary_flatten import TemporaryFlatten


class UNet(HParamsModule):
    hparams: UNetHParams

    def __init__(self, hparams: dict | UNetHParams):
        super().__init__(hparams)

        self.network = self.configure_network(
            input_shape=self.hparams.input_shape,
            output_shape=self.hparams.output_shape,
            down_blocks=self.hparams.down_blocks,
            up_blocks=self.hparams.up_blocks,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def configure_levels(self, input_shape: (int, int, int), output_shape: (int, int, int), down_blocks: list[dict], up_blocks: list[dict]):
        """
        Recursively configures the levels of the UNet.
        """
        if not down_blocks:
            return self.configure_fc(input_shape, output_shape)

        down_block = down_blocks[0]
        up_block = up_blocks[-1]

        down_hparams = AttributeDict(
            channels=[input_shape[0], *down_block["channels"]],
            kernel_sizes=down_block["kernel_sizes"],
            activation=self.hparams.activation,
            padding="same",
            pool=True,
            pool_direction="down",
            pool_position="last"
        )
        up_hparams = AttributeDict(
            channels=[*up_block["channels"], output_shape[0]],
            kernel_sizes=up_block["kernel_sizes"],
            activation=self.hparams.activation,
            padding="same",
            pool=True,
            pool_direction="up",
            pool_position="first"
        )

        next_input_shape = (down_hparams.channels[-1], input_shape[1] // 2, input_shape[2] // 2)
        next_output_shape = (up_hparams.channels[0], output_shape[1] // 2, output_shape[2] // 2)

        if self.hparams.skip_mode == "concat":
            up_hparams.channels[0] += down_hparams.channels[-1]

        down_block = ConvolutionalBlock(down_hparams)
        up_block = ConvolutionalBlock(up_hparams)

        next_level = self.configure_levels(
            input_shape=next_input_shape,
            output_shape=next_output_shape,
            down_blocks=down_blocks[1:],
            up_blocks=up_blocks[:-1],
        )

        return nn.Sequential(
            down_block,
            SkipConnection(next_level, mode=self.hparams.skip_mode),
            up_block,
        )

    def configure_fc(self, input_shape: (int, int, int), output_shape: (int, int, int)):
        """
        Configures the lowest level of the UNet as a fully connected network.
        """
        hparams = dict(
            input_dims=math.prod(input_shape),
            output_dims=math.prod(output_shape),
            layer_widths=self.hparams.bottom_widths,
            activation=self.hparams.activation,
        )
        return TemporaryFlatten(FullyConnectedNetwork(hparams), input_shape, output_shape)

    def configure_network(self, input_shape: (int, int, int), output_shape: (int, int, int), down_blocks: list[dict], up_blocks: list[dict]):
        """
        Configures the UNet.
        """
        return self.configure_levels(input_shape, output_shape, down_blocks, up_blocks)
