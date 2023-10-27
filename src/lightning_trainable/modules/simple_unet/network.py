
import torch
import torch.nn as nn
from torch import Tensor

from ..fully_connected import FullyConnectedNetwork
from ..hparams_module import HParamsModule

from .down_block import SimpleUNetDownBlock
from .hparams import SimpleUNetHParams
from .up_block import SimpleUNetUpBlock


class SimpleUNet(HParamsModule):
    hparams: SimpleUNetHParams

    def __init__(self, hparams: dict | SimpleUNetHParams):
        super().__init__(hparams)

        channels = [self.hparams.input_shape[0], *self.hparams.channels]

        self.down_blocks = nn.ModuleList([
            SimpleUNetDownBlock(c1, c2, kernel_size, self.hparams.block_size, self.hparams.activation)
            for c1, c2, kernel_size in zip(channels[:-1], channels[1:], self.hparams.kernel_sizes)
        ])

        fc_channels = self.hparams.channels[-1]
        height = self.hparams.input_shape[1] // 2 ** (len(channels) - 1)
        width = self.hparams.input_shape[2] // 2 ** (len(channels) - 1)

        fc_hparams = dict(
            input_dims=self.hparams.conditions + fc_channels * height * width,
            output_dims=fc_channels * height * width,
            activation=self.hparams.activation,
            layer_widths=self.hparams.fc_widths,
        )
        self.fc = FullyConnectedNetwork(fc_hparams)
        self.up_blocks = nn.ModuleList([
            SimpleUNetUpBlock(c2, c1, kernel_size, self.hparams.block_size, self.hparams.activation)
            for c1, c2, kernel_size in zip(channels[:-1], channels[1:], self.hparams.kernel_sizes)
        ][::-1])

    def forward(self, image: Tensor, condition: Tensor = None) -> Tensor:
        residuals = []
        for block in self.down_blocks:
            image = block(image)
            residuals.append(image)

        shape = image.shape
        image = image.flatten(start_dim=1)

        if condition is not None:
            image = torch.cat([image, condition], dim=1)

        image = self.fc(image)

        image = image.reshape(shape)

        for block in self.up_blocks:
            residual = residuals.pop()
            image = block(image + residual)

        return image

    def down(self, image: Tensor) -> Tensor:
        for block in self.down_blocks:
            image = block(image)
        return image

    def up(self, image: Tensor) -> Tensor:
        for block in self.up_blocks:
            image = block(image)
        return image
