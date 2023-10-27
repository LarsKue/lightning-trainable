
import torch
import torch.nn as nn
from torch.nn import Module

from lightning_trainable.utils import get_activation


class SimpleUNetUpBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, block_size: int = 2, activation: str = "relu"):
        super().__init__()

        self.channels = torch.linspace(in_channels, out_channels, block_size + 1, dtype=torch.int64).tolist()

        layers = []
        for c1, c2 in zip(self.channels[:-2], self.channels[1:-1]):
            layers.append(nn.Conv2d(c1, c2, kernel_size, padding="same"))
            layers.append(get_activation(activation)(inplace=True))

        layers.append(nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size, padding="same"))
        layers.append(nn.ConvTranspose2d(self.channels[-1], self.channels[-1], 2, 2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

    def extra_repr(self) -> str:
        return f"in_channels={self.channels[0]}, out_channels={self.channels[-1]}, kernel_size={self.block[0].kernel_size[0]}, block_size={len(self.channels) - 1}"
