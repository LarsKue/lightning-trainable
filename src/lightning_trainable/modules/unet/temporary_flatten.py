import torch
import torch.nn as nn


class TemporaryFlatten(nn.Module):
    def __init__(self, inner: nn.Module, input_shape, output_shape):
        super().__init__()
        self.inner = inner
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x.flatten(1))
        return out.reshape(x.shape[0], *self.output_shape)
