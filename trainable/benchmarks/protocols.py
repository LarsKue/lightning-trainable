
import torch
from typing import Protocol


class Distribution(Protocol):
    def sample(self, sample_shape: torch.Size | tuple) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

