
import torch
from torch import Tensor

from .sinkhorn import sinkhorn_auto


def wasserstein(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 0.1, steps: int = 10) -> Tensor:
    """
    Computes the Wasserstein distance between two distributions.
    See also: <cref>sinkhorn_auto</cref>
    """
    if cost is None:
        cost = x[:, None] - y[None, :]
        cost = torch.flatten(cost, start_dim=2)
        cost = torch.linalg.norm(cost, dim=-1)

    return torch.sum(sinkhorn_auto(x, y, cost, epsilon, steps) * cost)
