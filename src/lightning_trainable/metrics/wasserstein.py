
import torch
from torch import Tensor

from .sinkhorn import sinkhorn


def _euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    cost = x[:, None] - y[None, :]
    cost = torch.flatten(cost, start_dim=2)
    cost = torch.linalg.norm(cost, dim=-1)

    return cost


def wasserstein(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 1.0, steps: int = 1000) -> Tensor:
    """
    Computes the Wasserstein distance between two distributions.
    See also: <cref>sinkhorn_auto</cref>
    """
    if cost is None:
        cost = _euclidean_distance(x, y)

    return torch.sum(sinkhorn(x, y, cost, epsilon, steps) * cost)
