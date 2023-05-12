
import torch
from torch import Tensor


def plan(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: int | float = 0.1, steps: int = 100) -> Tensor:
    """
    Compute the optimal transport plan pi between two distributions.
    @param x: Samples from the first distribution.
    @param y: Samples from the second distribution.
    @param cost: Optional cost matrix. If not provided, the L2 distance is used.
    @param epsilon: The entropic regularization parameter.
    @param steps: The number of Sinkhorn iterations.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"x and y must have the same feature dimensions, but got {x.shape[1:]} and {y.shape[1:]}.")

    if cost is None:
        cost = x[:, None] - y[None, :]
        cost = torch.flatten(cost, start_dim=2)
        cost = torch.linalg.norm(cost, dim=-1)
    elif cost.shape != (x.shape[0], y.shape[0]):
        raise ValueError(f"Expected cost matrix of shape {(x.shape[0], y.shape[0])}, but got {cost.shape}.")

    u = torch.zeros(x.shape[0], device=x.device)
    v = torch.zeros(y.shape[0], device=y.device)

    for step in range(steps):
        u = epsilon * torch.logsumexp(-cost + v[None, :] / epsilon, dim=1)
        v = epsilon * torch.logsumexp(-cost + u[:, None] / epsilon, dim=0)

    pi = torch.exp(-(cost + u[:, None] + v[None, :]) / epsilon)

    return pi


def wasserstein(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: int | float = 0.1, steps: int = 100) -> Tensor:
    """
    Compute the Wasserstein distance between two distributions. See <cref>plan</cref> for parameter descriptions.
    """
    pi = plan(x, y, cost, epsilon, steps)

    w = torch.sum(pi * cost)

    return w
