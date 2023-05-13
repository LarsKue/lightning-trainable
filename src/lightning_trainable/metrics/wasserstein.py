
import warnings

import torch
from torch import Tensor


def sinkhorn(a: Tensor, b: Tensor, cost: Tensor, epsilon: float, steps: int = 10) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from sample weights of two distributions.
    @param a: Sample weights from the first distribution in shape (n,)
    @param b: Sample weights from the second distribution in shape (m,)
    @param cost: Cost matrix in shape (n, m).
    @param epsilon: Entropic regularization parameter.
    @param steps: Number of iterations.
    """
    if cost.shape != (len(a), len(b)):
        raise ValueError(f"Expected cost to have shape {(len(a), len(b))}, but got {cost.shape}.")

    gain = torch.exp(-cost / epsilon)

    if gain.mean() < 1e-30:
        warnings.warn(f"Detected low bandwidth ({epsilon:.1e}) relative to cost ({cost.mean().item():.1e}). "
                      f"You may experience numerical instabilities. Consider increasing epsilon.")

    # Initialize the dual variables.
    u = torch.ones(len(a), dtype=a.dtype, device=a.device)
    v = torch.ones(len(b), dtype=b.dtype, device=b.device)

    # Compute the Sinkhorn iterations.
    for _ in range(steps):
        v = b / (torch.matmul(gain.T, u) + 1e-50)
        u = a / (torch.matmul(gain, v) + 1e-50)

    # Return the transport plan.
    return u[:, None] * gain * v[None, :]


def sinkhorn_auto(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = None, steps: int = 10) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from samples from two distributions.
    See also: <cref>sinkhorn</cref>
    @param x: Samples from the first distribution in shape (n, ...).
    @param y: Samples from the second distribution in shape (m, ...).
    @param cost: Optional cost matrix in shape (n, m).
        If not provided, the Euclidean distance is used.
    @param epsilon: Optional entropic regularization parameter.
        If not provided, the half-mean of the cost matrix is used.
    @param steps: Number of iterations.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Expected x and y to live in the same feature space, "
                         f"but got {x.shape[1:]} and {y.shape[1:]}.")
    if cost is None:
        cost = x[:, None] - y[None, :]
        cost = torch.flatten(cost, start_dim=2)
        cost = torch.linalg.norm(cost, dim=-1)

    if epsilon is None:
        epsilon = cost.mean() / 2

    # Initialize the sample weights.
    a = torch.ones(len(x), dtype=x.dtype, device=x.device) / len(x)
    b = torch.ones(len(y), dtype=y.dtype, device=y.device) / len(y)

    return sinkhorn(a, b, cost, epsilon, steps)


def wasserstein(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 0.1, steps: int = 10) -> Tensor:
    """
    Computes the Wasserstein distance between two distributions.
    See also: <cref>sinkhorn_auto</cref>
    """
    # TODO: fix for cost = None
    return torch.sum(sinkhorn_auto(x, y, cost, epsilon, steps) * cost)
