
import torch
from torch import Tensor


def sinkhorn(a: Tensor, b: Tensor, cost: Tensor, epsilon: float = 0.1, steps: int = 100) -> Tensor:
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

    # Initialize the dual variables.
    u = torch.ones(len(a))
    v = torch.ones(len(b))

    # Compute the Sinkhorn iterations.
    for _ in range(steps):
        v = b / (torch.matmul(gain.T, u) + 1e-50)
        u = a / (torch.matmul(gain, v) + 1e-50)

    # Return the transport plan.
    return u[:, None] * gain * v[None, :]


def sinkhorn_auto(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 0.1, steps: int = 100) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from samples from two distributions.
    See also: <cref>sinkhorn</cref>
    @param x: Samples from the first distribution in shape (n, ...).
    @param y: Samples from the second distribution in shape (m, ...).
    @param cost: Optional cost matrix in shape (n, m). If not provided, the Euclidean distance is used.
    @param epsilon: Entropic regularization parameter.
    @param steps: Number of iterations.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Expected x and y to live in the same feature space, "
                         f"but got {x.shape[1:]} and {y.shape[1:]}.")
    if cost is None:
        cost = x[:, None] - y[None, :]
        cost = torch.flatten(cost, start_dim=2)
        cost = torch.linalg.norm(cost, dim=-1)

    # Initialize the sample weights.
    a = torch.ones(len(x)) / len(x)
    b = torch.ones(len(y)) / len(y)

    return sinkhorn(a, b, cost, epsilon, steps)


def wasserstein(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 0.1, steps: int = 100) -> Tensor:
    """
    Computes the Wasserstein distance between two distributions.
    See also: <cref>sinkhorn_auto</cref>
    """
    return torch.sum(sinkhorn_auto(x, y, cost, epsilon, steps) * cost)
