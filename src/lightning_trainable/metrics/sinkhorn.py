import warnings

import torch
from torch import Tensor

import torch.nn.functional as F
import numpy as np


def sinkhorn(a: Tensor, b: Tensor, cost: Tensor, epsilon: float, steps: int = 1000) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from sample weights of two distributions.
    This version does not use log-space computations, but allows for zero or negative weights.

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
                      f"You may experience numerical instabilities. Consider increasing epsilon or using sinkhorn_log.")

    # Initialize the dual variables.
    u = torch.ones(len(a), dtype=a.dtype, device=a.device)
    v = torch.ones(len(b), dtype=b.dtype, device=b.device)

    # Compute the Sinkhorn iterations.
    for _ in range(steps):
        v = b / (torch.matmul(gain.T, u) + 1e-50)
        u = a / (torch.matmul(gain, v) + 1e-50)

    # Return the transport plan.
    return u[:, None] * gain * v[None, :]


def sinkhorn_log(log_a: Tensor, log_b: Tensor, cost: Tensor, epsilon: float, steps: int = 1000) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from sample weights of two distributions.
    This version uses log-space computations to avoid numerical instabilities, but disallows zero or negative weights.

    @param log_a: Log sample weights from the first distribution in shape (n,)
    @param log_b: Log sample weights from the second distribution in shape (m,)
    @param cost: Cost matrix in shape (n, m).
    @param epsilon: Entropic regularization parameter.
    @param steps: Number of iterations.
    """
    if cost.shape != (len(log_a), len(log_b)):
        raise ValueError(f"Expected cost to have shape {(len(log_a), len(log_b))}, but got {cost.shape}.")

    log_gain = -cost / epsilon

    # Initialize the dual variables.
    log_u = torch.zeros(len(log_a), dtype=log_a.dtype, device=log_a.device)
    log_v = torch.zeros(len(log_b), dtype=log_b.dtype, device=log_b.device)

    # Compute the Sinkhorn iterations.
    for _ in range(steps):
        log_v = log_b - torch.logsumexp(log_gain + log_u[:, None], dim=0)
        log_u = log_a - torch.logsumexp(log_gain + log_v[None, :], dim=1)

    plan = torch.exp(log_u[:, None] + log_gain + log_v[None, :])

    if not torch.allclose(len(log_b) * plan.sum(dim=0), torch.ones(len(log_b), device=plan.device)) or not torch.allclose(len(log_a) * plan.sum(dim=1), torch.ones(len(log_a), device=plan.device)):
        warnings.warn(f"Sinkhorn did not converge. Consider increasing epsilon or number of iterations.")

    # Return the transport plan.
    return plan


def sinkhorn_auto(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 1.0, steps: int = 1000) -> Tensor:
    """
    Computes the Sinkhorn optimal transport plan from samples from two distributions.
    See also: <cref>sinkhorn_log</cref>

    @param x: Samples from the first distribution in shape (n, ...).
    @param y: Samples from the second distribution in shape (m, ...).
    @param cost: Optional cost matrix in shape (n, m).
        If not provided, the Euclidean distance is used.
    @param epsilon: Optional entropic regularization parameter.
        This parameter is normalized to the half-mean of the cost matrix. This helps keep the value independent
        of the data dimensionality. Note that this behaviour is exclusive to this method; sinkhorn_log only accepts
        the raw entropic regularization value.
    @param steps: Number of iterations.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Expected x and y to live in the same feature space, "
                         f"but got {x.shape[1:]} and {y.shape[1:]}.")
    if cost is None:
        cost = x[:, None] - y[None, :]
        cost = torch.flatten(cost, start_dim=2)
        cost = torch.linalg.norm(cost, dim=-1)

    # Initialize epsilon independent of the data dimension (i.e. dependent on the mean cost)
    epsilon = epsilon * cost.mean() / 2

    # Initialize the sample weights.
    log_a = torch.zeros(len(x), device=x.device) - np.log(len(x))
    log_b = torch.zeros(len(y), device=y.device) - np.log(len(y))

    return sinkhorn_log(log_a, log_b, cost, epsilon, steps)
