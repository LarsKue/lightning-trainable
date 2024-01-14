import warnings

import torch
import torch.nn.functional as F

from torch import Tensor


def _euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    cost = x[:, None] - y[None, :]
    cost = torch.flatten(cost, start_dim=2)
    cost = torch.linalg.norm(cost, dim=-1)

    return cost


def sinkhorn(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 1.0, scale_epsilon: bool = True, max_steps: int = 1000, atol: float = 1e-6) -> Tensor:
    """
    Computes the Sinkhorn-Knopp Optimal Transport Plan from samples from two distributions.
    This version is stabilized by performing the computations in the log space.

    @param x: Samples from the first distribution in shape (n, ...).
    @param y: Samples from the second distribution in shape (m, ...).
    @param cost: Optional cost matrix in shape (n, m).
        If not provided, the normalized Euclidean distance is used.
    @param epsilon: Optional entropic regularization parameter.
        This parameter controls the standard deviation of the Gaussian kernel.
    @param scale_epsilon: Whether to scale the epsilon parameter by the half-mean of the cost matrix.
        This is useful when the cost is not typically near 1.
    @param max_steps: Maximum number of iterations.
    @param atol: Absolute tolerance for convergence.

    @return log_plan: Tensor of shape (n, m) containing the log transport probabilities.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Expected x and y to live in the same feature space, "
                         f"but got {x.shape[1:]} and {y.shape[1:]}.")

    if cost is None:
        cost = _euclidean_distance(x, y)
    else:
        if cost.shape != (x.shape[0], y.shape[0]):
            raise ValueError(f"Expected cost to have shape ({x.shape[0]}, {y.shape[0]}), "
                             f"but got {cost.shape}.")

    if epsilon <= 0:
        raise ValueError(f"Expected epsilon to be positive, but got {epsilon}.")

    if scale_epsilon:
        epsilon *= 0.5 * cost.mean()

    # logarithmic Gaussian kernel
    log_plan = -0.5 * cost / epsilon

    converged = False
    for _ in range(max_steps):
        # Sinkhorn-Knopp: Repeatedly normalize along each dimension
        log_plan = torch.log_softmax(log_plan, dim=0)
        log_plan = torch.log_softmax(log_plan, dim=1)

        # check convergence, the plan should be doubly stochastic
        marginal = torch.logsumexp(log_plan, dim=0)
        converged = torch.all(marginal < atol)

        # no need to check dim 1 since we just normalized along that

        if converged:
            break

    if not converged:
        badness = torch.max(torch.abs(marginal))
        warnings.warn(f"Sinkhorn-Knopp did not converge (badness: {badness:.1e}). "
                      f"Consider relaxing epsilon ({epsilon:.1e}) or increasing the "
                      f"number of iterations ({max_steps}).")

    return log_plan
