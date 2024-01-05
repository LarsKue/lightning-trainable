
import torch
from torch import Tensor
from torch.nn import functional as F


def _normalized_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    cost = x[:, None] - y[None, :]
    cost = torch.flatten(cost, start_dim=2)
    cost = torch.linalg.norm(cost, dim=-1)

    cost = (cost - cost.mean()) / cost.std()

    return cost


def hungarian(x: Tensor, y: Tensor, cost: Tensor = None, epsilon: float = 1.0, max_steps: int = 1000, atol: float = 1e-6) -> Tensor:
    """
    Computes the Hungarian Optimal Assignment Plan from samples from two distributions.

    @param x: Samples from the first distribution in shape (n, ...).
    @param y: Samples from the second distribution in shape (m, ...).
    @param cost: Optional cost matrix in shape (n, m).
        If not provided, the normalized Euclidean distance is used.
    @param epsilon: Optional entropic regularization parameter.
        This parameter controls the standard deviation of the Gaussian kernel.
    @param max_steps: Maximum number of iterations.
    @param atol: Absolute tolerance for convergence.

    @return plan: Tensor of shape (n, m) containing the assignment from x to y.
    """
    if x.shape[1:] != y.shape[1:]:
        raise ValueError(f"Expected x and y to live in the same feature space, "
                         f"but got {x.shape[1:]} and {y.shape[1:]}.")
    if epsilon <= 0:
        raise ValueError(f"Expected epsilon to be positive, but got {epsilon}.")

    if cost is None:
        cost = _normalized_euclidean_distance(x, y)
    else:
        if cost.shape != (x.shape[0], y.shape[0]):
            raise ValueError(f"Expected cost to have shape ({x.shape[0]}, {y.shape[0]}), "
                             f"but got {cost.shape}.")

    # Gaussian kernel
    cost = torch.exp(-0.5 * cost / epsilon)

    # ensure matrix is square
    max_cost = torch.max(cost)
    padding_right = max(0, x.shape[0] - y.shape[0])
    padding_bottom = max(0, y.shape[0] - x.shape[0])
    cost = F.pad(cost, (0, padding_right, 0, padding_bottom), value=max_cost.item())

    # subtract minimum costs from each row and column
    cost -= cost.min(dim=1, keepdim=True).values
    cost -= cost.min(dim=0, keepdim=True).values

    covered_rows = torch.zeros(cost.shape[0], dtype=torch.bool)
    covered_cols = torch.zeros(cost.shape[1], dtype=torch.bool)

    while True:
        # cover all zeros with minimum number of lines
        uncovered_zeros = (cost < atol) & ~(covered_rows[:, None] | covered_cols[None, :])

        if torch.any(uncovered_zeros):
            # Update rows and columns
            rows, cols = torch.nonzero(uncovered_zeros, as_tuple=True)
            covered_rows[rows] = True
            covered_cols[cols] = True
        else:
            # all zeros are covered
            break

    # calculate assignment
    plan = ~(covered_rows[:, None] | covered_cols[None, :])

    # remove padding
    plan = plan[:x.shape[0], :y.shape[0]]

    return plan


# x = torch.randn(3, 2)
# y = torch.randn(3, 2)
#
# from sinkhorn import sinkhorn
# print(sinkhorn(x, y))
# print(hungarian(x, y))
