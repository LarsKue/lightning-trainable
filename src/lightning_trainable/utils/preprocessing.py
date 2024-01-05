
from torch import Tensor


def normalize(x: Tensor, dim: int) -> Tensor:
    """
    Normalizes a tensor along a given dimension.
    @param x: Tensor to normalize.
    @param dim: Dimension along which to normalize.
    @return: Normalized tensor.
    """
    return (x - x.mean(dim)) / x.std(dim)
