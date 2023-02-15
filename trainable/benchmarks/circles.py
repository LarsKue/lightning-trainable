
import torch
import torch.distributions as D
from torch.distributions import constraints

from sklearn.datasets import make_circles

from .distribution_dataset import DistributionDataset


class CirclesDistribution(D.Distribution):
    """
    Distribution based on the sklearn.make_circles function
    """

    arg_constraints = {"noise": constraints.positive, "factor": constraints.positive}

    def __init__(self, noise: float = 0.1, factor: float = 0.8):
        # these need to be tensors to avoid errors from arg constraint checking
        self.noise = torch.tensor(noise)
        self.factor = torch.tensor(factor)
        super().__init__(event_shape=(2,))

    def sample(self, sample_shape=torch.Size()):
        x, y = make_circles(sample_shape[0], noise=self.noise.item(), factor=self.factor.item())
        return torch.as_tensor(x, dtype=torch.float32)


class CirclesDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, factor: float = 0.5, **kwargs):
        super().__init__(CirclesDistribution(noise, factor), **kwargs)
