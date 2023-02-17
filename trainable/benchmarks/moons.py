
import torch
import torch.distributions as D
from torch.distributions import constraints

from sklearn.datasets import make_moons

from .distribution_dataset import DistributionDataset


class MoonsDistribution(D.Distribution):
    """
    Distribution based on the sklearn.make_moons function
    """

    arg_constraints = {"noise": constraints.positive}

    def __init__(self, noise: float = 0.1):
        # needs to be a tensor to avoid errors from arg constraint checks
        self.noise = torch.tensor(noise)
        super().__init__(event_shape=(2,))

    def sample(self, sample_shape=torch.Size()):
        sample_shape = sample_shape or (1,)
        x, y = make_moons(sample_shape[0], noise=self.noise.item())
        return torch.as_tensor(x, dtype=torch.float32).squeeze()


class MoonsDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, **kwargs):
        super().__init__(MoonsDistribution(noise), **kwargs)
