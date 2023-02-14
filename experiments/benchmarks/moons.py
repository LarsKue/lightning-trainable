
import torch
import torch.distributions as D

from sklearn.datasets import make_moons

from .distribution_dataset import DistributionDataset


class MoonsDistribution(D.Distribution):
    """
    Distribution based on the sklearn.make_moons function
    """
    def __init__(self, noise: float = 0.1):
        super().__init__()
        self.noise = noise

    def sample(self, sample_shape=torch.Size()):
        x, y = make_moons(sample_shape[0], noise=self.noise)
        return torch.as_tensor(x, dtype=torch.float32)


class MoonsDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, **kwargs):
        super().__init__(MoonsDistribution(noise), **kwargs)
