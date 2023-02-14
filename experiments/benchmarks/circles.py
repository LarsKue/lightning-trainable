
import torch
import torch.distributions as D

from sklearn.datasets import make_circles

from .distribution_dataset import DistributionDataset


class CirclesDistribution(D.Distribution):
    """
    Distribution based on the sklearn.make_circles function
    """
    def __init__(self, noise: float = 0.1, factor: float = 0.8):
        super().__init__()
        self.noise = noise
        self.factor = factor

    def sample(self, sample_shape=torch.Size()):
        x, y = make_circles(sample_shape[0], noise=self.noise, factor=self.factor)
        return torch.as_tensor(x, dtype=torch.float32)


class CirclesDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, factor: float = 0.5, **kwargs):
        super().__init__(CirclesDistribution(noise, factor), **kwargs)
