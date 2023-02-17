
import torch
import torch.distributions as D
from torch.distributions import constraints

from sklearn.datasets import make_moons

from trainable.datasets.core.distribution_dataset import DistributionDataset


class MoonsDistribution(D.Distribution):
    """
    Distribution based on the sklearn.make_moons function

    Note that this is not exactly equivalent to sklearn's make_moons
    The data is instead zero-centered
    """

    arg_constraints = {"noise": constraints.positive}

    def __init__(self, noise: float = 0.1):
        # needs to be a tensor to avoid errors from arg constraint checks
        self.noise = torch.tensor(noise)
        super().__init__(event_shape=(2,))

    def sample(self, sample_shape=torch.Size()):
        phi = torch.pi * torch.rand(sample_shape)
        x1 = torch.cos(phi) - 0.5
        y1 = torch.sin(phi)
        x2 = 0.5 - torch.cos(phi)
        y2 = - torch.sin(phi)

        which = torch.randint(0, 2, size=sample_shape).to(bool)

        x = torch.where(which, x1, x2)
        y = torch.where(which, y1, y2)

        out = torch.stack((x, y), dim=-1)

        noise = self.noise * torch.randn_like(out)

        return out + noise


class MoonsDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, **kwargs):
        super().__init__(MoonsDistribution(noise), **kwargs)
