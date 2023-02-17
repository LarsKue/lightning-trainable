
import torch
import torch.distributions as D
from torch.distributions import constraints

from trainable.datasets.core.distribution_dataset import DistributionDataset


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
        phi = 2 * torch.pi * torch.rand(sample_shape)
        x = torch.cos(phi)
        y = torch.sin(phi)

        factors = torch.ones(sample_shape)

        rng = torch.randint(0, 2, size=sample_shape).to(bool)
        factors[rng] = self.factor

        out = factors[..., None] * torch.stack((x, y), dim=-1)

        noise = self.noise * torch.randn_like(out)

        return out + noise


class CirclesDataset(DistributionDataset):
    def __init__(self, noise: float = 0.05, factor: float = 0.5, **kwargs):
        super().__init__(CirclesDistribution(noise, factor), **kwargs)
