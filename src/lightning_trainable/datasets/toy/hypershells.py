
import torch
import torch.distributions as D
from torch.distributions import constraints

from lightning_trainable.datasets.core.distribution_dataset import DistributionDataset
from .utils import sample_sphere


class HypershellDistribution(D.Distribution):
    """
    This distribution consists of multiple concentric hyperspherical shells,
    meaning the density is roughly zero everywhere except on the surface of the hypersphere.
    Gaussian noise is added to diffuse the density from a delta distribution.
    """

    arg_constraints = {"radii": constraints.positive}

    def __init__(self, radii: torch.Tensor, dimensions: int = 2, noise: float = 0.1):
        if not radii.dim() == 1:
            raise ValueError("radii must be a 1D tensor")
        self.radii = radii
        super().__init__(batch_shape=(len(radii),), event_shape=(dimensions,))

        self.dimensions = dimensions
        self.noise = noise

    def sample(self, sample_shape=torch.Size()):
        shells = sample_sphere(sample_shape, self.dimensions, self.radii)
        noise = self.noise * torch.randn_like(shells)
        return shells + noise


class HypershellsDataset(DistributionDataset):
    """
    Dataset consisting of multiple concentric hyperspherical shells
    In 2D, this is roughly equivalent to sklearn's make_circles.
    """
    def __init__(self, dimensions: int = 2, shells: int = 2, noise: float = 0.05, **kwargs):
        logits = torch.zeros(shells)
        radii = 2 * torch.rand(shells)

        distribution = D.MixtureSameFamily(
            D.Categorical(logits=logits),
            HypershellDistribution(radii, dimensions, noise)
        )

        super().__init__(distribution, **kwargs)
