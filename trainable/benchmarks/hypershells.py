
import torch
import torch.distributions as D

from .distribution_dataset import DistributionDataset
from .utils import sample_sphere


class HypershellDistribution(D.Distribution):
    def __init__(self, radii: torch.Tensor, dimensions: int = 2, noise: float = 0.1):
        assert radii.dim() == 1
        super().__init__(batch_shape=(len(radii),), event_shape=(dimensions,))
        self.radii = radii
        self.dimensions = dimensions
        self.noise = noise

    def sample(self, sample_shape=torch.Size()):
        shells = sample_sphere(sample_shape[0], self.dimensions, self.radii)
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