import torch
import torch.distributions as D

from .distribution_dataset import DistributionDataset
from .utils import sample_sphere


class HypersphereMixtureDataset(DistributionDataset):
    """
    Dataset consisting of multiple gaussian blobs situated uniformly on a hypersphere
    In 2D, this is roughly equivalent to sklearn's make_blobs
    """
    def __init__(self, dimensions: int = 2, mixtures: int = 8, radius: float = 1.0, std: float = 0.1, **kwargs):
        logits = torch.zeros(mixtures)
        means = sample_sphere((mixtures,), dimensions, radii=torch.tensor([radius])).squeeze(1)
        stds = torch.full_like(means, fill_value=std)
        distribution = D.MixtureSameFamily(
            D.Categorical(logits=logits),
            D.Independent(D.Normal(loc=means, scale=stds), 1)
        )

        super().__init__(distribution, **kwargs)
