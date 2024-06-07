import torch
import torch.distributions as D

from lightning_trainable.datasets.core.distribution_dataset import DistributionDataset
from .utils import sample_sphere, sample_cube


class GaussianMixtureDataset(DistributionDataset):
    """
    Dataset consisting of multiple gaussian blobs. There are several modes for the distribution of the means:
    For "sphere", the means are sampled from a hypersphere with radius `radius`. In 2D, this is roughly equivalent to sklearn's make_blobs
    For "cube", the means are sampled from a hypercube with side length `radius`.
    For "random", the means are sampled from a normal distribution with std `radius`.
    """
    def __init__(self, dimensions: int = 2, mixtures: int = 4, radius: float = 1.0, std: float = 0.1,
                 mode: str = "sphere", **kwargs):
        logits = torch.zeros(mixtures)
        if mode == "sphere":
            means = sample_sphere((mixtures,), dimensions, radii=torch.tensor([radius])).squeeze(1)
        elif mode == "cube":
            means = sample_cube(mixtures, dimensions, side_length=torch.tensor([radius]))
        elif mode == "random":
            means = radius * torch.randn(mixtures, dimensions)
        else:
            raise ValueError(f"Unknown mode {mode}")
        stds = torch.full_like(means, fill_value=std)
        distribution = D.MixtureSameFamily(
            D.Categorical(logits=logits),
            D.Independent(D.Normal(loc=means, scale=stds), 1)
        )

        super().__init__(distribution, **kwargs)
