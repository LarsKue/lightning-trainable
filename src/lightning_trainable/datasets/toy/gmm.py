import torch
import torch.distributions as D

from ..core.distribution_dataset import DistributionDataset


class GaussianMixtureModelDistribution(D.MixtureSameFamily):
    """
    Distribution that allows for multiple gaussian blobs with different means, stddevs and weights.

    This behaviour is roughly equivalent to sklearn's make_blobs, but with control over the placement of the means in
    comparison to the hyperspheres.
    """

    def __init__(self, means, stddevs, weights):
        super().__init__(D.Categorical(weights), D.MultivariateNormal(means, torch.diag_embed(stddevs)))


class GaussianMixtureModelDataset(DistributionDataset):
    def __init__(self, means, stddevs, weights, **kwargs):
        super().__init__(GaussianMixtureModelDistribution(means, stddevs, weights), **kwargs)
