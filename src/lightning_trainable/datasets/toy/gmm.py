import torch
import torch.distributions as D

from ..core.distribution_dataset import DistributionDataset


class GaussianMixtureModelDistribution(D.MixtureSameFamily):
    """
    Gaussian Mixture Model Distribution
    """

    def __init__(self, means, stddevs, weights):
        super().__init__(D.Categorical(weights), D.MultivariateNormal(means, torch.diag_embed(stddevs)))


class GaussianMixtureModelDataset(DistributionDataset):
    def __init__(self, means, stddevs, weights, **kwargs):
        super().__init__(GaussianMixtureModelDistribution(means, stddevs, weights), **kwargs)
