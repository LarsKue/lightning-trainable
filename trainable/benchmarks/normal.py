
import torch
import torch.distributions as D

from .distribution_dataset import DistributionDataset


class NormalDataset(DistributionDataset):
    """
    This dataset consists of normally distributed points
    It is useful for training generative models, like diffusion models
    """
    def __init__(self, dimensions: int, **kwargs):
        distribution = D.Normal(torch.zeros(dimensions), torch.ones(dimensions))

        super().__init__(distribution, **kwargs)
