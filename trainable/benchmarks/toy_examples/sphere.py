from .benchmark import GenerativeBenchmark
from trainable import Trainable

import torch
import torch.distributions as D


class SphereBenchmark(GenerativeBenchmark):
    """ N-dimensional Gaussian Blobs situated on a hyper-sphere """
    def __init__(self, module_cls: Trainable, dimensions: int = 2, mixtures: int = 12):
        logits = torch.zeros(mixtures)
        means = sample_sphere(dimensions, mixtures, radius=10.0)
        stds = torch.ones_like(means)
        distribution = D.MixtureSameFamily(
            D.Categorical(logits=logits),
            D.Independent(D.Normal(loc=means, scale=stds), 1)
        )

        super().__init__(module_cls, distribution)


def sample_sphere(samples: int, dimensions: int, radius: float | int = 1.0):
    """ Monte-Carlo sample points uniformly on a hyper-sphere """
    points = torch.randn(samples, dimensions)
    points = radius * points / torch.linalg.norm(points, dim=-1, keepdim=True)

    return points
