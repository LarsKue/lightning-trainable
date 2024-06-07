import torch
import numpy as np


def sample_sphere(sample_shape: tuple[int], dimensions: int, radii: torch.Tensor = torch.tensor(1.0)):
    """ Uniformly sample points on a hypersphere """
    if not radii.dim() == 1:
        raise ValueError("radii must be a 1D tensor")

    points = torch.randn(*sample_shape, len(radii), dimensions)
    shape = [1 for _ in points.shape]
    shape[-2] = len(radii)
    radii = radii.reshape(shape)

    points = radii * points / torch.linalg.norm(points, dim=-1, keepdim=True)

    return points.squeeze(-1)


def sample_cube(samples: int, dimensions: int, side_length: torch.Tensor):
    """ Sample points on the corners of a hypercube """
    if not side_length.dim() == 1:
        raise ValueError("side_length must be a 1D tensor")
    if not samples <= 2 ** dimensions:
        raise ValueError("Too many samples for a hypercube of this dimensionality")

    selected_corners = np.random.choice(np.arange(2 ** dimensions), samples, replace=False)
    points = torch.Tensor([[float(axis) for axis in bin(corner)[2:].zfill(dimensions)] for corner in selected_corners])
    return points * side_length
