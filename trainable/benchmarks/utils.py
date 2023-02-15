
import torch
from trainable import utils


def sample_sphere(sample_shape: tuple[int], dimensions: int, radii: torch.Tensor = torch.tensor(1.0)):
    assert radii.dim() == 1

    points = torch.randn(*sample_shape, len(radii), dimensions)
    radii = utils.unsqueeze_as(radii, points)
    points = radii * points / torch.linalg.norm(points, dim=-1, keepdim=True)

    return points.squeeze(-1)
