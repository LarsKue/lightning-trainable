
import torch


def sample_sphere(samples: int, dimensions: int, radii: torch.Tensor = torch.tensor(1.0)):
    """ Monte-Carlo sample points uniformly on hyper-spheres """
    assert radii.dim() == 1

    points = torch.randn(samples, len(radii), dimensions)
    points = radii[None, :, None] * points / torch.linalg.norm(points, dim=-1, keepdim=True)

    return points
