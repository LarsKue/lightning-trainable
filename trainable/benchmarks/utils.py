
import torch


def sample_sphere(sample_shape: tuple[int], dimensions: int, radii: torch.Tensor = torch.tensor(1.0)):
    assert radii.dim() == 1

    points = torch.randn(*sample_shape, len(radii), dimensions)
    shape = [1 for _ in points.shape]
    shape[-2] = len(radii)
    radii = radii.reshape(shape)

    points = radii * points / torch.linalg.norm(points, dim=-1, keepdim=True)

    return points.squeeze(-1)
