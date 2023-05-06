import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, inner: nn.Module, mode: str = "add"):
        super().__init__()
        self.inner = inner
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.mode:
            case "add":
                return self.inner(x) + x
            case "concat":
                return torch.cat((self.inner(x), x), dim=1)
            case "none":
                return self.inner(x)
            case other:
                raise NotImplementedError(f"Unrecognized skip connection mode '{other}'.")

    def extra_repr(self) -> str:
        return f"mode={self.mode}"
