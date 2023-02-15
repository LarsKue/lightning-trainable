
import torch
import torch.nn as nn

import warnings

from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


def unsqueeze_to(x: torch.Tensor, dim: int, side="right"):
    """ Unsqueeze x1 on the right to match the given dimensionality """
    if dim < x.dim():
        raise RuntimeError(f"Cannot unsqueeze tensor of dim {x.dim()} to {dim}.")

    idx = [None] * (dim - x.dim())
    if side == "right":
        idx = [..., *idx]
    elif side == "left":
        idx = [*idx, ...]
    else:
        raise ValueError(f"Unknown side: {side}")

    return x[idx]


def unsqueeze_as(x1: torch.Tensor, x2: torch.Tensor, **kwargs):
    return unsqueeze_to(x1, x2.dim(), **kwargs)


def get_activation(activation):
    """ Return the corresponding torch Activation function by string """
    match activation.lower():
        case "relu":
            return nn.ReLU
        case "elu":
            return nn.ELU
        case "selu":
            return nn.SELU
        case activation:
            raise NotImplementedError(f"Unsupported Activation: {activation}")


def make_dense(widths: list[int], activation: str, dropout: float = None):
    """ Make a Dense Network from given layer widths and activation function """
    if len(widths) < 2:
        raise ValueError("Need at least Input and Output Layer.")
    elif len(widths) < 3:
        warnings.warn("Should use more than zero hidden layers.")

    Activation = get_activation(activation)

    network = nn.Sequential()

    # input is x, time, condition
    input_layer = nn.Linear(in_features=widths[0], out_features=widths[1])
    network.add_module("Input_Layer", input_layer)
    network.add_module("Input_Activation", Activation())

    for i in range(1, len(widths) - 2):
        if dropout is not None:
            network.add_module(f"Dropout_{i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden_Layer_{i}", hidden_layer)
        network.add_module(f"Hidden_Activation_{i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output_Layer", output_layer)

    return network


class EpochProgressBar(ProgressBarBase):
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        self.bar = Tqdm(
            desc="Epoch",
            leave=False,
            dynamic_ncols=True,
            total=trainer.max_epochs,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.bar.update(1)
        self.bar.set_description(f"Epoch {trainer.current_epoch}")
        self.bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer, pl_module):
        self.bar.close()
