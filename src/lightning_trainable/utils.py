from inspect import isclass

import torch
import torch.nn as nn


import re
from pathlib import Path


def get_optimizer(name):
    """ Get an optimizer in a case-insensitive way """
    optimizers = torch.optim.__dict__
    optimizers = {
        key.lower(): value for key, value in optimizers.items()
        if isclass(value) and issubclass(value, torch.optim.Optimizer)
    }

    return optimizers[name.lower()]


def get_scheduler(name):
    """ Get a scheduler in a case-insensitive way """
    schedulers = torch.optim.lr_scheduler.__dict__
    schedulers = {
        key.lower(): value for key, value in schedulers.items()
        if isclass(value) and (
                issubclass(value, torch.optim.lr_scheduler.LRScheduler)
                or issubclass(value, torch.optim.lr_scheduler.ReduceLROnPlateau)
        )
    }

    return schedulers[name.lower()]


def get_activation(name):
    """ Get an activation in a case-insensitive way """
    activations = torch.nn.modules.activation.__dict__
    activations = {
        key.lower(): value for key, value in activations.items()
        if isclass(value) and issubclass(value, torch.nn.Module)
    }

    return activations[name.lower()]


def get_module(name):
    """ Get a nn.Module in a case-insensitive way """
    modules = torch.nn.__dict__
    modules = {
        key.lower(): value for key, value in modules.items()
        if isclass(value) and issubclass(value, torch.nn.Module)
    }

    return modules[name.lower()]


def make_dense(widths: list[int], activation: str, dropout: float = None):
    """ Make a Dense Network from given layer widths and activation function """
    if len(widths) < 2:
        raise ValueError("Need at least Input and Output Layer.")

    Activation = get_module(activation)

    network = nn.Sequential()

    # input is x, time, condition
    for i in range(0, len(widths) - 2):
        if i > 0 and dropout is not None:
            network.add_module(f"Dropout_{i}", nn.Dropout1d(p=dropout))
        hidden_layer = nn.Linear(in_features=widths[i], out_features=widths[i + 1])
        network.add_module(f"Hidden_Layer_{i}", hidden_layer)
        network.add_module(f"Hidden_Activation_{i}", Activation())

    # output is velocity
    output_layer = nn.Linear(in_features=widths[-2], out_features=widths[-1])
    network.add_module("Output_Layer", output_layer)

    return network


def type_name(type):
    if isclass(type):
        return type.__name__
    return str(type)


def find_version(root: str | Path = "lightning_logs", version: int = "last") -> int:
    root = Path(root)

    # Determine latest version number if "last" is passed as version number
    if version == "last":
        version_folders = [f for f in root.iterdir() if f.is_dir() and re.match(r"^version_(\d+)$", f.name)]
        version_numbers = [int(re.match(r"^version_(\d+)$", f.name).group(1)) for f in version_folders]
        version = max(version_numbers)

    return version


def find_epoch_step(root: str | Path, epoch: int = "last", step: int = "last") -> (int, int):
    """
    Find epoch and step number for given checkpoint root. Checks if such a checkpoint exists.
    Note that this method *ignores* last.ckpt files (these are handled in find_checkpoint)

    @param root: Checkpoint root directory. Usually lightning_logs/version_i/checkpoints/
    @param epoch: Epoch number or "last"
    @param step: Step number or "last"
    @return: epoch and step numbers
    """

    root = Path(root)

    # get checkpoint filenames
    checkpoints = [f.name for f in root.glob("*.ckpt")]

    pattern = re.compile(r"^epoch=(\d+)-step=(\d+)\.ckpt$")

    # remove invalid files
    checkpoints = [cp for cp in checkpoints if pattern.match(cp)]

    # get epochs and steps as list
    matches = [pattern.match(cp) for cp in checkpoints]
    epochs, steps = zip(*[(int(match.group(1)), int(match.group(2))) for match in matches])

    if epoch == "last":
        epoch = max(epochs)
    elif epoch not in epochs:
        raise FileNotFoundError(f"No checkpoint in '{root}' for epoch '{epoch}'.")

    # keep only steps for this epoch
    steps = [s for i, s in enumerate(steps) if epochs[i] == epoch]

    if step == "last":
        step = max(steps)
    elif step not in steps:
        raise FileNotFoundError(f"No checkpoint in '{root}' for epoch '{epoch}', step '{step}'")

    return epoch, step


def find_checkpoint(root: str | Path = "lightning_logs", version: int = "last",
                    epoch: int = "last", step: int = "last") -> str:
    """
    Helper method to find a lightning checkpoint based on version, epoch and step numbers.

    @param root: logs root directory. Usually "lightning_logs/"
    @param version: version number or "last"
    @param epoch: epoch number or "last"
    @param step: step number or "last"
    @return: path to the checkpoint, relative to root
    """
    root = Path(root)

    if not root.is_dir():
        raise ValueError(f"Root directory '{root}' does not exist")

    # get existing version number or error
    version = find_version(root, version)

    checkpoint_folder = root / f"version_{version}" / "checkpoints"

    if epoch == "last" and step == "last":
        # return last.ckpt if it exists
        checkpoint = checkpoint_folder / "last.ckpt"
        if checkpoint.is_file():
            return str(checkpoint)

    # get existing epoch and step number or error
    epoch, step = find_epoch_step(checkpoint_folder, epoch, step)

    checkpoint = checkpoint_folder / f"epoch={epoch}-step={step}.ckpt"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"{version=}, {epoch=}, {step=}")

    return str(checkpoint)
