from inspect import isclass

import torch
import torch.nn as nn


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
