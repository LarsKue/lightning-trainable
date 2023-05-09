
import torch
from inspect import isclass


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
