from importlib import import_module

import torch
import lightning.pytorch.loggers
from inspect import isclass


def find_class(name: str, module, *base_classes):
    """
    Find a class in a module in a case-insensitive way.
    Only classes that are a subclass of one of the base_classes are returned.

    The class name must either be contained in specified module,
    or be a fully qualified name of a class in a module.
    """
    if "." in name:
        module = ".".join(name.split(".")[:-1])
        name = name.split(".")[-1]

        module = import_module(module)

    classes = module.__dict__
    classes = {
        key.lower(): value for key, value in classes.items()
        if isclass(value) and any(
            issubclass(value, base_class) for base_class in base_classes
        )
    }

    return classes[name.lower()]


def get_optimizer(name):
    """ Get an optimizer in a case-insensitive way """
    return find_class(name, torch.optim, torch.optim.Optimizer)


def get_scheduler(name):
    """ Get a scheduler in a case-insensitive way """
    return find_class(
        name, torch.optim.lr_scheduler,
        torch.optim.lr_scheduler.LRScheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau
    )


def get_activation(name):
    """ Get an activation in a case-insensitive way """
    return find_class(
        name, torch.nn.modules.activation,
        torch.nn.Module
    )


def get_module(name):
    """ Get a nn.Module in a case-insensitive way """
    return find_class(name, torch.nn, torch.nn.Module)


def get_logger(name):
    return find_class(
        name, lightning.pytorch.loggers,
        lightning.pytorch.loggers.Logger
    )
