
from inspect import isclass


def type_name(type):
    if isclass(type):
        return type.__name__
    return str(type)
