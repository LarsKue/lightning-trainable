
import warnings


def deprecate(message: str):
    warnings.warn(message, DeprecationWarning, stacklevel=2)
