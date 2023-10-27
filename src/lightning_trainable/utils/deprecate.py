
import warnings
from functools import wraps
import inspect


def deprecate(message: str, version: str = None):

    def wrapper(obj):
        nonlocal message

        if inspect.isclass(obj):
            name = "Class"
        elif inspect.isfunction(obj):
            name = "Function"
        elif inspect.ismethod(obj):
            name = "Method"
        else:
            name = "Object"

        if version is not None:
            message = f"{name} '{obj.__name__}' is deprecated since version {version}: {message}"
        else:
            message = f"{name} '{obj.__name__}' is deprecated: {message}"

        @wraps(obj)
        def wrapped(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapped

    return wrapper
