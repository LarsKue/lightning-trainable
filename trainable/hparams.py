"""
MIT License

Copyright (c) 2023 Lars Erik Kühmichel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from types import GenericAlias
from typing import get_origin


class HParams(dict):
    """
    Wrapper class to handle hparams with defaults, required and optional keys, and optional strict type checks
    Usage:
    ```
        class MyNetworkHParams(HParams):
            required_hparam: str
            optional_hparam: int = 0
        hparams = MyNetworkHParams(required_hparam="required")
    ```

    You can turn strict type checking off with the class attribute `strict_types`:
    ```
        class NoStrictHParams(HParams):
            strict_types = False

            required_hparam: str
            optional_hparam: int = 0
    ```

    Note that this class uses the __annotations__ attribute of its subclasses to identify hparams and their types.
    This means that any class parameter that has a type hint will be treated as an hparam. If you want to add
    class attributes that are not hparams, do not use a type hint.
    """
    strict_types = True

    def __init__(self, **hparams):
        hparams = self.validate_parameters(hparams)
        super().__init__(**hparams)

    def __init_subclass__(cls):
        if not cls.strict_types:
            return

        # check that subclasses do not use generic types
        for parameter, T in cls.parameters().items():
            if isinstance(T, GenericAlias):
                basic_type = get_origin(T)
                raise NotImplementedError(f"HParams does not support type checking for generics. "
                                          f"Use the basic type `{basic_type.__name__}` instead of `{T}` "
                                          f"for parameter '{parameter}'.")

    @classmethod
    def validate_parameters(cls, hparams: dict) -> dict:
        """
        Check given hparams for validity, and fill missing ones with defaults.

        By default, hparams are valid if and only if:
            1. All required parameters are filled
            2. No unknown parameters are given
            3. All hparams match their designated type

        """
        required_parameters = cls.required_parameters()
        all_parameters = cls.parameters()

        have_keys = set(hparams.keys())
        required_keys = set(required_parameters.keys())
        all_keys = set(all_parameters.keys())

        # check required keys are in hparams
        if not required_keys.issubset(have_keys):
            missing_keys = required_keys - have_keys
            types = [all_parameters[key] for key in missing_keys]
            message = "Missing the following required hparams:\n"
            message += "\n".join([
                f"{i + 1:4d}: '{key}' of type `{T.__name__}`" for i, (key, T) in enumerate(zip(missing_keys, types))
            ])
            raise ValueError(message)

        # check no extra keys are in hparams
        if not have_keys.issubset(all_keys):
            extra_keys = have_keys - all_keys
            values = [hparams[key] for key in extra_keys]
            message = "Received the following extra hparams:"
            message += "\n".join([
                f"{i + 1:4d}: '{key}' = {value}" for i, (key, value) in enumerate(zip(extra_keys, values))
            ])
            raise ValueError(message)

        # insert defaults
        hparams = cls.defaults() | hparams

        if cls.strict_types:
            # check types match
            for key, value in hparams.items():
                T = all_parameters[key]

                # noinspection PyTypeHints
                if not isinstance(value, T):
                    raise TypeError(f"Hparam '{key}' is required to be of type `{T.__name__}`, "
                                    f"but got `{value}` of type `{type(value).__name__}`.")

        return hparams

    @classmethod
    def parameters(cls) -> dict[str, type]:
        """ Return names and types for all hparams """
        if cls is HParams:
            return dict()

        types = cls.__annotations__.copy()

        superclasses = cls.__mro__[1:]

        for c in superclasses:
            if issubclass(c, HParams):
                # override superclasses with subclasses
                types = c.parameters() | types

        return types

    @classmethod
    def required_parameters(cls) -> dict[str, type]:
        """ Return names and types for all required hparams """
        # parameters that do not have a default value are required
        defaults = dir(cls)
        return {key: value for key, value in cls.parameters().items() if key not in defaults}

    @classmethod
    def optional_parameters(cls) -> dict[str, type]:
        """ Return names and types for all optional hparams """
        required = cls.required_parameters()
        return {key: value for key, value in cls.parameters().items() if key not in required}

    @classmethod
    def defaults(cls) -> dict[str, any]:
        """ Return names and default values for all optional hparams """
        optional_keys = cls.optional_parameters().keys()
        return {key: getattr(cls, key) for key in optional_keys}
