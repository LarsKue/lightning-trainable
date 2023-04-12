from inspect import isclass
from types import GenericAlias, UnionType
from typing import get_origin, Union, get_args

import os

import json
import tomli
import yaml

from lightning_trainable.utils import type_name


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
        hparams = NoStrictHParams(required_hparam=42)
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
                                          f"Use the basic type `{type_name(basic_type)}` instead of `{type_name(T)}` "
                                          f"for parameter '{parameter}'.")

    @classmethod
    def validate_parameters(cls, hparams: dict) -> dict:
        """
        Check given hparams for validity, and fill missing ones with defaults.

        Convert nested dicts to HParams if specified.

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
            typenames = [T.__name__ if hasattr(T, "__name__") else repr(T) for T in types]
            message = "Missing the following required hparams:\n"
            message += "\n".join([
                f"{i + 1:4d}: '{key}' of type `{type_name(T)}`" for i, (key, T) in enumerate(zip(missing_keys, typenames))
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

        # Modify hparams in-place
        cls._convert_dicts(hparams)

        if cls.strict_types:
            # check types match
            for key, value in hparams.items():
                T = all_parameters[key]
                if T is None:
                    T = type(None)

                # noinspection PyTypeHints
                if not isinstance(value, T):
                    raise TypeError(f"Hparam '{key}' is required to be of type `{type_name(T)}`, "
                                    f"but got `{value}` of type `{type_name(type(value))}`.")

        return hparams

    @classmethod
    def _convert_dicts(cls, hparams):
        """ Attempts to convert nested dicts to HParams subclasses, based on the type hint """
        all_parameters = cls.parameters()
        for key, value in hparams.items():
            T = all_parameters[key]

            if not isinstance(value, dict) or isinstance(value, HParams):
                # either not a dict or already a kind of HParams,
                # so no conversion is necessary
                continue

            if isclass(T) and issubclass(T, HParams):
                # convert dict to HParams
                hparam_type = T
            elif get_origin(T) in [Union, UnionType]:
                # explicitly defined Unions, or UnionType for type(int | str)
                # convert type hinted ... | XHParams | ... to XHParams
                types = get_args(T)
                if dict in types:
                    # the user may want a dict, do not attempt implicit conversion
                    continue

                hparam_types = [t for t in types if issubclass(t, HParams)]
                if len(hparam_types) == 0:
                    # no HParams classes to convert to
                    continue
                elif len(hparam_types) == 1:
                    hparam_type = hparam_types[0]
                else:
                    raise RuntimeError(f"Cannot implicitly convert dict to {type_name(T)} "
                                       f"when multiple subclasses of HParams are type-hinted for key {key!r}.")
            else:
                continue

            hparams[key] = hparam_type(**value)

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

    @classmethod
    def from_yaml(cls, path: str):
        """ Load hparams from a YAML file """
        with open(path, "r") as f:
            hparams = yaml.safe_load(f)

        return cls(**hparams)

    @classmethod
    def from_json(cls, path: str):
        """ Load hparams from a JSON file """
        with open(path, "r") as f:
            hparams = json.load(f)

        return cls(**hparams)

    @classmethod
    def from_toml(cls, path: str):
        """ Load hparams from a TOML file """
        with open(path, "rb") as f:
            hparams = tomli.load(f)

        return cls(**hparams)

    @classmethod
    def from_file(cls, path: str):
        """ Load hparams from a file, based on the file extension """
        ext = os.path.splitext(path)[1]

        match ext:
            case "yaml":
                return cls.from_yaml(path)
            case "json":
                return cls.from_json(path)
            case "toml":
                return cls.from_toml(path)
            case _:
                raise NotImplementedError(f"Cannot auto-infer file type from extension {ext!r}.")

    def __getattribute__(self, item):
        if item in self:
            return self[item]

        return super().__getattribute__(item)

    def __setattr__(self, key, value):
        self[key] = value
