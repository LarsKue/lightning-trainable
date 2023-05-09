from inspect import isclass
from types import GenericAlias, UnionType
from typing import get_origin, Union, get_args

import os

import json
import tomli
import yaml

from .attribute_dict import AttributeDict
from .type_name import type_name


class HParams(AttributeDict):
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
        hparams = AttributeDict(**hparams)
        hparams = self.validate_parameters(hparams)
        if hparams is None:
            raise ValueError(f"You must return hparams from `validate_parameters()`.")
        super().__init__(**hparams)

    def __init_subclass__(cls):
        if not cls.strict_types:
            return

        # check that subclasses do not use non-supported generic types
        for parameter, T in cls.parameters().items():
            cls._check_is_valid_hint(parameter, T)

    @classmethod
    def _check_is_valid_hint(cls, parameter, T, recurse=True):
        """
        Checks that the given type hint is supported by HParams.
        Most importantly, this covers checking that no unsupported generic types are used.
        """
        basic_type = get_origin(T)
        type_args = get_args(T)

        if basic_type in [Union, UnionType]:
            # cover hints like Union[str, list[int]] or str | list[int]
            for arg in type_args:
                cls._check_is_valid_hint(parameter, arg, recurse=True)
            return

        if basic_type is GenericAlias:
            # cover hints like list[str] or dict[str, int]
            if not recurse:
                raise NotImplementedError(f"HParams does not support nested generic types. "
                                          f"Please use a concrete type for parameter '{parameter}'.")
            for arg in type_args:
                cls._check_is_valid_hint(parameter, arg, recurse=False)
            return

        if type_args:
            supported_types = [dict, list, tuple]
            if basic_type in supported_types and recurse:
                # cover hints like list[int] or dict[str, int]
                for arg in type_args:
                    cls._check_is_valid_hint(parameter, arg, recurse=False)
                return

            # cover hints like MyGeneric[int] or frozenset[int]
            git_url = "https://github.com/LarsKue/lightning-trainable"
            raise NotImplementedError(f"HParams does not support generic types for the basic type "
                                      f"'{basic_type}' of parameter '{parameter}'. "
                                      f"Please file an issue at {git_url} if you need this feature.")

        # other types are just basic types,
        # e.g. str, int, float, etc.
        # so these are fine

    @classmethod
    def validate_parameters(cls, hparams: AttributeDict) -> AttributeDict:
        """
        Check given hparams for validity, and fill missing ones with defaults.

        Migrate hparams via cls._migrate_hparams(hparams).
        Convert nested dicts to HParams if specified.

        By default, hparams are valid if and only if:
            1. All required parameters are filled
            2. No unknown parameters are given
            3. All hparams match their designated type

        """
        required_parameters = cls.required_parameters()
        all_parameters = cls.parameters()

        hparams = cls._migrate_hparams(hparams)

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
                f"{i + 1:4d}: '{key}' of type `{type_name(T)}`" for i, (key, T) in
                enumerate(zip(missing_keys, typenames))
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
        hparams = AttributeDict(**(cls.defaults() | hparams))

        # Modify hparams in-place
        cls._convert_dicts(hparams)

        if cls.strict_types:
            # check types match
            for key, value in hparams.items():
                T = all_parameters[key]
                cls._check_type(key, value, T)

        return hparams

    @classmethod
    def _migrate_hparams(cls, hparams):
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
    def _check_type(cls, key, value, T):
        """
        Check that the type of `value` matches the type hint `T`.
        """
        if T is None:
            # allow hinting NoneType as None
            T = type(None)

        basic_type = get_origin(T)
        type_args = get_args(T)

        if basic_type in [Union, UnionType]:
            if any(isinstance(t, GenericAlias) for t in type_args):
                # cover hints like str | list[int]
                for arg in type_args:
                    cls._check_type(key, value, arg)
                return

        if isinstance(T, GenericAlias):
            # cover hints like list[int] or dict[str, int]
            cls._check_generic_type(key, value, T)
            return

        # noinspection PyTypeHints
        if not isinstance(value, T):
            raise TypeError(f"Hparam '{key}' is required to be of type `{type_name(T)}`, "
                            f"but got `{value}` of type `{type_name(type(value))}`.")

    @staticmethod
    def _check_generic_type(key, value, T):
        """
        Check that the type of `value` matches the type hint `T` for a GenericAlias like list[int].
        """
        assert isinstance(T, GenericAlias)

        basic_type = get_origin(T)
        type_args = get_args(T)

        if not isinstance(value, basic_type):
            raise TypeError(f"Hparam '{key}' is required to be of base type `{type_name(basic_type)}`, "
                            f"but got `{value}` of base type `{type_name(get_origin(type(value)))}`.")

        # we already know type(value) is a supported type, since we check this at class creation

        if basic_type is dict:
            K, V = type_args
            for k, v in value.items():
                if not isinstance(k, K):
                    raise TypeError(f"Dict key '{k}' is required to be of type `{type_name(K)}`, "
                                    f"but got type `{type_name(type(k))}`.")
                if not isinstance(v, V):
                    raise TypeError(f"Dict value for key '{k}' is required to be of type `{type_name(V)}`, "
                                    f"but got `{v}` of type `{type_name(type(v))}`.")

        if basic_type is list:
            V = type_args[0]
            for i, v in enumerate(value):
                # noinspection PyTypeHints
                if not isinstance(v, V):
                    raise TypeError(f"List value at index {i} is required to be of type `{type_name(V)}`, "
                                    f"but got `{v}` of type `{type_name(type(v))}`.")

        if basic_type is tuple:
            for i, (v, V) in enumerate(zip(value, type_args)):
                if not isinstance(v, V):
                    raise TypeError(f"Tuple value at index {i} is required to be of type `{type_name(V)}`, "
                                    f"but got `{v}` of type `{type_name(type(v))}`.")

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
                types = AttributeDict(**(c.parameters() | types))

        return types

    @classmethod
    def required_parameters(cls) -> dict[str, type]:
        """ Return names and types for all required hparams """
        # parameters that do not have a default value are required
        defaults = dir(cls)
        return AttributeDict(**{key: value for key, value in cls.parameters().items() if key not in defaults})

    @classmethod
    def optional_parameters(cls) -> dict[str, type]:
        """ Return names and types for all optional hparams """
        required = cls.required_parameters()
        return AttributeDict(**{key: value for key, value in cls.parameters().items() if key not in required})

    @classmethod
    def defaults(cls) -> dict[str, any]:
        """ Return names and default values for all optional hparams """
        optional_keys = cls.optional_parameters().keys()
        return AttributeDict(**{key: getattr(cls, key) for key in optional_keys})

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
