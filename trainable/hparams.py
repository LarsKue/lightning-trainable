from typing import TypedDict

from pytorch_lightning.profiler import Profiler


class HParams(dict):
    """
    Wrapper class to handle hparams with defaults, required and optional keys, and strict type checks
    Usage:
    ```
        class MyNetworkHParams(HParams):
            required_hparam: str
            optional_hparam: int = 0

        hparams = MyNetworkHParams(required_hparam="required")
    ```
    """
    def __init__(self, **hparams):
        hparams = self.validate_hparams(**hparams)
        super().__init__(**hparams)

    @classmethod
    def validate_hparams(cls, **hparams):
        have_keys = set(hparams.keys())
        required_keys = cls.required_keys()
        all_keys = set(cls.__annotations__.keys())

        # check required keys are in hparams
        if not required_keys.issubset(have_keys):
            missing_keys = required_keys - have_keys
            types = [cls.__annotations__[key] for key in missing_keys]
            message = "Missing the following required hparams:\n"
            message += "\n".join([
                f"'{key}' of type {T}" for key, T in zip(missing_keys, types)
            ])
            raise KeyError(message)

        # check no extra keys are in hparams
        if not have_keys.issubset(all_keys):
            extra_keys = have_keys - all_keys
            values = [hparams[key] for key in extra_keys]
            message = "Received the following extra hparams:"
            message += "\n".join([
                f"'{key}' = {value}" for key, value in zip(extra_keys, values)
            ])

        # insert defaults
        hparams = cls.defaults() | hparams

        # check types match
        for key, value in hparams.items():
            T = cls.__annotations__[key]
            if not isinstance(value, T):
                raise TypeError(f"Hparam '{key}' is required to be of type {T}, but got {value} of type {type(value)}.")

        return hparams

    @classmethod
    def required_keys(cls):
        return {key for key in cls.__annotations__.keys() if key not in dir(cls)}

    @classmethod
    def optional_keys(cls):
        return {key for key in cls.__annotations__.keys() if key in dir(cls)}

    @classmethod
    def defaults(cls):
        return {key: getattr(cls, key) for key, value in cls.__annotations__.items() if key in dir(cls)}
