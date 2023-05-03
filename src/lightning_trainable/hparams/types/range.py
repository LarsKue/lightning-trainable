
from .hparam_type import HParamType


class Range(HParamType):
    """
    A float within a given range. Not to be confused with the built-in range.
    Usage:
    class MyHParams(HParams):
        value: Range(0.0, 1.0, exclude="upper")

    hparams = MyHParams(value=0.5)
    assert hparams.value == 0.5
    """
    def __init__(self, lower: float | int, upper: float | int, exclude: str | None = None):
        self.lower = lower
        self.upper = upper
        self.exclude = exclude

    def __instancecheck__(self, instance):
        match self.exclude:
            case "lower":
                return self.lower < instance <= self.upper
            case "upper":
                return self.lower <= instance < self.upper
            case "both":
                return self.lower < instance < self.upper
            case "neither" | None:
                return self.lower <= instance <= self.upper
            case other:
                raise NotImplementedError(f"Unrecognized exclude option: {other}")

    def __repr__(self):
        return f"Range({self.lower}, {self.upper}, exclude={self.exclude})"
