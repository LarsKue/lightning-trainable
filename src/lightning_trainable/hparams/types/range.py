

class RangeMeta(type):
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

    def __call__(cls, lower: float | int, upper: float | int, exclude: str | None = None):
        # dynamically construct a new class with the given choices
        name = f"Range({lower!r}, {upper!r}, exclude={exclude!r})"
        bases = (Range,)
        namespace = {"lower": lower, "upper": upper, "exclude": exclude}
        return type(name, bases, namespace)


class Range(metaclass=RangeMeta):
    """
    A float within a given range. Not to be confused with the built-in range.
    Usage:
    class MyHParams(HParams):
        value: Range(0.0, 1.0, exclude="upper")

    hparams = MyHParams(value=0.5)
    assert hparams.value == 0.5
    """
    pass
