
import pytest

from trainable import HParams


def test_defaults():
    class Defaults(HParams):
        a: int = 1
        b: float

    assert Defaults.defaults() == {"a": 1}


def test_missing():
    class Missing(HParams):
        a: int

    with pytest.raises(ValueError):
        Missing()


def test_extra():
    class Extra(HParams):
        a: int = 1

    with pytest.raises(ValueError):
        Extra(b=2.0)


def test_types():
    class WrongTypes(HParams):
        a: int

    with pytest.raises(TypeError):
        WrongTypes(a=2.0)


def test_required():
    class Required(HParams):
        required: int
        optional: float = 1.0

    assert Required.required_parameters() == {"required": int}
    assert Required.optional_parameters() == {"optional": float}


def test_inheritance():
    class BaseHParams(HParams):
        required: int
        override: int
        override_default: int = 1

    class DerivedHParams(BaseHParams):
        optional: int = 1
        override: float
        override_default: int = 2

    assert DerivedHParams.required_parameters() == {
        "required": int,
        "override": float,
    }
    assert DerivedHParams.defaults() == {
        "optional": 1,
        "override_default": 2,
    }

    d = DerivedHParams(required=1, override=1.0)
