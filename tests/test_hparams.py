
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

    with pytest.raises(KeyError):
        m = Missing()


def test_extra():
    class Extra(HParams):
        a: int = 1

    with pytest.raises(KeyError):
        e = Extra(b=2.0)


def test_types():
    class WrongTypes(HParams):
        a: int

    with pytest.raises(TypeError):
        w = WrongTypes(a=2.0)


def test_required():
    class Required(HParams):
        required: int
        optional: float = 1.0

    assert Required.required_keys() == {"required"}
    assert Required.optional_keys() == {"optional"}
