from typing import Optional, Union

import pytest

from lightning_trainable import HParams

import pytorch_lightning as pl

pl.LightningModule


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

    DerivedHParams(required=1, override=1.0)


def test_getattr():
    class GetAttr(HParams):
        foo: int = 3

    hparams = GetAttr()
    assert hparams.foo == hparams["foo"] == hparams.get("foo")

    with pytest.raises(AttributeError):
        print(hparams.bar)


def test_riddle():
    class Riddle(HParams):
        foo: int = 3

    assert Riddle(foo=4).foo == 4


def test_nested():
    class SubHParams(HParams):
        foo: int

    class MainHParams(HParams):
        sub_hparams: SubHParams
        # Welcome to Python -- three ways to specify the same meaning, resulting in two different types
        optional_sub_hparams_v1: SubHParams | None = None
        optional_sub_hparams_v2: Union[SubHParams, None] = None
        optional_sub_hparams_v3: Optional[SubHParams] = None

    main_hparams = MainHParams(
        sub_hparams=dict(foo=3),
        optional_sub_hparams_v1=dict(foo=1),
        optional_sub_hparams_v2=dict(foo=4),
        optional_sub_hparams_v3=dict(foo=1),
    )
    assert isinstance(main_hparams.sub_hparams, SubHParams)
