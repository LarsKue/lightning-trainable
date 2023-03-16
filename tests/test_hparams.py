from typing import Optional, Union

import pytest

from lightning_trainable import HParams
from lightning_trainable.hparams import Choice

from typing import Literal


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

    foo_val = 4
    hparams = GetAttr(foo=foo_val)
    assert hparams.foo == hparams["foo"] == hparams.get("foo") == foo_val

    with pytest.raises(AttributeError):
        print(hparams.bar)


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
    assert isinstance(main_hparams.optional_sub_hparams_v1, SubHParams)
    assert isinstance(main_hparams.optional_sub_hparams_v2, SubHParams)
    assert isinstance(main_hparams.optional_sub_hparams_v3, SubHParams)


def test_nested_dict():
    class SubHParams(HParams):
        foo: int

    class MainHParams(HParams):
        sub_hparams: SubHParams | dict

    main_hparams = MainHParams(
        sub_hparams=dict(foo=7)
    )

    # may be subject to change
    assert isinstance(main_hparams.sub_hparams, dict)

    class SubHParams2(HParams):
        bar: int

    class MainHParams(HParams):
        sub_hparams: SubHParams | SubHParams2

    main_hparams = MainHParams(
        sub_hparams=SubHParams(foo=7)
    )

    assert isinstance(main_hparams.sub_hparams, SubHParams)

    main_hparams = MainHParams(
        sub_hparams=SubHParams2(bar=7)
    )

    assert isinstance(main_hparams.sub_hparams, SubHParams2)

    with pytest.raises(RuntimeError):
        MainHParams(
            sub_hparams=dict(foo=7)
        )


def test_choice():
    class ChoiceHParams(HParams):
        value: Choice("x", "y", "z")

    hparams = ChoiceHParams(value="x")
    assert isinstance(hparams.value, str)
    assert hparams.value == "x"

    hparams = ChoiceHParams(value="y")
    assert hparams.value == "y"

    hparams = ChoiceHParams(value="z")
    assert hparams.value == "z"

    with pytest.raises(TypeError):
        hparams = ChoiceHParams(value="asdf")
