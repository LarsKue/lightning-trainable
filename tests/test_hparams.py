from typing import Generic, Optional, TypeVar, Union

import pytest

from lightning_trainable.hparams import Choice, HParams, Range


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

    class UnionTypeChoiceHParams(HParams):
        value: int | Choice("x", "y")

    class _(HParams):
        # constructing this class between defining UnionTypeChoiceHParams and using it
        # should not affect the behavior of UnionTypeChoiceHParams
        unrelated: int | Choice("a", "b")

    hparams = UnionTypeChoiceHParams(value="x")
    assert isinstance(hparams.value, str)
    assert hparams.value == "x"

    hparams = UnionTypeChoiceHParams(value=1)
    assert isinstance(hparams.value, int)
    assert hparams.value == 1

    with pytest.raises(TypeError):
        hparams = UnionTypeChoiceHParams(value="z")

    with pytest.raises(TypeError):
        hparams = UnionTypeChoiceHParams(value=1.0)


def test_range():
    class RangeHParams(HParams):
        value: Range(0, 1, exclude="upper")

    hparams = RangeHParams(value=0.0)
    assert isinstance(hparams.value, float)
    assert hparams.value == 0.0

    with pytest.raises(TypeError):
        # out of range (lower)
        hparams = RangeHParams(value=-1.0)

    with pytest.raises(TypeError):
        # out of range (upper)
        hparams = RangeHParams(value=1.5)

    with pytest.raises(TypeError):
        # we exclude upper, so this should raise
        hparams = RangeHParams(value=1.0)

    class UnionTypeRangeHParams(HParams):
        value: str | Range(0.0, 1.0)

    class _(HParams):
        # constructing this class between defining UnionTypeRangeHParams and using it
        # should not affect the behavior of UnionTypeRangeHParams
        unrelated: int | Range(-1.0, 0.0)

    hparams = UnionTypeRangeHParams(value="x")
    assert isinstance(hparams.value, str)
    assert hparams.value == "x"

    hparams = UnionTypeRangeHParams(value=0.5)
    assert isinstance(hparams.value, float)
    assert hparams.value == 0.5

    with pytest.raises(TypeError):
        hparams = UnionTypeRangeHParams(value=1.5)

    with pytest.raises(TypeError):
        hparams = UnionTypeRangeHParams(value=-0.5)


def test_migrate():
    class NewHParams(HParams):
        value: int

        @classmethod
        def _migrate_hparams(cls, hparams):
            if "old_value" in hparams:
                hparams["value"] = hparams["old_value"]
                del hparams["old_value"]
            return hparams

    with pytest.raises(ValueError):
        NewHParams(some_value=1)
    hparams = NewHParams(old_value=1)
    hparams = NewHParams(value=1)


def test_generics():
    # list
    class GenericHParams(HParams):
        value: list[int]

    hparams = GenericHParams(value=[1, 2, 3])
    assert isinstance(hparams.value, list)
    assert all(isinstance(item, int) for item in hparams.value)
    assert hparams.value == [1, 2, 3]

    with pytest.raises(TypeError):
        # str is not int
        hparams = GenericHParams(value=[1, 2, "3"])

    # dict
    class GenericHParams(HParams):
        value: dict[str, int]

    hparams = GenericHParams(value={"a": 1, "b": 2})
    assert isinstance(hparams.value, dict)
    assert all(isinstance(key, str) for key in hparams.value.keys())
    assert all(isinstance(value, int) for value in hparams.value.values())
    assert hparams.value == {"a": 1, "b": 2}

    with pytest.raises(TypeError):
        # str is not int
        hparams = GenericHParams(value={"a": 1, "b": "2"})

    # tuple
    class GenericHParams(HParams):
        value: tuple[int, str]

    hparams = GenericHParams(value=(1, "2"))
    assert isinstance(hparams.value, tuple)
    assert isinstance(hparams.value[0], int)
    assert isinstance(hparams.value[1], str)
    assert hparams.value == (1, "2")

    with pytest.raises(TypeError):
        # str is not int
        hparams = GenericHParams(value=("1", "2"))

    with pytest.raises(TypeError):
        # int is not str
        hparams = GenericHParams(value=(1, 2))


def test_unsupported_generics():
    T = TypeVar("T")

    class MyGeneric(Generic[T]):
        pass

    with pytest.raises(NotImplementedError):
        class UnsupportedGenericHParams(HParams):
            value: MyGeneric[int]


def test_nested_generics():
    with pytest.raises(NotImplementedError):
        class NestedGenericHParams(HParams):
            value: list[list[int]]
