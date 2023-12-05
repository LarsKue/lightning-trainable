
import pytest

from lightning_trainable.hparams import AttributeDict


def test_nested_dicts_on_construct():
    d = AttributeDict(dictionary=dict())

    assert isinstance(d.dictionary, AttributeDict)

    d.dictionary.key = 1

    assert "key" in d.dictionary
    assert d.dictionary.key == 1
    assert d.dictionary["key"] == 1
    assert d.dictionary == dict(key=1)


def test_as_dict():
    d = AttributeDict(dictionary=dict(key=1))
    d = d.as_dict()

    assert type(d) is dict
    assert type(d["dictionary"]) is dict
    assert d["dictionary"]["key"] == 1
    assert d["dictionary"] == dict(key=1)
    assert d == dict(dictionary=dict(key=1))

    assert not hasattr(d, "dictionary")


def test_nested_dicts_on_assign():
    d = AttributeDict()

    d.dictionary = dict()

    assert isinstance(d.dictionary, AttributeDict)

    d.dictionary.key = 1

    assert "key" in d.dictionary
    assert d.dictionary.key == 1
    assert d.dictionary["key"] == 1
    assert d.dictionary == dict(key=1)



