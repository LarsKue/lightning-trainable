
from lightning_trainable.hparams import AttributeDict


def test_nested_dicts():
    d = AttributeDict(dictionary=dict())

    assert isinstance(d.dictionary, AttributeDict)

    d.dictionary.key = 1

    assert "key" in d.dictionary
    assert d.dictionary.key == 1
