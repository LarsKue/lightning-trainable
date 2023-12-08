
import pytest

import torch
import torch.nn as nn


from lightning_trainable.modules import FullyConnectedNetwork

from .fixtures import *


def assert_modules_equal(expected, actual):
    assert len(expected) == len(actual)

    for e, a in zip(expected, actual):
        assert type(e) is type(a)

        # hacky way to check the extra_repr
        assert str(e) == str(a)

    # check that the state dicts are compatible
    missing_keys, unexpected_keys = expected.load_state_dict(actual.state_dict(), strict=False)

    assert not missing_keys
    assert not unexpected_keys


def test_basic():
    hparams = dict(
        input_dims=32,
        output_dims=64,
        layer_widths=[64, 128, 64],
        activation="relu",
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64)
    )
    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_lazy():
    hparams = dict(
        input_dims="lazy",
        output_dims=64,
        layer_widths=[64, 128, 64],
        activation="relu",
    )

    expected = nn.Sequential(
        nn.LazyLinear(64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 64)
    )
    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_activation():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="selu",
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.SELU(inplace=True),
        nn.Linear(64, 128),
        nn.SELU(inplace=True),
        nn.Linear(128, 64),
        nn.SELU(inplace=True),
        nn.Linear(64, 32)
    )
    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_activation_last_layer():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="selu",
        last_layer_activation=True,
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.SELU(inplace=True),
        nn.Linear(64, 128),
        nn.SELU(inplace=True),
        nn.Linear(128, 64),
        nn.SELU(inplace=True),
        nn.Linear(64, 32),
        nn.SELU(inplace=True),
    )
    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_dropout():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="relu",
        dropout=0.25,
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(64, 32)
    )

    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_dropout_last_layer():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="relu",
        dropout=0.25,
        last_layer_dropout=True,
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(64, 32),
        nn.Dropout(0.25),
    )

    actual = FullyConnectedNetwork(hparams).network

    print()
    print(expected)
    print("=" * 80)
    print(actual)
    print()

    assert_modules_equal(expected, actual)


def test_norm():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="relu",
        norm="batch",
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(64),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32)
    )

    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_norm_last_layer():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="relu",
        norm="batch",
        last_layer_norm=True,
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(64),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
    )

    actual = FullyConnectedNetwork(hparams).network

    assert_modules_equal(expected, actual)


def test_everything():
    hparams = dict(
        input_dims=32,
        output_dims=32,
        layer_widths=[64, 128, 64],
        activation="relu",
        norm="batch",
        dropout=0.25,
        last_layer_activation=True,
        last_layer_dropout=True,
        last_layer_norm=True,
    )

    expected = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.BatchNorm1d(64),
        nn.Linear(64, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.BatchNorm1d(64),
        nn.Linear(64, 32),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.BatchNorm1d(32),
    )

    actual = FullyConnectedNetwork(hparams).network

    print()
    print(expected)
    print("=" * 80)
    print(actual)
    print()

    assert_modules_equal(expected, actual)

