
import pytest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from lightning_trainable.trainable import Trainable, TrainableHParams


@pytest.fixture
def dummy_dataset():
    return TensorDataset(torch.randn(128, 2))


@pytest.fixture
def dummy_network():
    return nn.Linear(2, 2)


@pytest.fixture
def dummy_hparams_cls():
    class DummyHParams(TrainableHParams):
        max_epochs: int = 10
        batch_size: int = 4
        accelerator: str = "cpu"

    return DummyHParams


@pytest.fixture
def dummy_hparams(dummy_hparams_cls):
    return dummy_hparams_cls()


@pytest.fixture
def dummy_model_cls(dummy_network, dummy_dataset, dummy_hparams_cls):
    class DummyModel(Trainable):
        hparams: dummy_hparams_cls

        def __init__(self, hparams):
            super().__init__(hparams, train_data=dummy_dataset, val_data=dummy_dataset, test_data=dummy_dataset)
            self.network = dummy_network

        def compute_metrics(self, batch, batch_idx) -> dict:
            return dict(
                loss=torch.tensor(0.0, requires_grad=True)
            )

    return DummyModel


@pytest.fixture
def dummy_model(dummy_model_cls, dummy_hparams):
    return dummy_model_cls(dummy_hparams)


def test_fit(dummy_model):
    train_metrics = dummy_model.fit()

    assert isinstance(train_metrics, dict)
    assert "training/loss" in train_metrics
    assert "validation/loss" in train_metrics


def test_fit_fast(dummy_model):
    loss = dummy_model.fit_fast(device="cpu")

    assert torch.isclose(loss, torch.tensor(0.0))


def test_hparams_invariant(dummy_model_cls, dummy_hparams):
    """ Ensure HParams are left unchanged after instantiation and training """
    hparams = dummy_hparams.copy()

    dummy_model1 = dummy_model_cls(hparams)

    assert hparams == dummy_hparams

    dummy_model1.fit()

    assert hparams == dummy_hparams


def test_checkpoint(dummy_model):
    dummy_model.fit()

    trained_model = dummy_model.find_and_load_from_checkpoint()


def test_nested_checkpoint(dummy_model_cls, dummy_hparams_cls):

    class MyHParams(dummy_hparams_cls):
        pass

    class MyModel(dummy_model_cls):
        def __init__(self, hparams):
            super().__init__(hparams)

    hparams = MyHParams()
    model = MyModel(hparams)

    assert model._hparams_name == "hparams"

    model.fit()

    model.find_and_load_from_checkpoint()
