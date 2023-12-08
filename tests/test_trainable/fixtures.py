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