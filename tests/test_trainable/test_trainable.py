
import pytest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from pathlib import Path

from lightning_trainable.hparams import HParams
from lightning_trainable.trainable import Trainable, TrainableHParams
from lightning_trainable.utils import find_checkpoint

from .fixtures import *


def test_fit(dummy_model):
    train_metrics = dummy_model.fit()

    assert isinstance(train_metrics, dict)
    assert "training/loss" in train_metrics
    assert "validation/loss" in train_metrics


def test_fit_fast(dummy_model):
    loss = dummy_model.fit_fast(device="cpu")

    assert torch.isclose(loss, torch.tensor(0.0))


def test_hparams_copy(dummy_hparams, dummy_hparams_cls):
    assert isinstance(dummy_hparams, HParams)
    assert isinstance(dummy_hparams, TrainableHParams)
    assert isinstance(dummy_hparams, dummy_hparams_cls)

    hparams_copy = dummy_hparams.copy()

    assert isinstance(hparams_copy, HParams)
    assert isinstance(dummy_hparams, TrainableHParams)
    assert isinstance(dummy_hparams, dummy_hparams_cls)


def test_hparams_invariant(dummy_model_cls, dummy_hparams):
    """ Ensure HParams are left unchanged after instantiation and training """
    hparams = dummy_hparams.copy()

    dummy_model1 = dummy_model_cls(hparams)

    assert hparams == dummy_hparams

    dummy_model1.fit()

    assert hparams == dummy_hparams


def test_checkpoint(dummy_model):
    # TODO: temp directory

    dummy_model.fit()

    checkpoint = find_checkpoint()

    assert Path(checkpoint).is_file()


def test_load_checkpoint(dummy_model_cls, dummy_hparams):
    dummy_model = dummy_model_cls(dummy_hparams)

    assert dummy_model._hparams_name == "hparams"

    dummy_model.fit()

    checkpoint = find_checkpoint()

    loaded_model = dummy_model_cls.load_from_checkpoint(checkpoint)

    assert isinstance(loaded_model, dummy_model_cls)
    assert loaded_model.hparams == dummy_hparams
    loaded_dict = loaded_model.state_dict()
    dummy_dict = dummy_model.state_dict()

    assert set(loaded_dict.keys()) == set(dummy_dict.keys())

    for key in loaded_dict.keys():
        assert torch.allclose(loaded_dict[key], dummy_dict[key])


def test_nested_checkpoint(dummy_model_cls, dummy_hparams_cls):

    class MyHParams(dummy_hparams_cls):
        pass

    class MyModel(dummy_model_cls):
        hparams: MyHParams

        def __init__(self, hparams):
            super().__init__(hparams)

    hparams = MyHParams()
    model = MyModel(hparams)

    assert model._hparams_name == "hparams"


def test_load_hparams(dummy_model):
    dummy_model.fit()

    checkpoint = find_checkpoint()

    hparams_file = Path(checkpoint).parent.parent / "hparams.yaml"
    assert hparams_file.is_file()
    hparams = dummy_model.hparams_type.from_yaml(hparams_file)
    assert hparams == dummy_model.hparams


def test_continue_training(dummy_model):
    print("Starting Training.")
    dummy_model.fit()

    print("Finished Training. Loading Checkpoint.")
    checkpoint = find_checkpoint()

    trained_model = dummy_model.__class__.load_from_checkpoint(checkpoint)

    print("Continuing Training.")
    trained_model.fit(
        trainer_kwargs=dict(max_epochs=2 * dummy_model.hparams.max_epochs),
        fit_kwargs=dict(ckpt_path=checkpoint)
    )

    print("Finished Continued Training.")

    # TODO: add check that the model was actually trained for 2x epochs


def test_lr_scheduler(dummy_model_cls, dummy_hparams):
    dummy_hparams.lr_scheduler = dict(
        name="OneCycleLR",
        interval="step",
        kwargs=dict(
            max_lr=1e-4,
        )
    )

    dummy_model = dummy_model_cls(dummy_hparams)

    dummy_model.fit()


def test_load_best_model(dummy_model_cls, dummy_hparams):
    dummy_hparams.model_checkpoint = dict(
        monitor="auto",
        save_last=True,
        every_n_epochs=1,
        save_top_k=5
    )
    dummy_model = dummy_model_cls(dummy_hparams)

    dummy_model.fit()
    dummy_model.__class__.load_best_checkpoint()