import torch
from torch.utils.data import TensorDataset, Dataset

from lightning_trainable import Trainable, TrainableHParams


def test_instantiate():
    hparams = TrainableHParams(max_epochs=10, batch_size=32)
    Trainable(hparams)


def test_simple_model():
    class SimpleTrainable(Trainable):
        def __init__(self, hparams: TrainableHParams | dict,
                     train_data: Dataset = None,
                     val_data: Dataset = None,
                     test_data: Dataset = None
                     ):
            super().__init__(hparams, train_data, val_data, test_data)
            self.param = torch.nn.Parameter(torch.randn(8, 1))

        def compute_metrics(self, batch, batch_idx) -> dict:
            return {
                "loss": ((batch[0] @ self.param) ** 2).mean()
            }

    train_data = TensorDataset(torch.randn(128, 8))

    hparams = TrainableHParams(
        accelerator="cpu",
        max_epochs=10,
        batch_size=32,
        lr_scheduler="OneCycleLR"
    )
    model = SimpleTrainable(hparams, train_data=train_data)
    model.fit()

    model.load_checkpoint()


def test_double_train():
    class SimpleTrainable(Trainable):
        def __init__(self, hparams: TrainableHParams | dict,
                     train_data: Dataset = None,
                     val_data: Dataset = None,
                     test_data: Dataset = None
                     ):
            super().__init__(hparams, train_data, val_data, test_data)
            self.param = torch.nn.Parameter(torch.randn(8, 1))

        def compute_metrics(self, batch, batch_idx) -> dict:
            return {
                "loss": ((batch[0] @ self.param) ** 2).mean()
            }

    hparams = TrainableHParams(
        accelerator="cpu",
        max_epochs=1,
        batch_size=8,
        optimizer=dict(
            name="Adam",
            lr=1e-3,
        )
    )

    train_data = TensorDataset(torch.randn(128, 8))

    t1 = SimpleTrainable(hparams, train_data=train_data, val_data=train_data)
    t1.fit()

    t2 = SimpleTrainable(hparams, train_data=train_data, val_data=train_data)
    t2.fit()
