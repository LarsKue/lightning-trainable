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

    max_epochs = 10
    hparams = TrainableHParams(
        accelerator="cpu",
        max_epochs=10,
        batch_size=32,
        lr_scheduler=dict(
            name="onecyclelr",
            max_lr=1e-3,
            epochs=max_epochs,
            steps_per_epoch=len(train_data),
        )
    )
    model = SimpleTrainable(hparams, train_data=train_data)
    model.fit()
