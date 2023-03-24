import torch
from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.launcher.fit import main
from torch.utils.data import TensorDataset


class BasicTrainableHParams(TrainableHParams):
    data_set_name: str


class BasicTrainable(Trainable):
    def __init__(self, hparams: BasicTrainableHParams | dict):
        if not isinstance(hparams, BasicTrainableHParams):
            hparams = BasicTrainableHParams(**hparams)

        assert hparams.data_set_name == "sine"
        x = torch.linspace(-5, 5, 1000)[:, None]
        y = torch.sin(x)[:, None]
        train_data = TensorDataset(x, y)

        super().__init__(hparams, train_data=train_data)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def compute_metrics(self, batch, batch_idx) -> dict:
        x, y = batch
        return {
            "loss": ((self.model(x) - y) ** 2).mean()
        }


def test_fit_launcher():
    main([
        "model=tests.test_launcher.BasicTrainable",
        "batch_size=128",
        "max_epochs=1",
        "data_set_name='sine'",
        "accelerator='cpu'"
    ])
