import os
from pathlib import Path

import torch
from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.launcher.fit import main
from torch.utils.data import TensorDataset

from lightning_trainable.launcher.grid import GridLauncher


class BasicTrainableHParams(TrainableHParams):
    domain: list


class BasicTrainable(Trainable):
    def __init__(self, hparams: BasicTrainableHParams | dict):
        if not isinstance(hparams, BasicTrainableHParams):
            hparams = BasicTrainableHParams(**hparams)

        x = torch.linspace(*hparams.domain, 1000)[:, None]
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
        str(Path(__file__).parent / "test_launcher_config.yaml"),
        "max_epochs=1",
        "domain=[-5, 5]",
        "accelerator='cpu'"
    ])


def test_grid_launcher():
    launcher = GridLauncher()
    config_list = launcher.grid_spec_to_list([
        ("model", ["tests.test_launcher.BasicTrainable"]),
        ([Path(__file__).parent / "test_launcher_config.yaml"]),
        ("max_epochs", [1]),
        ("domain", [[-5, 5], [-3, 3]]),
        ("accelerator", ['cpu'])
    ])
    results = launcher.run_configs_and_wait(config_list)
    print(results[0].stderr.decode())
