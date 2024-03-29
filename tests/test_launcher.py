from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from lightning_trainable import Trainable, TrainableHParams
from lightning_trainable.launcher.fit import main
from lightning_trainable.launcher.grid import GridLauncher, status_count_counter
from lightning_trainable.launcher.utils import parse_config_dict


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
        "accelerator='cpu'",
        "--name", "{model_name};{max_epochs}"
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
    results = launcher.run_configs_and_wait(config_list, cli_args=["--name", "{model_name};{max_epochs}"])
    assert status_count_counter(results) == {0: 2}


def test_fit_start_from():
    model = BasicTrainable(dict(
        domain=[-3, 3],
        max_epochs=1,
        batch_size=128,
        accelerator="cpu"
    ))
    ckpt_name = "test_launcher.ckpt"
    trainer = model.configure_trainer()
    trainer.fit(model)
    trainer.save_checkpoint(ckpt_name)

    # Continue running, but with larger domain (other variables should be loaded)
    main([
        "model=tests.test_launcher.BasicTrainable",
        "domain=[-5, 5]",
        "--name", "{model_name};{max_epochs}",
        "--start-from", ckpt_name
    ])

    # TODO: use pytest resources for the checkpoint file


def test_list_hparam_append():
    config_dict = parse_config_dict([*{
        "list": [1, 2, 3],
        "list.+": -1,
    }.items()])
    assert config_dict["list"] == [1, 2, 3, -1]


def test_gradient_regex():
    common_args = [
        "model=tests.test_launcher.BasicTrainable",
        "domain=[-5, 5]",
        "max_epochs=1",
        "batch_size=128",
        "accelerator='cpu'",
        "--name", "{model_name};{max_epochs}",
    ]

    with pytest.raises(RuntimeError):
        main(common_args + [
            # No gradient to any parameter causes gradient-free loss
            "--gradient-regex", "$^"
        ])

    main(common_args + [
        "--gradient-regex", "^model.0.weight"
    ])
