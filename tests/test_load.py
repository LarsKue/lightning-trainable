import string
import random
from pathlib import Path

from torch.nn import Parameter
import torch

from lightning_trainable import Trainable, TrainableHParams


def test_load_checkpoint():
    class LoadHParams(TrainableHParams):
        a: int
        b: int = 1

    class LoadModel(Trainable):
        hparams: LoadHParams

        def __init__(self, hparams: LoadHParams | dict):
            super().__init__(hparams)
            self.s = Parameter(torch.ones(1))

        def compute_metrics(self, batch, batch_idx) -> dict:
            return {"loss": self.s ** 2}

    # Random checkpoint
    while True:
        random_name = "".join(random.choices(string.ascii_letters, k=10))
        path = Path.cwd() / f"ckpt-{random_name}.ckpt"
        if not path.exists():
            break

    model = LoadModel(dict(
        a=1,
        max_epochs=1,
        batch_size=1,
    ))
    trainer = model.configure_trainer()
    trainer.fit(model)
    trainer.save_checkpoint(path)

    loaded_model = LoadModel.load_from_checkpoint(path)
    assert loaded_model.hparams == model.hparams
    assert loaded_model.s == model.s
