
import warnings

from lightning_trainable.hparams import HParams
from lightning.pytorch.profilers import Profiler
from lightning_trainable.utils import deprecate



class TrainableHParams(HParams):
    # name of the loss, your `compute_metrics` should return a dict with this name in its keys
    loss: str = "loss"

    accelerator: str = "gpu"
    devices: int = 1
    max_epochs: int | None
    max_steps: int = -1
    optimizer: str | dict | None = "adam"
    lr_scheduler: str | dict | None = None
    batch_size: int
    accumulate_batches: int = 1
    track_grad_norm: int | None = None
    gradient_clip: float | int | None = None
    profiler: str | Profiler | None = None
    num_workers: int = 4
    pin_memory: bool | None = None
    early_stopping: int | None = None

    @classmethod
    def _migrate_hparams(cls, hparams):
        if "accumulate_batches" in hparams and hparams["accumulate_batches"] is None:
            deprecate("accumulate_batches changed default value: None -> 1")
            hparams["accumulate_batches"] = 1

        if "optimizer" in hparams:
            match hparams["optimizer"]:
                case str() as name:
                    if name == name.lower():
                        deprecate("optimizer name is now case-sensitive.")
                    if name == "adam":
                        hparams["optimizer"] = "Adam"
                case dict() as kwargs:
                    name = kwargs["name"]
                    if name == name.lower():
                        deprecate("optimizer name is now case-sensitive.")
                    if name == "adam":
                        hparams["optimizer"]["name"] = "Adam"

        if "lr_scheduler" in hparams:
            match hparams["lr_scheduler"]:
                case str() as name:
                    if name == name.lower():
                        deprecate("lr_scheduler name is now case-sensitive.")
                    if name == "onecyclelr":
                        hparams["lr_scheduler"] = "OneCycleLR"
                case dict() as kwargs:
                    name = kwargs["name"]
                    if name == name.lower():
                        deprecate("lr_scheduler name is now case-sensitive.")
                    if name == "onecyclelr":
                        hparams["lr_scheduler"]["name"] = "OneCycleLR"

        return hparams
