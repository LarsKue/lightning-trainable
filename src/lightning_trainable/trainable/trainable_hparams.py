
from lightning_trainable.hparams import HParams
from lightning.pytorch.profilers import Profiler


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
    track_grad_norm: int | None = 2
    gradient_clip: float | int | None = None
    profiler: str | Profiler | None = None
    num_workers: int = 4
    pin_memory: bool = True
