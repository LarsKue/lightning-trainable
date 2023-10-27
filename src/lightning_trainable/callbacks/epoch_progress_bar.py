
from lightning.pytorch.callbacks import ProgressBar
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

from lightning_trainable.utils import deprecate


@deprecate("EpochProgressBar causes issues when continuing training or using multi-GPU. "
           "Use the default Lightning ProgressBar instead.")
class EpochProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        self.bar = Tqdm(
            desc="Epoch",
            leave=True,
            dynamic_ncols=True,
            total=trainer.max_epochs,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        self.bar.update(1)
        self.bar.set_description(f"Epoch {trainer.current_epoch}")
        self.bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer, pl_module):
        self.bar.close()


@deprecate("StepProgressBar causes issues when continuing training or using multi-GPU. "
           "Use the default Lightning ProgressBar instead.")
class StepProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()
        self.bar = None

    def on_train_start(self, trainer, pl_module):
        batches = len(pl_module.train_dataloader())

        if trainer.max_epochs is not None:
            total = batches * trainer.max_epochs
        else:
            total = trainer.max_steps

        self.bar = Tqdm(
            desc="Step",
            leave=True,
            dynamic_ncols=True,
            total=total,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.bar.update(1)
        self.bar.set_description(f"Epoch {trainer.current_epoch:04d}: Batch {batch_idx:04d}")
        self.bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer, pl_module):
        self.bar.close()
