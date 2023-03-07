
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm


class EpochProgressBar(ProgressBarBase):
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
