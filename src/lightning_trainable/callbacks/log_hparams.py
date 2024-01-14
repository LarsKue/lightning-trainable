
import warnings

import torch
import lightning

from torch.utils.data import DataLoader


class LogHParamsCallback(lightning.Callback):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_train_start(self, trainer, model) -> None:
        # get hparams metrics with a test batch
        model.eval()

        if model.val_data is not None:
            prefix = "validation"
            if not isinstance(trainer.val_dataloaders, DataLoader):
                validation_loader = trainer.val_dataloaders[0]
            else:
                validation_loader = trainer.val_dataloaders

            test_batch = next(iter(validation_loader))
        elif model.train_data is not None:
            # no validation data, fallback to train data
            prefix = "training"
            test_batch = next(iter(trainer.train_dataloader))
        else:
            warnings.warn("Could not log hyperparameters because no train or validation data was provided.")
            return

        test_batch = model.transfer_batch_to_device(test_batch, model.device, 0)

        metrics = model.validation_metrics(test_batch, 0)

        model.logger.log_hyperparams(model.hparams, {f"{prefix}/{key}": value for key, value in metrics.items()})

        model.train()
