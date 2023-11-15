
from lightning_trainable.meta import issues_url


def get_config(scheduler_name):
    match scheduler_name:
        case "OneCycleLR":
            return {
                "interval": "step",
                "frequency": 1,
                "monitor": "validation/loss",
                "strict": True,
            }
        case other:
            raise NotImplementedError(f"Unrecognized Scheduler: '{other}'. "
                                      f"Please file an issue at {issues_url} if you need this scheduler.")


def get_kwargs(scheduler_name, model, optimizer):
    match scheduler_name:
        case "OneCycleLR":
            kwargs = dict()
            kwargs["max_lr"] = optimizer.defaults["lr"]
            if model.hparams.max_steps != -1:
                kwargs["total_steps"] = model.hparams.max_steps // model.hparams.accumulate_batches
            else:
                kwargs["total_steps"] = model.hparams.max_epochs * int(len(model.train_dataloader()) / model.hparams.accumulate_batches)

            return kwargs
        case other:
            raise NotImplementedError(f"Unrecognized Scheduler: '{other}'. "
                                      f"Please file an issue at {issues_url} if you need this scheduler.")
