def get_config(scheduler_name):
    match scheduler_name:
        case "OneCycleLR":
            return {
                "interval": "step",
                "frequency": 1,
                "monitor": "validation/loss",
                "strict": True,
            }
        case _:
            return {}


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
        case _:
            return {}
