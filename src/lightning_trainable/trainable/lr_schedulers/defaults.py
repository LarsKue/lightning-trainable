
def get_defaults(scheduler_name, model, optimizer):
    match scheduler_name:
        case "OneCycleLR":
            max_lr = optimizer.defaults["lr"]
            total_steps = model.hparams.max_steps
            if total_steps == -1:
                total_steps = model.hparams.max_epochs * len(model.train_dataloader())
                total_steps = int(total_steps / model.hparams.accumulate_batches)

            return dict(
                max_lr=max_lr,
                total_steps=total_steps,
                interval="step"
            )
        case _:
            return dict()
