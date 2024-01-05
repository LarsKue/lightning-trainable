
def get_kwargs(optimizer_name, model):
    match optimizer_name:
        case _:
            return dict(
                params=model.parameters(),
            )
