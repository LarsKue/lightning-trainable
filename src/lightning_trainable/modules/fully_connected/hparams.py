
from lightning_trainable.hparams import HParams, Choice


class FullyConnectedNetworkHParams(HParams):
    input_dims: int | Choice("lazy")
    output_dims: int

    layer_widths: list[int]
    activation: str = "relu"
