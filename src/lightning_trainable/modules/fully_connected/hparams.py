
from lightning_trainable.hparams import HParams, Choice


class FullyConnectedNetworkHParams(HParams):
    input_dims: int | Choice("lazy")
    output_dims: int

    layer_widths: list[int]
    activation: str = "relu"

    norm: Choice("none", "batch", "layer") = "none"
    dropout: float = 0.0

    last_layer_dropout: bool = False
    last_layer_activation: bool = False
