from lightning_trainable.hparams import HParams, Choice


class ConvolutionalBlockHParams(HParams):
    channels: list[int]
    kernel_sizes: list[int]
    activation: str = "relu"
    padding: str | int = 0
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    padding_mode: str = "zeros"

    pool: bool = False
    pool_direction: Choice("up", "down") = "down"
    pool_position: Choice("first", "last") = "last"

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        if not len(hparams.channels) == len(hparams.kernel_sizes):
            raise ValueError(f"{cls.__name__} needs same number of channels and kernel sizes.")

        return hparams
