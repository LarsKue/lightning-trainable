from lightning_trainable.hparams import HParams


class SimpleUNetHParams(HParams):
    input_shape: tuple[int, int, int]
    conditions: int = 0

    channels: list[int]
    kernel_sizes: list[int]
    fc_widths: list[int]
    activation: str = "ReLU"

    block_size: int = 2

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        if len(hparams.channels) + 2 != len(hparams.kernel_sizes):
            raise ValueError(f"Number of channels ({len(hparams.channels)}) + 2 must be equal "
                             f"to the number of kernel sizes ({len(hparams.kernel_sizes)})")

        return hparams
