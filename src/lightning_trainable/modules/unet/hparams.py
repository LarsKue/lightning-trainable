from lightning_trainable.hparams import HParams, Choice


class UNetBlockHParams(HParams):
    channels: list[int]
    kernel_sizes: list[int]

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        if len(hparams.channels) + 1 != len(hparams.kernel_sizes):
            raise ValueError(f"Block needs one more kernel size than channels.")

        return hparams


class UNetHParams(HParams):
    # input/output image size, not including batch dimensions
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]

    # list of hparams for individual down/up blocks
    down_blocks: list[dict | UNetBlockHParams]
    up_blocks: list[dict | UNetBlockHParams]

    # hidden layer sizes for the bottom, fully connected part of the UNet
    bottom_widths: list[int]

    # skip connection mode
    skip_mode: Choice("add", "concat", "none") = "add"
    activation: str = "relu"

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)

        for i in range(len(hparams.down_blocks)):
            hparams.down_blocks[i] = UNetBlockHParams(**hparams.down_blocks[i])
        for i in range(len(hparams.up_blocks)):
            hparams.up_blocks[i] = UNetBlockHParams(**hparams.up_blocks[i])

        url = "https://github.com/LarsKue/lightning-trainable/"
        if hparams.input_shape[1:] != hparams.output_shape[1:]:
            raise ValueError(f"Different image sizes for input and output are not yet supported. "
                             f"If you need this feature, please file an issue or pull request at {url}.")

        if hparams.input_shape[1] % 2 or hparams.input_shape[2] % 2:
            raise ValueError(f"Odd input shape is not yet supported. "
                             f"If you need this feature, please file an issue or pull request at {url}.")

        minimum_size = 2 ** len(hparams.down_blocks)
        if hparams.input_shape[1] < minimum_size or hparams.input_shape[2] < minimum_size:
            raise ValueError(f"Input shape {hparams.input_shape[1:]} is too small for {len(hparams.down_blocks)} "
                             f"down blocks. Minimum size is {(minimum_size, minimum_size)}.")

        if hparams.skip_mode == "add":
            # ensure matching number of channels for down output as up input
            for i, (down_block, up_block) in enumerate(zip(hparams.down_blocks, reversed(hparams.up_blocks))):
                if down_block["channels"][-1] != up_block["channels"][0]:
                    raise ValueError(f"Output channels of down block {i} must match input channels of up block "
                                     f"{len(hparams.up_blocks) - (i + 1)} for skip mode '{hparams.skip_mode}'.")

        return hparams
