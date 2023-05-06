
import pytest

import torch

from lightning_trainable.modules import UNet, UNetHParams


@pytest.mark.parametrize("skip_mode", ["add", "concat", "none"])
@pytest.mark.parametrize("width", [32, 48, 64])
@pytest.mark.parametrize("height", [32, 48, 64])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_basic(skip_mode, channels, width, height):
    hparams = UNetHParams(
        input_shape=(channels, height, width),
        output_shape=(channels, height, width),
        down_blocks=[
            dict(channels=[16, 32], kernel_sizes=[3, 3, 3]),
            dict(channels=[32, 64], kernel_sizes=[3, 3, 3]),
        ],
        up_blocks=[
            dict(channels=[64, 32], kernel_sizes=[3, 3, 3]),
            dict(channels=[32, 16], kernel_sizes=[3, 3, 3]),
        ],
        bottom_widths=[32, 32],
        skip_mode=skip_mode,
        activation="relu",
    )
    unet = UNet(hparams)

    x = torch.randn(1, *hparams.input_shape)
    y = unet(x)
    assert y.shape == (1, *hparams.output_shape)


@pytest.mark.parametrize("skip_mode", ["concat", "none"])
@pytest.mark.parametrize("width", [32, 48, 64])
@pytest.mark.parametrize("height", [32, 48, 64])
@pytest.mark.parametrize("channels", [1, 3, 5])
def test_inconsistent_channels(skip_mode, channels, width, height):
    hparams = UNetHParams(
        input_shape=(channels, height, width),
        output_shape=(channels, height, width),
        down_blocks=[
            dict(channels=[16, 24], kernel_sizes=[3, 3, 3]),
            dict(channels=[48, 17], kernel_sizes=[3, 3, 3]),
        ],
        up_blocks=[
            dict(channels=[57, 31], kernel_sizes=[3, 3, 3]),
            dict(channels=[12, 87], kernel_sizes=[3, 3, 3]),
        ],
        bottom_widths=[32, 32],
        skip_mode=skip_mode,
        activation="relu",
    )
    unet = UNet(hparams)

    x = torch.randn(1, *hparams.input_shape)
    y = unet(x)
    assert y.shape == (1, *hparams.output_shape)
