
import torch
import torch.nn as nn

from lightning_trainable.hparams import HParams
import lightning_trainable.utils as utils

from .hparams_module import HParamsModule


class DenseModuleHParams(HParams):
    inputs: int
    outputs: int
    layer_widths: list[int]
    activations: str | list = "relu"

    @classmethod
    def validate_parameters(cls, hparams):
        hparams = super().validate_parameters(hparams)
        activations = hparams["activations"]
        widths = hparams["layer_widths"]
        if isinstance(activations, list) and len(activations) != len(widths):
            raise ValueError(f"Need one activation for each hidden layer ({len(widths)}), "
                             f"but got {len(activations)}.")

        return hparams


class DenseModule(HParamsModule):
    hparams: DenseModuleHParams

    def __init__(self, hparams: DenseModuleHParams | dict):
        super().__init__(hparams)
        self.network = self.configure_network()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch)

    def configure_network(self) -> nn.Module:
        widths = [self.hparams.inputs, *self.hparams.layer_widths, self.hparams.outputs]
        if isinstance(self.hparams.activations, str):
            activations = [self.hparams.activations] * len(self.hparams.layer_widths)
        else:
            activations = self.hparams.activations

        layers = []
        for i, (w1, w2) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(w1, w2))

            if i < len(activations):
                layers.append(self.configure_activation(activations[i]))

        return nn.Sequential(*layers)

    def configure_activation(self, activation: str) -> nn.Module:
        return utils.get_activation(activation)()
