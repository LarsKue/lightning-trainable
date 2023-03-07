
from .trainable import Trainable, TrainableHParams

import torch.nn as nn


class DenseTrainableHParams(TrainableHParams):
    inputs: int
    layer_widths: list
    outputs: int
    activation: str = "relu"


class DenseTrainable(Trainable):
    def __init__(self, hparams: DenseTrainableHParams | dict, **kwargs):
        if not isinstance(hparams, DenseTrainableHParams):
            hparams = DenseTrainableHParams(**hparams)

        super().__init__(hparams, **kwargs)

        self.network = self.configure_network()

    def configure_network(self):
        widths = [self.hparams.inputs, *self.hparams.layer_widths, self.hparams.outputs]

        layers = []
        for (w1, w2) in zip(widths[:-1], widths[1:]):
            layers.append(nn.Linear(w1, w2))
            layers.append(self.configure_activation())

        # remove last activation
        layers = layers[:-1]

        return nn.Sequential(*layers)

    def configure_activation(self):
        match self.hparams.activation.lower():
            case "relu":
                return nn.ReLU()
            case "elu":
                return nn.ELU()
            case "selu":
                return nn.SELU()
            case "leakyrelu":
                return nn.LeakyReLU()
            case "tanh":
                return nn.Tanh()
            case "sigmoid":
                return nn.Sigmoid()
            case other:
                raise NotImplementedError(f"{self.__class__.__name__} does not support activation '{other}'.")
