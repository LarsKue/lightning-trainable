
from trainable import Trainable, TrainableHParams
from .protocols import Distribution

from torch.utils.data import TensorDataset


class Benchmark:
    pass


class GenerativeBenchmark(Benchmark):
    def __init__(self, module_cls: type(Trainable), distribution: Distribution):
        self.module_cls = module_cls
        self.distribution = distribution

    def run(self, hparams: TrainableHParams, train_samples: int = 10_000, val_samples: int = 1_000, **fit_kwargs):
        train_data = TensorDataset(self.distribution.sample((train_samples,)))
        val_data = TensorDataset(self.distribution.sample((val_samples,)))
        module = self.module_cls(hparams, train_data=train_data, val_data=val_data)
        validation_loss = module.fit(**fit_kwargs)

        # TODO: plot samples? this interface is a bit meh

        return validation_loss
