# lightning-trainable

![Build status](https://github.com/LarsKue/lightning-trainable/workflows/Tests/badge.svg)

A light-weight trainable module for `pytorch-lightning`, aimed at fast prototyping.

This package is intended to further simplify the definition of your `LightningModules`
such that you only need to define a network, hyperparameters, and train metrics.

It also provides some default datasets that you can benchmark your models on.

## Install
Clone the repository

`git clone https://github.com/LarsKue/lightning-trainable`

and then use `pip` to install it in editable mode:

`pip install -e lightning-trainable/`

## Usage
### 1. Define your module and datasets, inheriting from `Trainable`:

```python
from lightning_trainable import Trainable, utils


class MyNetwork(Trainable):
    hparams: MyHParams  # only if you use custom hparams
    
    def __init__(self, hparams):
        super().__init__(hparams)
        self.network = utils.make_dense(
            [self.hparams.inputs, *self.hparams.layer_sizes, self.hparams.outputs],
            self.hparams.activation
        )
        self.train_data = TensorDataset(...)
        self.val_data = TensorDataset(...)

    def compute_metrics(self, batch, batch_idx):
        x, y = batch
        yhat = self.network(x)
        mse = F.mse_loss(yhat, y)

        return dict(
            loss=mse
        )
```

### 2. Define your model hparams, inheriting from `TrainableHParams`

```python
from lightning_trainable import TrainableHParams


class MyHParams(TrainableHParams):
    inputs: int = 28 * 28  # MNIST
    outputs: int = 10

    layer_sizes: list
    activation: str = "relu"
    dropout: float | int | None = None
```

### 3. Train your model with `model.fit()`
```python
hparams = MyHParams(
    layer_sizes=[32, 64, 32],
    max_epochs=10,
    batch_size=32,
)

model = MyNetwork(hparams)
model.fit()
```

## Datasets
We aim to provide a rich collection of both toy and benchmark datasets, which work out-of-the-box.

You can find datasets in `lightning_trainable/datasets`. Currently, mostly generative datasets are available.

For example, you can create an infinite, iterable dataset from a generative distribution like this:

```python
from lightning_trainable.datasets import *

dataset = HypershellsDataset()
```

## Experiment Launcher
(details follow)

## Additional Details
For more details, check out the documentation.
