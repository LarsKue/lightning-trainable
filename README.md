# lightning-trainable
A light-weight trainable module for `pytorch-lightning`, aimed at fast prototyping.

This package is intended to further simplify the definition of your `LightningModules` such that you only need to define a network, hyperparameters, and a loss function.

It also provides some default benchmarks that you can run your models on.

## Install
Clone the repository

`git clone https://github.com/LarsKue/lightning-trainable`

and then use `pip` to install it in editable mode:

`pip install -e lightning-trainable/`

## Usage
### 1. Define your module and datasets, inheriting from `Trainable`:
```python
from trainable import Trainable, utils

class MyNetwork(Trainable):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.network = utils.make_dense(
            [self.hparams.inputs, *self.hparams.layer_sizes, self.hparams.outputs],
            self.hparams.activation
        )
        self.train_data = TensorDataset(...)
        self.val_data = TensorDataset(...)
    
    def loss(self, batch, batch_idx):
        x, y = batch
        yhat = self.network(x)
        return F.mse_loss(yhat, y)
```

### 2. Define your model hparams, inheriting from `TrainableHParams`
```python
from trainable import TrainableHParams

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
)

model = MyNetwork(hparams)
model.fit()
```

## Benchmarks
Benchmarks provide an easy, clean way to test the inference performance of your models.

*Note:* Benchmarks are still a work-in-progress,
and as such the process of using them may be either incomplete or at least not pretty.

You can find benchmarks in `experiments/benchmarks`.
Currently, only generative benchmarks are available.
You can create an infinite, iterable dataset from a generative benchmark like this:

```python
from experiments.benchmarks import *
dataset = HypershellsDataset()
```

And run your model on this benchmark with `benchmark_grid`:

```python
results_df = benchmark_grid(models=[MyNetwork], parameters=[hparams], datasets=[dataset])
```

As the name implies, `benchmark_grid` supports passing any number of model classes,
parameter sets, or datasets. The benchmarks will be run in a meshgrid over what you pass.

## Additional Details
For more details, check out the documentation.