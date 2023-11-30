# lightning-trainable

![Build status](https://github.com/LarsKue/lightning-trainable/workflows/Tests/badge.svg)

A light-weight trainable module for `pytorch-lightning`, aimed at fast prototyping,
particularly for generative models.

This package is intended to further simplify the definition of your `LightningModules`
such that you only need to define a network, hyperparameters, and train metrics.

It also provides some default datasets and module building blocks.

## Install
Clone the repository

`git clone https://github.com/LarsKue/lightning-trainable`

and then use `pip` to install it in editable mode:

`pip install -e lightning-trainable/`

## Usage
### 1. Define your module and datasets, inheriting from `Trainable`:

```python
import torch.nn.functional as F
from lightning_trainable.trainable import Trainable
from lightning_trainable.modules import FullyConnectedNetwork
from lightning_trainable.metrics import accuracy


class MyClassifier(Trainable):
    # specify your hparams class
    hparams: MyClassifierHParams
    
    def __init__(self, hparams, **datasets):
        super().__init__(hparams, **datasets)
        self.network = FullyConnectedNetwork(self.hparams.network_hparams)

    def compute_metrics(self, batch, batch_idx):
        # Compute loss and analysis metrics on a batch
        x, y = batch
        yhat = self.network(x)
        
        cross_entropy = F.cross_entropy(yhat, y)
        top1_accuracy = accuracy(yhat, y, k=1)
        
        metrics = {
            "loss": cross_entropy,
            "cross_entropy": cross_entropy,
            "top1_accuracy": top1_accuracy,
        }
        
        if self.hparams.network_hparams.output_size > 5:
            # only log top-5 accuracy if it can be computed
            metrics["top5_accuracy"] = accuracy(yhat, y, k=5)

        return metrics
```

### 2. Define your model hparams, inheriting from `TrainableHParams`

**New**: You can now use generic type hints in your `HParams`! 

```python
from lightning_trainable.trainable import TrainableHParams
from lightning_trainable.modules import FullyConnectedNetworkHParams


class MyClassifierHParams(TrainableHParams):
    network_hparams: FullyConnectedNetworkHParams
```

### 3. Train your model with `model.fit()`
```python
hparams = MyClassifierHParams(
    network_hparams=dict(
        input_dims=28 * 28,
        output_dims=10,
        layer_widths=[1024, 512, 256, 128],
        activation="relu",
    ),
)

model = MyClassifier(hparams)
model.fit()
```


### Done! Now your model can train with automatic metrics logging, hparams logging and checkpointing!

To look at metrics, use [Tensorboard](https://www.tensorflow.org/tensorboard), e.g. from the Terminal:
```
tensorboard --logdir lightning_logs/
```

To load a model checkpoint, use

```python
checkpoint = lightning_trainable.utils.find_checkpoint(version=7, epoch="last")
MyModel.load_from_checkpoint(checkpoint)
```

Here, you can specify a root directory, version, epoch and step number
to load your precise checkpoint, or simply load the latest of each.

## Datasets
We aim to provide a rich collection of both toy and benchmark datasets, which work out-of-the-box.

You can find datasets in `lightning_trainable/datasets`. Currently, mostly generative datasets are available.

For example, you can create an infinite, iterable dataset from a generative distribution like this:

```python
from lightning_trainable.datasets import *

dataset = HypershellsDataset()
```

## Modules
We also provide a collection of modules that you can use to build your models,
e.g. `FullyConnectedNetwork` or `UNet`.
Modules come with pre-packaged `HParams` classes that you can use to configure them.

For example, you can create a fully-connected network like this:

```python
from lightning_trainable.modules import FullyConnectedNetwork

hparams = dict(
    input_dims=28 * 28,
    output_dims=10,
    layer_widths=[1024, 512, 256, 128],
    activation="relu",
)

network = FullyConnectedNetwork(hparams)
```

## Experiment Launcher
(details follow)

## Additional Details
For more details, check out the documentation.
