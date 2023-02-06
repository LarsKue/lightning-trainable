from trainable import Trainable, TrainableHParams


def test_instantiate():
    hparams = TrainableHParams(max_epochs=10, batch_size=32)
    Trainable(hparams)
