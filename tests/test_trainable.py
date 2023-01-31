
from trainable import Trainable


def test_create():
    t = Trainable(max_epochs=10, batch_size=32)
