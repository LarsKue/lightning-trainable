
from .accuracy import accuracy


def error(logits, targets, *, k=1):
    return 1.0 - accuracy(logits, targets, k=k)
