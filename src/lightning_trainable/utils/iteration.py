
import builtins
import itertools

from typing import Iterable


def flatten(iterable):
    """
    Recursively flatten an iterable.
    """
    for item in iterable:
        if isinstance(item, Iterable):
            yield from flatten(item)
        else:
            yield item


def zip(*iterables, exhaustive: bool = False, nested: bool = True):
    match (exhaustive, nested):
        case (True, True):
            # both exhaustive and nested
            fill_value = object()

            # TODO: improve this bs
            class zip_longest_no_fill(itertools.zip_longest):
                def __next__(self):
                    values = super().__next__()
                    return tuple(value for value in values if value is not fill_value)

            yield from zip_longest_no_fill(*iterables, fillvalue=fill_value)
        case (True, False):
            # exhaustive but not nested
            iterators = [iter(iterable) for iterable in iterables]
            i = 0

            while True:
                iterator = iterators[i]

                try:
                    yield next(iterator)
                    i += 1
                except StopIteration:
                    iterators.pop(i)
                    if not iterators:
                        return

                i = i % len(iterators)

        case (False, True):
            # not exhaustive but nested
            yield from builtins.zip(*iterables)
        case (False, False):
            # not exhaustive and not nested
            yield from itertools.chain.from_iterable(builtins.zip(*iterables))


def test_zip():
    x = [1, 4, 7]
    y = [2, 5, 8]
    z = [3, 6, 9, 10]

    assert list(zip(x, y, z, nested=True, exhaustive=True)) == [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10,)]
    assert list(zip(x, y, z, nested=True, exhaustive=False)) == list(builtins.zip(x, y, z))
    assert list(zip(x, y, z, nested=False, exhaustive=True)) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert list(zip(x, y, z, nested=False, exhaustive=False)) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
