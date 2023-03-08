from lightning_trainable.utils import make_dense


def test_make_dense_linear():
    widths = [1, 2]
    network = make_dense(widths, "relu")
    assert len(network) == 1
    assert network[0].weight.shape == tuple(widths[::-1])


def test_make_dense_single_hidden():
    widths = [1, 2, 3]
    network = make_dense(widths, "relu")
    assert len(network) == 3
    assert network[0].weight.shape == tuple(widths[:-1][::-1])
    assert network[2].weight.shape == tuple(widths[1:][::-1])
