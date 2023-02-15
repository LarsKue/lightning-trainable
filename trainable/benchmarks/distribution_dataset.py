
import torch.distributions as D
from torch.utils.data import IterableDataset


class DistributionDataset(IterableDataset):
    """
    Infinite dataset based on a torch.distributions.Distribution.
    Each access contains newly sampled values.
    Values are sampled in batches according to `batch_size`.
    Sampling can be parallelized with multiple workers from a DataLoader.
    """
    def __init__(self, distribution: D.Distribution, batch_size: int = 32):
        self.distribution = distribution
        self.batch_size = batch_size
        self.iterator = None
        self.resample()

    def resample(self):
        # resample the data in the current iterator
        self.iterator = iter(self.distribution.sample((self.batch_size,)))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.resample()
            return next(self.iterator)
