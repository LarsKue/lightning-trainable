
import torch.distributions as D
from torch.utils.data import IterableDataset, get_worker_info


class DistributionDataset(IterableDataset):
    """
    Possibly infinite dataset based on a torch.distributions.Distribution.
    Each access contains newly sampled values.
    Values are sampled in batches according to `batch_size`.
    Sampling can be parallelized with multiple workers from a DataLoader.
    """
    def __init__(self, distribution: D.Distribution, max_samples: int = None):
        self.distribution = distribution
        self.max_samples = max_samples

    def __iter__(self):
        if self.max_samples is None:
            return DistributionSampler(self.distribution)

        return DistributionSampler(self.distribution, len(self))

    def __len__(self):
        if self.max_samples is None:
            raise NotImplementedError(f"Infinite {self.__class__.__name__} has no length.")

        worker_info = get_worker_info()
        if worker_info is None:
            # single-process data loading
            return self.max_samples
        else:
            return self.max_samples // worker_info.num_workers


class DistributionSampler:
    def __init__(self, distribution, max_samples: int = None):
        self.distribution = distribution
        self.max_samples = max_samples
        self.samples_taken = 0

    def __next__(self):
        if self.max_samples is not None and self.samples_taken >= self.max_samples:
            raise StopIteration
        self.samples_taken += 1
        return self.distribution.sample(())
