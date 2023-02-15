
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
        worker_info = get_worker_info()
        if self.max_samples is None or worker_info is None:
            # single-process data loading, return full iterator
            return DistributionSampler(self.distribution, self.max_samples)
        else:
            # in a worker process, split workload
            return DistributionSampler(self.distribution, self.max_samples // worker_info.num_workers)


class DistributionSampler:
    def __init__(self, distribution, max_samples: int = None):
        self.distribution = distribution
        self.max_samples = max_samples
        self.samples_taken = 0

    def __next__(self):
        if self.max_samples is not None and self.samples_taken >= self.max_samples:
            raise StopIteration
        return self.distribution.sample(())
