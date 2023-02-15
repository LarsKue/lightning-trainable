
from torch.utils.data import Dataset, IterableDataset


class JointDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, item):
        return [ds[item] for ds in self.datasets]


class JointIterableDataset(IterableDataset):
    def __init__(self, *datasets: Dataset):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        return zip([iter(ds) for ds in self.datasets])
