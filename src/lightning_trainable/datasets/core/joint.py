
from torch.utils.data import Dataset, IterableDataset


class JointDataset(Dataset):
    """ Dataset that basically zips two or more datasets """
    def __init__(self, *datasets: Dataset):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, item):
        return [ds[item] for ds in self.datasets]


class JointIterableDataset(IterableDataset):
    """ Iterable Dataset version of JointDataset """
    def __init__(self, *datasets: Dataset):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        return zip(*self.datasets)
