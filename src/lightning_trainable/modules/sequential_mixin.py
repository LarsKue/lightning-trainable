
import torch.nn as nn


class SequentialMixin:
    network: nn.Sequential

    def __getitem__(self, item):
        return self.network[item]

    def __len__(self):
        return len(self.network)

    def __iter__(self):
        return iter(self.network)
