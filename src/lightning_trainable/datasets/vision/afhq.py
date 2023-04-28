
import os
import requests

import torch
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import extract_archive

from tqdm import tqdm


class AFHQDataset(ImageFolder):
    url = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=1"
    dirname = "afhq"
    filename = "afhq.zip"
    md5 = "c8e9e9fcf2e3b6c9fbd8e3d9e7dadb0e"
    chunk_size = 1024 * 1024
    _remove_finished = True

    default_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    default_target_transform = transforms.Compose([
        transforms.Lambda(lambda label: F.one_hot(torch.tensor(label), num_classes=3))
    ])

    def __init__(self, root: str, split: str = "train", download: bool = True, **kwargs):
        self.root = root
        if download:
            self.download()

        kwargs.setdefault("transform", self.default_transform)
        kwargs.setdefault("target_transform", self.default_target_transform)

        super().__init__(os.path.join(root, self.dirname, split), **kwargs)

    def download(self):
        archive = os.path.join(self.root, self.filename)
        target = os.path.join(self.root, self.dirname)

        if os.path.isdir(target):
            print(f"Found existing dataset, skipping download.")
            return

        print(f"Downloading {self.__class__.__name__} to {archive}.")

        # download the archive
        response = requests.get(self.url, stream=True)

        # determine total size of download
        total_size = int(response.headers.get("Content-Length", 0))

        with open(archive, "wb") as f:
            it = response.iter_content(chunk_size=self.chunk_size)
            it = tqdm(it, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024)
            for chunk in it:
                if chunk:
                    f.write(chunk)
                    it.update(len(chunk))

        print(f"Extracting {archive} to {target}.")
        extract_archive(archive, target, remove_finished=self._remove_finished)
