import torch
import pathlib
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


class BigImageDataset(VisionDataset):
    """
    File-By-File Dataset for big images
    Files are only loaded into memory once requested

    Currently assumes the following file structure
    - root
    -- split
    --- files

    The assumed file structure is subject to change.
    """

    url = None
    filename = None

    def __init__(self, root, split="train", download=False, **kwargs):
        super().__init__(root, **kwargs)
        if download:
            self.download()

        self.split = split
        split_root = pathlib.Path(self.root) / self.split

        # TODO: improve targets (more than just simple class, could be in metadata of file)
        self.filenames = []
        self.targets = []

        for i, cls in enumerate(split_root.glob("*")):
            directory = split_root / cls
            filenames = list(directory.glob("*"))
            targets = torch.full((len(filenames),), fill_value=i, dtype=torch.int64).tolist()
            self.filenames.extend(filenames)
            self.targets.extend(targets)

    def download(self) -> None:
        download_and_extract_archive(self.url, self.root, filename=self.filename)

    def __getitem__(self, item):
        img = Image.open(self.filenames[item])
        target = self.targets[item]

        return img, target

    def __len__(self):
        return len(self.filenames)

    def extra_repr(self) -> str:
        return f"Split: {self.split}"
