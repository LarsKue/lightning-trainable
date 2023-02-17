
from .image import BigImageDataset


class AFHQDataset(BigImageDataset):
    paper_url = "https://github.com/clovaai/stargan-v2"

    # original dropbox url from the paper
    url = "https://www.dropbox.com/s/vkzjokiwof5h8w6/afhq_v2.zip?dl=0"
    filename = "afhq_v2.zip"
