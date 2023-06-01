import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from zzsn.constants import *

NAME = "zzsn"


class BatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_dir: str,
        n_support: int,
        n_query: int,
        transform: callable = None,
        target_transform: callable = None,
    ):
        split: pd.DataFrame = pd.read_csv(annotations_file, names=["class"])
        self.classes = split["class"].to_numpy()
        self.n_support = n_support
        self.n_query = n_query
        self.data_dir: str = data_dir
        self.transform: callable = transform
        self.target_transform: callable = target_transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx: int):
        cl: str = self.classes[idx]
        files: list = expand(cl, self.data_dir)
        images: list = [read_image(i) for i in files]

        if self.transform:
            _, _, rotation = cl.split("/")
            images = [self.transform(i, rotation) for i in images]
        d = extract_episode(self.n_support, self.n_query, cl, images)
        return d


def expand(label: str, data_dir: str):
    alphabet, character, _ = label.split("/")
    img_dir: str = os.path.join(data_dir, alphabet, character)
    files: list[str] = sorted(glob(os.path.join(img_dir, "*.png")))
    return files


def extract_episode(n_support, n_query, cl, images):
    n_examples = len(images)
    images = [convert_tensor(i) for i in images]

    example_inds = random.sample(range(n_examples), n_support + n_query)

    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = [images[i] for i in support_inds]
    xq = [images[i] for i in query_inds]

    return {
        "class": cl,
        "xs": torch.stack(xs, dim=0),
        "xq": torch.stack(xq, dim=0),
    }


def convert_tensor(x):
    x = 1.0 - torch.from_numpy(np.array(x, np.float32, copy=False)).transpose(
        0, 1
    ).contiguous().view(1, x.size[0], x.size[1])
    return x


def read_image(path: str):
    return Image.open(path)


def transform_image(img, rot: str):
    return img.rotate(float(rot[3:])).resize((IMG_HEIGHT, IMG_WIDTH))


def create_dataset(split: str, n_support: int, n_query: int):
    print("Loading {} dataset...".format(split))
    ds: CustomImageDataset = CustomImageDataset(
        annotations_file=os.path.join(OMNIGLOT_SPLITS_DIR, split + ".txt"),
        n_support=n_support,
        n_query=n_query,
        data_dir=OMNIGLOT_DATA_DIR,
        transform=transform_image,
    )
    print("   Done")
    return ds


def create_data_loader(
    ds: CustomImageDataset, split: str, sampler: BatchSampler
):
    print("Creating {} data loader...".format(split))
    dl: DataLoader = DataLoader(ds, batch_sampler=sampler, num_workers=0)
    print("   Done")
    return dl


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
