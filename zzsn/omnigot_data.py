import os
import random
from glob import glob
from typing import Callable, Iterator, Optional, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from zzsn.constants import DATASET_DIR, SPLITS_DIR, X_DIM, OMNIGOT


class BatchSampler(object):
    def __init__(self, n_classes: int, n_way: int, n_episodes: int) -> None:
        self.n_classes: int = n_classes
        self.n_way: int = n_way
        self.n_episodes: int = n_episodes

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self) -> Iterator:
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[: self.n_way]


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_dir: str,
        n_support: int,
        n_query: int,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        split: pd.DataFrame = pd.read_csv(annotations_file, names=["class"])
        self.classes = split["class"].to_numpy()
        self.n_support: int = n_support
        self.n_query: int = n_query
        self.data_dir: str = data_dir
        self.transform: Optional[Callable] = transform
        self.target_transform: Optional[Callable] = target_transform

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int) -> dict[str, Union[str, Tensor]]:
        cl: str = self.classes[idx]
        files: list = expand_class(cl, self.data_dir)
        images: list = [read_image(i) for i in files]

        if self.transform:
            rotation: str
            _, _, rotation = cl.split("/")
            images = [self.transform(i, rotation) for i in images]
        d: dict[str, Union[str, Tensor]] = extract_episode(
            self.n_support, self.n_query, cl, images
        )
        return d


def expand_class(label: str, data_dir: str) -> list[str]:
    alphabet: str
    character: str
    alphabet, character, _ = label.split("/")
    img_dir: str = os.path.join(data_dir, alphabet, character)
    files: list[str] = sorted(glob(os.path.join(img_dir, "*.png")))
    return files


def extract_episode(
    n_support: int, n_query: int, cl: str, images: list[Image]
) -> dict[str, Union[str, Tensor]]:
    n_examples: int = len(images)
    images_tensor: list[Tensor] = [convert_to_tensor(i) for i in images]

    example_inds: list[int] = random.sample(
        range(n_examples), n_support + n_query
    )

    support_inds: list[int] = example_inds[:n_support]
    query_inds: list[int] = example_inds[n_support:]

    xs: list[Tensor] = [images_tensor[i] for i in support_inds]
    xq: list[Tensor] = [images_tensor[i] for i in query_inds]

    return {
        "class": cl,
        "xs": torch.stack(xs, dim=0),
        "xq": torch.stack(xq, dim=0),
    }


def convert_to_tensor(x: Image) -> Tensor:
    xt: Tensor = 1.0 - torch.from_numpy(
        np.array(x, np.float32, copy=False)
    ).transpose(0, 1).contiguous().view(1, x.size[0], x.size[1])
    return xt


def read_image(path: str) -> Image:
    return Image.open(path)


def transform_image(img: Image, rot: str) -> Image:
    return img.rotate(float(rot[3:])).resize((X_DIM[OMNIGOT][1], X_DIM[OMNIGOT][2]))


def create_dataset(
    split: str, n_support: int, n_query: int, transform: Callable
) -> CustomImageDataset:
    ds: CustomImageDataset = CustomImageDataset(
        annotations_file=os.path.join(SPLITS_DIR, split + ".txt"),
        data_dir=DATASET_DIR,
        n_support=n_support,
        n_query=n_query,
        transform=transform,
    )
    return ds


def create_data_loader(
    split: str,
    n_support: int,
    n_query: int,
    n_way: int,
    n_episodes: int,
    transform=transform_image,
) -> DataLoader:
    ds: CustomImageDataset = create_dataset(
        split=split, n_support=n_support, n_query=n_query, transform=transform
    )
    sampler: BatchSampler = BatchSampler(
        n_classes=len(ds), n_way=n_way, n_episodes=n_episodes
    )
    return DataLoader(dataset=ds, batch_sampler=sampler, num_workers=0)
