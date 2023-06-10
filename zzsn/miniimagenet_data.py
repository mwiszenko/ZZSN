import os
import pickle
import random
from glob import glob
from typing import Iterator, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from zzsn.constants import (
    MINIIMAGENET_DIR,
    MINIIMAGENET_IMG_SHAPE,
    MINIIMAGENET_SAMPLES_PER_CLASS,
    MINIIMAGENET_SPLITS_DIR,
    MINIIMAGENET_TESTCL,
    MINIIMAGENET_TRAINCL,
    MINIIMAGENET_VALIDCL,
)


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


class MiniImageNetDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_dir: str,
        n_support: int,
        n_query: int,
    ) -> None:
        split: pd.DataFrame = pd.read_csv(annotations_file)
        self.annotations_file = annotations_file
        self.classes = np.unique(split["label"].to_numpy(dtype=str))
        self.n_support: int = n_support
        self.n_query: int = n_query
        self.data_dir: str = data_dir

    def __len__(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int) -> dict[str, Union[str, Tensor]]:
        cl: str = self.classes[idx]

        shape: list
        if "train" in self.annotations_file:
            data_file: str = "mini-imagenet-cache-train.pkl"
            shape = [
                MINIIMAGENET_TRAINCL,
                MINIIMAGENET_SAMPLES_PER_CLASS,
                *MINIIMAGENET_IMG_SHAPE,
            ]
        elif "test" in self.annotations_file:
            shape = [
                MINIIMAGENET_TESTCL,
                MINIIMAGENET_SAMPLES_PER_CLASS,
                *MINIIMAGENET_IMG_SHAPE,
            ]
            data_file: str = "mini-imagenet-cache-test.pkl"
        elif "val" in self.annotations_file:
            data_file: str = "mini-imagenet-cache-val.pkl"
            shape = [
                MINIIMAGENET_VALIDCL,
                MINIIMAGENET_SAMPLES_PER_CLASS,
                *MINIIMAGENET_IMG_SHAPE,
            ]

        with open(os.path.join(self.data_dir, data_file), "rb") as fd:
            data = pickle.load(fd)

        images: np.ndarray = data["image_data"]
        images = images.reshape(shape)[idx]

        d: dict[str, Union[str, Tensor]] = extract_episode(
            self.n_support, self.n_query, cl, images
        )
        return d

    def _initialize_classes(
        self,
        split: pd.DataFrame,
    ) -> np.ndarray:
        return np.unique(split["label"].to_numpy(dtype=str))


def extract_episode(
    n_support: int, n_query: int, cl: str, images: list[np.array]
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


def convert_to_tensor(x: np.array) -> Tensor:
    xt: Tensor = (
        1
        - torch.from_numpy(np.array(x / 255, np.float32, copy=False))
        .permute(2, 0, 1)
        .contiguous()
    )

    return xt


def create_dataset(
    split: str, n_support: int, n_query: int
) -> MiniImageNetDataset:
    ds: MiniImageNetDataset = MiniImageNetDataset(
        annotations_file=os.path.join(MINIIMAGENET_SPLITS_DIR, split + ".csv"),
        data_dir=MINIIMAGENET_DIR,
        n_support=n_support,
        n_query=n_query,
    )
    return ds


def create_data_loader(
    split: str,
    n_support: int,
    n_query: int,
    n_way: int,
    n_episodes: int,
) -> DataLoader:
    ds: MiniImageNetDataset = create_dataset(
        split=split, n_support=n_support, n_query=n_query
    )
    sampler: BatchSampler = BatchSampler(
        n_classes=len(ds), n_way=n_way, n_episodes=n_episodes
    )
    dl = DataLoader(dataset=ds, batch_sampler=sampler, num_workers=0)

    return dl
