from glob import glob

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from zzsn.constants import *

NAME = "zzsn"


class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        data_dir: str,
        transform: callable = None,
        target_transform: callable = None,
    ):
        classes: pd.DataFrame = pd.read_csv(annotations_file, names=["class"])
        y_classes: list = []
        y_files: list = []

        for i, lab in classes.iterrows():
            alphabet, character, _ = lab["class"].split("/")
            img_dir: str = os.path.join(data_dir, alphabet, character)
            files: list[str] = sorted(glob(os.path.join(img_dir, "*.png")))
            y_files.extend(files)
            y_classes.extend([lab["class"]] * len(files))

        self.img_labels: pd.DataFrame = pd.DataFrame(
            {"file": y_files, "class": y_classes}
        )
        self.data_dir: str = data_dir
        self.transform: callable = transform
        self.target_transform: callable = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        image = read_image(self.img_labels.iloc[idx, 0])
        label: str = self.img_labels.iloc[idx, 1]
        if self.transform:
            _, _, rotation = label.split("/")
            image = self.transform(image, rotation)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def read_image(path: str):
    return Image.open(path)


def transform_image(img, rot: str):
    return img.rotate(float(rot[3:])).resize((IMG_HEIGHT, IMG_WIDTH))


def create_dataset(split: str):
    print("Loading {} dataset...".format(split))
    ds: CustomImageDataset = CustomImageDataset(
        annotations_file=os.path.join(OMNIGLOT_SPLITS_DIR, split + ".txt"),
        data_dir=OMNIGLOT_DATA_DIR,
        transform=transform_image,
    )
    print("   Done")
    return ds


def create_data_loader(ds: CustomImageDataset, split: str):
    print("Creating {} data loader...".format(split))
    dl: DataLoader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    print("   Done")
    return dl
