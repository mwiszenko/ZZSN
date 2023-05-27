import glob

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from zzsn.constants import *

NAME = "zzsn"


class CustomImageDataset(Dataset):
    def __init__(
        self, annotations_file, data_dir, transform=None, target_transform=None
    ):
        classes = pd.read_csv(annotations_file, names=["class"])
        y_classes = []
        y_files = []

        for i, lab in classes.iterrows():
            alphabet, character, _ = lab["class"].split("/")
            img_dir = os.path.join(data_dir, alphabet, character)
            files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            y_files.extend(files)
            y_classes.extend([lab["class"]] * len(files))

        self.img_labels = pd.DataFrame({"file": y_files, "class": y_classes})
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            _, _, rotation = label.split("/")
            image = self.transform(image, rotation)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def read_image(path: str):
    return Image.open(path)


def transform_image(img, rot):
    rotated_img = img.rotate(float(rot[3:]))
    scaled_img = rotated_img.resize((IMG_HEIGHT, IMG_WIDTH))
    return scaled_img


def create_dataset(split: str):
    print("Loading {} dataset...".format(split))
    ds = CustomImageDataset(
        annotations_file=os.path.join(OMNIGLOT_SPLITS_DIR, split + ".txt"),
        data_dir=OMNIGLOT_DATA_DIR,
        transform=transform_image,
    )
    print("   Done")
    return ds
