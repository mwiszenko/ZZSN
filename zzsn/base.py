import glob
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

NAME = "zzsn"
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OMNIGLOT_DATA_DIR = os.path.join(DATA_DIR, "omniglot", "data")
OMNIGLOT_SPLITS_DIR = os.path.join(DATA_DIR, "omniglot", "splits")
IMG_HEIGHT = 28
IMG_WIDTH = 28
SPLIT = "train"


def get_images(description, data_dir, height, width):
    images = []
    alphabet, character, rot = description.split("/")
    img_dir = os.path.join(data_dir, alphabet, character)
    class_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    for i in class_images:
        img = Image.open(i)
        rotated_img = img.rotate(float(rot[3:]))
        scaled_img = rotated_img.resize((height, width))
        images.append(scaled_img)
    return images


def load(split: str):
    images = []
    splits_file_path: str = os.path.join(OMNIGLOT_SPLITS_DIR, split + ".txt")
    df: pd.DataFrame = pd.read_csv(splits_file_path, names=["Path"])
    tqdm.pandas()
    df.progress_apply(
        lambda row: images.extend(
            get_images(row["Path"], OMNIGLOT_DATA_DIR, IMG_HEIGHT, IMG_WIDTH)
        ),
        axis=1,
    )
    return df


print("Loading {} dataset...".format(SPLIT))
df_og: pd.DataFrame = load(SPLIT)
print(len(df_og))
print("   Done")
