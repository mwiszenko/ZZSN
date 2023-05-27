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


def get_images(test, data_dir):
    images = []
    alphabet, character, rot = test.split("/")
    img_dir = os.path.join(data_dir, alphabet, character)
    class_images = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    for i in class_images:
        images.append(Image.open(i).rotate(float(rot[3:])))
    return images


def load(split: str):
    images = []
    splits_file_path: str = os.path.join(OMNIGLOT_SPLITS_DIR, split + ".txt")
    df: pd.DataFrame = pd.read_csv(splits_file_path, names=["Path"])
    tqdm.pandas()
    df.progress_apply(
        lambda row: images.extend(get_images(row["Path"], OMNIGLOT_DATA_DIR)),
        axis=1,
    )
    return df


print("Loading dataset...")
df_og: pd.DataFrame = load("train")
print(len(df_og))
print("   Done")
