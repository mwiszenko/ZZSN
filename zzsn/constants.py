import os

RANDOM_SEED = 42

# data
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "omniglot", "data")
SPLITS_DIR = os.path.join(DATA_DIR, "omniglot", "splits")

# model
X_DIM = (1, 28, 28)
HID_DIM = 64
Z_DIM = 64
KERNEL = (3, 3)
PADDING = 1
POOLING = (2, 2)
