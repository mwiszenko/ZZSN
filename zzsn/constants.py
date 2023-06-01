import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OMNIGLOT_DATA_DIR = os.path.join(DATA_DIR, "omniglot", "data")
OMNIGLOT_SPLITS_DIR = os.path.join(DATA_DIR, "omniglot", "splits")
RANDOM_SEED = 42
X_DIM = (1, 28, 28)
HID_DIM = 64
Z_DIM = 64
