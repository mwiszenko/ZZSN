import os

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OMNIGLOT_DATA_DIR = os.path.join(DATA_DIR, "omniglot", "data")
OMNIGLOT_SPLITS_DIR = os.path.join(DATA_DIR, "omniglot", "splits")
IMG_HEIGHT = 28
IMG_WIDTH = 28
BATCH_SIZE = 64
RANDOM_SEED = 42
EPOCHS = 5
