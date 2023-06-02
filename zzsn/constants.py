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
DISTANCE_FUNCTIONS = ["euclidean"]

# training
DEFAULT_EPOCHS = 5
DEFAULT_N_WAY = 5
DEFAULT_N_SUPPORT = 5
DEFAULT_N_QUERY = 5
DEFAULT_N_TRAIN_EPISODES = 100
DEFAULT_N_EVAL_EPISODES = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISTANCE_FUNC = DISTANCE_FUNCTIONS[0]
