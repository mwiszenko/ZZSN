import os

RANDOM_SEED = 42

# data
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "omniglot", "data")
SPLITS_DIR = os.path.join(DATA_DIR, "omniglot", "splits")
OMNIGLOT_SCRIPT_PATH = os.path.join(ROOT_DIR, "download_omniglot.sh")
MINIIMAGENET_SCRIPT_PATH = os.path.join(ROOT_DIR, "download_miniimagenet.sh")
OMNIGLOT = "OMNIGLOT"
MINIIMAGENET = "MINIIMAGENET"
DATASETS = [OMNIGLOT, MINIIMAGENET]

# miniimagenet metadata
MINIIMAGENET_DIR = os.path.join(DATA_DIR, "miniImageNet", "data")
MINIIMAGENET_SPLITS_DIR = os.path.join(DATA_DIR, "miniImageNet", "splits")
MINIIMAGENET_CLASSES = 100
MINIIMAGENET_TRAINCL = 64
MINIIMAGENET_VALIDCL = 16
MINIIMAGENET_TESTCL = 20
MINIIMAGENET_SAMPLES_PER_CLASS = 600
MINIIMAGENET_IMG_SHAPE = [84, 84, 3]

# model
MODELS_PATH = "models"
X_DIM: dict = {}
HID_DIM: dict = {}
Z_DIM: dict = {}

# miniimagenet
X_DIM[MINIIMAGENET] = (3, 84, 84)
HID_DIM[MINIIMAGENET] = 64
Z_DIM[MINIIMAGENET] = 1600
KERNEL = (3, 3)
PADDING = 1
POOLING = (2, 2)
DISTANCE_FUNCTIONS = ["euclidean", "cosine"]

# omniglot
X_DIM[OMNIGLOT] = (1, 28, 28)
HID_DIM[OMNIGLOT] = 64
Z_DIM[OMNIGLOT] = 64

# train mode
DEFAULT_EPOCHS = 5
DEFAULT_N_WAY = 5
DEFAULT_N_SUPPORT = 5
DEFAULT_N_QUERY = 5
DEFAULT_N_TRAIN_EPISODES = 100
DEFAULT_N_EVAL_EPISODES = 10
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DISTANCE_FUNC = DISTANCE_FUNCTIONS[0]
DEFAULT_DATASET = OMNIGLOT

# test mode
TEST_ITERATIONS = 1
DEFAULT_TEST_RESULT_FILE = "testing_results.txt"
