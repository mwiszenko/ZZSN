import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import os

from zzsn.constants import HID_DIM, RANDOM_SEED, X_DIM, Z_DIM, OMNIGLOT, MINIIMAGENET, MODELS_PATH

import zzsn.omniglot_data as omniglot
import zzsn.miniimagenet_data as mimagenet
from zzsn.model import ProtoNetwork, evaluate, train, get_model_name
from zzsn.utils import euclidean_dist

DISTANCE_FUNC_MAPPER = {"euclidean": euclidean_dist}

def run_test(
    model: str,
    dataset: str,
    n_way: int,
    n_support: int,
    n_query: int,
    n_eval_episodes: int,
    distance_func: str,    
) -> None:
    pass

    checkpoint = torch.load(os.path.join(MODELS_PATH, model + ".bin"), map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model: ProtoNetwork = ProtoNetwork(
        x_dim=X_DIM[dataset],
        hid_dim=HID_DIM[dataset],
        z_dim=Z_DIM[dataset],
        dist=DISTANCE_FUNC_MAPPER.get(distance_func, euclidean_dist),
    )
    custom_model.load_state_dict(checkpoint)
    custom_model.to(device)
    

    if dataset == MINIIMAGENET:
        create_data_loader = mimagenet.create_data_loader
    elif dataset == OMNIGLOT:
        create_data_loader = omniglot.create_data_loader
    else:
        exit(-1)

    dl_test: DataLoader = create_data_loader(
        split="test",
        n_support=n_support,
        n_query=n_query,
        n_way=n_way,
        n_episodes=n_eval_episodes,
    )

    print("Running test...")

    test_acc: float
    test_acc, _ = evaluate(model=custom_model, data_loader=dl_test)

    print("  Test accuracy: {0:.2f}".format(test_acc))