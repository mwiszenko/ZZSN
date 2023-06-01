import numpy as np
import torch
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

from zzsn.constants import *
from zzsn.data import create_data_loader
from zzsn.model import ProtoNetwork, evaluate, train
from zzsn.utils import euclidean_dist

DISTANCE_FUNC_MAPPER = {"euclidean": euclidean_dist}


def run_train(
    epochs: int,
    n_support: int,
    n_query: int,
    n_way: int,
    n_train_episodes: int,
    n_eval_episodes: int,
    learning_rate: float,
    distance_func: str,
):
    dl_train: DataLoader = create_data_loader(
        split="train",
        n_support=n_support,
        n_query=n_query,
        n_way=n_way,
        n_episodes=n_train_episodes,
    )
    dl_val: DataLoader = create_data_loader(
        split="val",
        n_support=n_support,
        n_query=n_query,
        n_way=n_way,
        n_episodes=n_eval_episodes,
    )
    dl_test: DataLoader = create_data_loader(
        split="test",
        n_support=n_support,
        n_query=n_query,
        n_way=n_way,
        n_episodes=n_eval_episodes,
    )

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model: ProtoNetwork = ProtoNetwork(
        x_dim=X_DIM,
        hid_dim=HID_DIM,
        z_dim=Z_DIM,
        dist=DISTANCE_FUNC_MAPPER.get(distance_func),
    ).to(device)

    optimizer = AdamW(custom_model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.5)

    best_acc: float = 0

    for epoch_i in range(epochs):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))

        print("Training...")

        train_acc, train_loss = train(
            model=custom_model,
            data_loader=dl_train,
            optim=optimizer,
            sched=scheduler,
        )

        print("  Train accuracy: {0:.2f}".format(train_acc))
        print("  Train loss: {0:.2f}".format(train_loss))

        print("Running validation...")

        val_acc, val_loss = evaluate(model=custom_model, data_loader=dl_val)

        print("  Validation accuracy: {0:.2f}".format(val_acc))
        print("  Validation loss: {0:.2f}".format(val_loss))

        # save model state with best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(custom_model.state_dict(), "best_model.bin")

    # check model accuracy on test data
    print("Running test...")

    test_acc, _ = evaluate(model=custom_model, data_loader=dl_test)

    print("  Test accuracy: {0:.2f}".format(test_acc))
