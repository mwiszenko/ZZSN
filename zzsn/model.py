import torch
from numpy import mean
from torch import nn
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from zzsn.constants import *


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)


class ProtoNetwork(nn.Module):
    def __init__(self, x_dim: tuple, hid_dim: int, z_dim: int, dist: callable):
        super(ProtoNetwork, self).__init__()
        self.encoder = nn.Sequential(
            get_conv_block(x_dim[0], hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, z_dim),
            Flatten(),
        )
        self.distance = dist

    def loss(self, sample):
        xs: torch.Tensor = sample["xs"]  # support
        xq: torch.Tensor = sample["xq"]  # query

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)

        targets: torch.Tensor = (
            torch.arange(0, n_class)
            .view(n_class, 1, 1)
            .expand(n_class, n_query, 1)
            .long()
        )

        if xq.is_cuda:
            targets = targets.cuda()

        x = torch.cat(
            [
                xs.view(n_class * n_support, *xs.size()[2:]),
                xq.view(n_class * n_query, *xq.size()[2:]),
            ],
            0,
        )

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = (
            z[: n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        )
        zq = z[n_class * n_support :]

        dists = self.distance(zq, z_proto)

        log_p_y = log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, targets).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, targets.squeeze()).float().mean()

        return loss_val, {"loss": loss_val.item(), "acc": acc_val.item()}


def get_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=KERNEL, padding=PADDING),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(POOLING),
    )


def train(model: ProtoNetwork, data_loader: DataLoader, optim, sched):
    # set mode
    model = model.train()

    losses = []
    accuracies = []

    loop = tqdm(data_loader)
    for idx, sample in enumerate(loop):
        optim.zero_grad()

        # get model outputs
        loss, metrics = model.loss(sample)

        losses.append(metrics["loss"])
        accuracies.append(metrics["acc"])
        loss.backward()

        optim.step()
    sched.step()

    return mean(accuracies), mean(losses)


def evaluate(
    model: ProtoNetwork,
    data_loader: DataLoader,
):
    # set mode
    model = model.eval()

    losses = []
    accuracies = []

    with torch.no_grad():
        loop = tqdm(data_loader)
        for idx, sample in enumerate(loop):
            _, metrics = model.loss(sample)

            losses.append(metrics["loss"])
            accuracies.append(metrics["acc"])

    return mean(accuracies), mean(losses)
