from typing import Callable

import torch
from numpy import mean
from torch import Tensor, nn
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from zzsn.constants import KERNEL, PADDING, POOLING


class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class ProtoNetwork(nn.Module):
    def __init__(
        self,
        x_dim: tuple,
        hid_dim: int,
        z_dim: int,
        dist: Callable[[Tensor, Tensor], Tensor],
    ) -> None:
        super(ProtoNetwork, self).__init__()
        self.encoder = nn.Sequential(
            get_conv_block(x_dim[0], hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, hid_dim),
            get_conv_block(hid_dim, z_dim),
            Flatten(),
        )
        self.distance: Callable = dist

    def loss(
        self, sample: dict[str, Tensor]
    ) -> tuple[Tensor, dict[str, float]]:
        xs: Tensor = sample["xs"]  # support
        xq: Tensor = sample["xq"]  # query

        if torch.cuda.is_available:
            xs = xs.cuda()
            xq = xq.cuda()

        n_class: int = xs.size(0)
        assert xq.size(0) == n_class
        n_support: int = xs.size(1)
        n_query: int = xq.size(1)

        targets: Tensor = (
            torch.arange(0, n_class)
            .view(n_class, 1, 1)
            .expand(n_class, n_query, 1)
            .long()
        )

        if xq.is_cuda:
            targets = targets.cuda()

        x: Tensor = torch.cat(
            [
                xs.view(n_class * n_support, *xs.size()[2:]),
                xq.view(n_class * n_query, *xq.size()[2:]),
            ],
            0,
        )

        z: Tensor = self.encoder.forward(x)
        z_dim: int = z.size(-1)

        z_proto: Tensor = (
            z[: n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        )
        zq: Tensor = z[n_class * n_support :]

        dists: Tensor = self.distance(zq, z_proto)

        log_p_y: Tensor = log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val: Tensor = (
            -log_p_y.gather(2, targets).squeeze().view(-1).mean()
        )

        y_hat: Tensor
        _, y_hat = log_p_y.max(2)
        acc_val: Tensor = torch.eq(y_hat, targets.squeeze()).float().mean()

        return loss_val, {"loss": loss_val.item(), "acc": acc_val.item()}


def get_conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=KERNEL, padding=PADDING
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(POOLING),
    )


def train(model: ProtoNetwork, data_loader: DataLoader, optim, sched) -> tuple:
    # set mode
    model = model.train()

    losses: list[float] = []
    accuracies: list[float] = []

    loop = tqdm(data_loader)
    idx: int
    sample: dict[str, Tensor]

    for idx, sample in enumerate(loop):
        optim.zero_grad()

        loss: Tensor
        metrics: dict[str, float]
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
) -> tuple:
    # set mode
    model = model.eval()

    losses: list[float] = []
    accuracies: list[float] = []

    with torch.no_grad():
        loop = tqdm(data_loader)
        idx: int
        sample: dict[str, Tensor]

        for idx, sample in enumerate(loop):
            metrics: dict[str, float]
            _, metrics = model.loss(sample)

            losses.append(metrics["loss"])
            accuracies.append(metrics["acc"])

    return mean(accuracies), mean(losses)
