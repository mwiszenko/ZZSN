import torch
import torchmetrics.functional as tm

COSINE_NORMALIZATION = 10**2


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n: int = x.size(0)
    m: int = y.size(0)
    d: int = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (1 - tm.pairwise_cosine_similarity(x, y)) * COSINE_NORMALIZATION
