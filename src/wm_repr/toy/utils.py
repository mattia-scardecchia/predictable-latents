import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Sequence, Optional
from collections import defaultdict


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_mlp(
    sizes: Sequence[int], activation: str = "relu", out_act: str = None
) -> nn.Sequential:
    acts = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(acts[activation]())
    if out_act is not None:
        layers.append(acts[out_act]())
    return nn.Sequential(*layers)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)

    def update(self, vals: Dict[str, float], weights: Optional[Dict[str, int]] = None):
        """
        vals: dictionary mapping metric names to their average values over a batch
        weights: dictionary mapping metric names to the number of samples they were averaged over. If None,
        each metric is assumed to be averaged over 1 sample.
        Updates the running sums and counts for each metric. The purpose of weights is to average correctly when
        batches have different sizes.
        """
        if weights is None:
            weights = {k: 1 for k in vals}
        for k in vals:
            self.sums[k] += vals[k] * weights.get(k, 1)
            self.counts[k] += weights.get(k, 1)

    @property
    def avg(self) -> Dict[str, float]:
        return {k: self.sums[k] / max(1, self.counts[k]) for k in self.sums}
