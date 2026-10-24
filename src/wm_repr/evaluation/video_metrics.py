# The I3D wrapper, the Fréchet distance metrics and the I3D feature extraction are adapted from the
# Diffusion Forcing Transformer (https://github.com/kwsong0113/diffusion-forcing-transformer),
# Copyright (c) 2025 Kiwhan Song, Boyuan Chen, MIT License.
"""
FID and FVD for the world-model evaluation.

These are distributional metrics: they accumulate statistics over every evaluated
clip and produce one score per evaluation run.

Feature definitions follow the released Diffusion Forcing / DFoT metrics.
"""

import logging
import os
from pathlib import Path
from abc import ABC, abstractmethod

import einops
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm as scipy_sqrtm
from torch import Tensor
from torchmetrics import Metric

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I3D model
# ---------------------------------------------------------------------------


def _load_i3d_model() -> torch.jit.ScriptModule:
    """Download and load the pretrained I3D torchscript model."""
    from huggingface_hub import hf_hub_download

    path = os.environ.get("WM_I3D_PATH")
    if not path or not Path(path).is_file():
        path = hf_hub_download(
            repo_id="kiwhansong/DFoT",
            filename="metrics_models/i3d_torchscript.pt",
            revision="0959defb4c4fe010f84791d732cc978ad7d49fef",  # as in configs/assets.json
        )
    model = torch.jit.load(path)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class I3D(nn.Module):
    """Wrapper around pretrained I3D for video feature extraction."""

    def __init__(self):
        super().__init__()
        self.model = _load_i3d_model()

    def train(self, mode: bool = True) -> "I3D":
        return super().train(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


# ---------------------------------------------------------------------------
# Fréchet distance
# ---------------------------------------------------------------------------


def _compute_frechet_distance(
    mu_fake: Tensor, sigma_fake: Tensor, mu_real: Tensor, sigma_real: Tensor
) -> Tensor:
    """Fréchet distance between two multivariate Gaussians."""
    diff = mu_real - mu_fake
    covmean = scipy_sqrtm((sigma_real @ sigma_fake).cpu().double().numpy())
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    covmean = torch.from_numpy(covmean.copy()).to(sigma_real)
    return (diff @ diff + torch.trace(sigma_real + sigma_fake - 2 * covmean)).clamp(
        min=0
    )


class FrechetDistance(Metric, ABC):
    """Base class for Fréchet distance metrics (FID, FVD).

    Accumulates running sums of features and their outer products across
    batches, then computes mean/covariance and the Fréchet distance at
    ``compute()`` time.
    """

    higher_is_better = False
    is_differentiable = False
    full_state_update = False

    def __init__(self, num_features: int, **kwargs):
        super().__init__(**kwargs)
        d = num_features
        self.add_state("real_sum", torch.zeros(d).double(), dist_reduce_fx="sum")
        self.add_state(
            "real_cov_sum", torch.zeros(d, d).double(), dist_reduce_fx="sum"
        )
        self.add_state("real_count", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.add_state("fake_sum", torch.zeros(d).double(), dist_reduce_fx="sum")
        self.add_state(
            "fake_cov_sum", torch.zeros(d, d).double(), dist_reduce_fx="sum"
        )
        self.add_state("fake_count", torch.tensor(0).long(), dist_reduce_fx="sum")

    @abstractmethod
    def extract_features(self, x: Tensor) -> Tensor: ...

    def _accumulate(self, x: Tensor, real: bool):
        feats = self.extract_features(x).double()
        if feats.dim() == 1:
            feats = feats.unsqueeze(0)
        if real:
            self.real_sum += feats.sum(0)
            self.real_cov_sum += feats.t().mm(feats)
            self.real_count += feats.size(0)
        else:
            self.fake_sum += feats.sum(0)
            self.fake_cov_sum += feats.t().mm(feats)
            self.fake_count += feats.size(0)

    def update(self, fake: Tensor, real: Tensor):
        self._accumulate(fake, real=False)
        self._accumulate(real, real=True)

    def compute(self) -> Tensor:
        if self.real_count < 2 or self.fake_count < 2:
            # Unbiased covariance requires at least two samples.
            logging.warning(
                "Frechet distance needs >=2 samples per side "
                "(real=%d, fake=%d); returning NaN.",
                int(self.real_count), int(self.fake_count),
            )
            return torch.tensor(float("nan"))
        mu_r = self.real_sum / self.real_count
        mu_f = self.fake_sum / self.fake_count
        cov_r = (self.real_cov_sum - self.real_count * mu_r.outer(mu_r)) / (
            self.real_count - 1
        )
        cov_f = (self.fake_cov_sum - self.fake_count * mu_f.outer(mu_f)) / (
            self.fake_count - 1
        )
        return _compute_frechet_distance(mu_f, cov_f, mu_r, cov_r)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------


class FID(FrechetDistance):
    """Frame-level Fréchet Inception Distance (InceptionV3, 2048-d features)."""

    def __init__(self, inception: nn.Module | None = None, **kwargs):
        super().__init__(num_features=2048, **kwargs)
        if inception is None:
            from torchmetrics.image.fid import NoTrainInceptionV3

            inception = NoTrainInceptionV3(
                name="inception-v3-compat", features_list=["2048"]
            )
            inception.eval()
        self.inception = inception

    def extract_features(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W) in [0, 1]; clamped because the uint8 cast wraps out-of-range values.
        x = (x.clamp(0.0, 1.0) * 255).to(torch.uint8)
        return self.inception(x)  # (N, 2048)


# ---------------------------------------------------------------------------
# FVD
# ---------------------------------------------------------------------------


class FVD(FrechetDistance):
    """Video-level Fréchet Video Distance.  Expects pre-extracted I3D features."""

    def __init__(self, **kwargs):
        super().__init__(num_features=400, **kwargs)

    def extract_features(self, x: Tensor) -> Tensor:
        return x  # already (B, 400)


# ---------------------------------------------------------------------------
# Helper: extract I3D features from a batch of videos
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_i3d_features(i3d: I3D, x: Tensor) -> Tensor:
    """
    Args:
        i3d: Pretrained I3D model.
        x: (B, T, C, H, W) in [0, 1].

    Returns:
        (B, 400) feature tensor.
    """
    # I3D requires at least 9 frames — pad by repeating boundary frames
    if x.shape[1] < 9:
        pad = (10 - x.shape[1]) // 2
        x = torch.cat(
            [
                einops.repeat(x[:, 0:1], "b 1 c h w -> b t c h w", t=pad).clone(),
                x,
                einops.repeat(x[:, -1:], "b 1 c h w -> b t c h w", t=pad),
            ],
            dim=1,
        )
    x = 2.0 * x - 1.0  # [0, 1] → [-1, 1]
    x = einops.rearrange(
        torch.clamp(x, -1.0, 1.0), "b t c h w -> b c t h w"
    ).contiguous()
    return i3d(x, rescale=False, resize=True, return_features=True)
