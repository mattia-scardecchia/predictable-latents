"""
RAE-based encoder wrapper for JEPA representation learning.

Provides the interface of EncoderFromSDVAE, backed by the DINOv2 ViT encoder of a pretrained RAE,
which is fine-tuned together with the rest of the model.
"""

from math import sqrt

import torch
import torch.nn as nn

from wm_repr.jepa.autoencoder import BaseJEPAEncoder
from wm_repr.rae.rae import RAE


class EncoderFromRAE(BaseJEPAEncoder):
    """Wraps an RAE's encoder for use in JEPA representation learning.

    Extracts the ViT encoder and normalization components from a pretrained RAE
    and exposes them through the BaseJEPAEncoder interface.
    """

    def __init__(self, rae: RAE):
        super().__init__()
        # ViT encoder (e.g. DINOv2)
        self.encoder = rae.encoder
        self.encoder_input_size: int = rae.encoder_input_size
        self.encoder_patch_size: int = rae.encoder_patch_size

        # Enable gradient checkpointing for memory savings
        hf_backbone = getattr(self.encoder, "encoder", None)
        if hf_backbone is not None and hasattr(hf_backbone, "gradient_checkpointing_enable"):
            hf_backbone.gradient_checkpointing_enable()

        # ImageNet mean/std for input normalisation
        self.register_buffer("encoder_mean", rae.encoder_mean.clone())
        self.register_buffer("encoder_std", rae.encoder_std.clone())

        # Latent z-score normalisation stats (optional)
        self._do_normalization: bool = rae.do_normalization
        self._eps: float = getattr(rae, "eps", 1e-5)
        if self._do_normalization:
            self.register_buffer(
                "latent_mean",
                rae.latent_mean.clone() if rae.latent_mean is not None else None,
            )
            self.register_buffer(
                "latent_var",
                rae.latent_var.clone() if rae.latent_var is not None else None,
            )

        # Compute dimensions
        self._latent_dim: int = rae.latent_dim
        self._base_patches: int = rae.base_patches
        self._latent_h = self._latent_w = int(sqrt(self._base_patches))
        self._embedding_dim: int = self._latent_dim * self._base_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to latent embeddings.

        Args:
            x: Images ``(B, 3, H, W)`` in ``[0, 1]``.

        Returns:
            Latent embeddings ``(B, C, h, w)`` where ``C = latent_dim``
            and ``h = w = encoder_input_size // encoder_patch_size``.
        """
        # Resize to encoder's expected resolution if necessary
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(
                x,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bicubic",
                align_corners=False,
            )

        # ImageNet normalisation
        x = (x - self.encoder_mean) / self.encoder_std

        # ViT encoder forward — returns (B, N, C)
        z = self.encoder(x)

        # Reshape to spatial grid: (B, N, C) → (B, C, h, w)
        b, n, c = z.shape
        h_lat = w_lat = int(sqrt(n))
        z = z.transpose(1, 2).view(b, c, h_lat, w_lat)

        # Z-score normalisation with precomputed stats
        if self._do_normalization:
            mean = self.latent_mean if self.latent_mean is not None else 0
            var = self.latent_var if self.latent_var is not None else 1
            z = (z - mean) / torch.sqrt(var + self._eps)

        return z

    @property
    def latent_channels(self) -> int:
        return self._latent_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

