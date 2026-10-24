from abc import ABC, abstractmethod

import einops
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch import nn

from wm_repr.vae.autoencoder import BaseAEWrapper
from wm_repr.vae.distributions import DiagonalGaussianDistribution


class ProjectionHead(nn.Module):
    def __init__(self, cfg, latent_channels: int):
        super().__init__()
        self.depth = cfg.model.encoder.proj_head.depth
        self.hidden_dim = cfg.model.encoder.proj_head.hidden_dim
        self.kernel_size = cfg.model.encoder.proj_head.kernel_size

        layers, prev_dim = [], latent_channels
        for _ in range(self.depth - 1):
            layers.append(
                nn.Conv2d(
                    prev_dim,
                    self.hidden_dim,
                    self.kernel_size,
                    padding="same",
                )
            )
            layers.append(nn.ReLU())
            prev_dim = self.hidden_dim
        layers.append(
            nn.Conv2d(
                prev_dim,
                latent_channels,
                self.kernel_size,
                padding="same",
            )
        )
        self.head = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        # zero-init + residual connection --> preserve initial behavior of pretrained encoder
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.head(x)


class BaseJEPAEncoder(nn.Module, ABC):
    @property
    @abstractmethod
    def latent_channels(self) -> int:
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    def encode(
        self, x: torch.Tensor, *, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode with optional stochastic sampling.

        Returns:
            (z, kl_loss_or_none): latent tensor and per-batch-mean KL to N(0,1),
            or None when KL is not available (e.g. non-variational encoders).
        """
        return self.forward(x), None


class BaseJEPADecoder(nn.Module):
    """Decode latents (B, C, h, w) to images (B, 3, H, W) in [0, 1]."""


class EncoderFromSDVAE(BaseJEPAEncoder):
    def __init__(self, vae: AutoencoderKL, cfg):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

        self.input_h = cfg.data.params.input_h
        self.input_w = cfg.data.params.input_w
        self._embedding_dim, self._latent_channels = self._compute_embedding_dim()

    def encode(self, x, *, sample: bool = False, kl_of_mode: bool = False):
        x = 2.0 * x - 1.0  # [0, 1] -> [-1, 1]
        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)
        posterior = DiagonalGaussianDistribution(enc)
        if sample:
            return posterior.sample(), posterior.kl().mean()
        return posterior.mode(), posterior.kl().mean() if kl_of_mode else None

    def forward(self, x):
        return self.encode(x, sample=False)[0]

    @torch.no_grad()
    def _compute_embedding_dim(self):
        dummy_input = torch.zeros(1, 3, self.input_h, self.input_w).to(
            next(self.parameters()).device
        )
        embedding = self.forward(dummy_input)
        B, C, h, w = embedding.shape
        return C * h * w, C

    @property
    def latent_channels(self) -> int:
        return self._latent_channels

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


class DecoderFromSDVAE(BaseJEPADecoder):
    def __init__(self, vae: AutoencoderKL, cfg):
        super().__init__()
        self.decoder = vae.decoder
        self.post_quant_conv = vae.post_quant_conv

    def forward(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        x_hat = self.decoder(z)
        x_hat = (x_hat + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        return x_hat


class _BaseReprWrapper(BaseAEWrapper):
    """Shared encoder-side logic for learned-representation AE wrappers."""

    def __init__(
        self, encoder, mode: str = "eval", latent_normalization: str = "global"
    ):
        super().__init__(latent_normalization=latent_normalization)
        training = mode == "train"
        self.encoder = encoder.train(training).requires_grad_(training)
        self.train(training)

    def encode_batch(
        self, x: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # World-model training uses the normalized representation and no encoder KL.
        z, _ = self.encoder.encode(x, sample=sample)
        z = self.normalize(z)
        return z, None

    def _raw_encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    @property
    def latent_channels(self) -> int:
        return self.encoder.latent_channels  # type: ignore


class JEPAWrapper(_BaseReprWrapper):
    def __init__(self, jepa, mode: str = "eval", latent_normalization: str = "global"):
        super().__init__(
            jepa.encoder, mode=mode, latent_normalization=latent_normalization
        )
        training = mode == "train"
        self.decoder = jepa.decoder.train(training).requires_grad_(training)

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        z = self.denormalize(z)
        x_hat = self.decoder(z)
        return x_hat


class HierarchicalJEPAWrapper(_BaseReprWrapper):
    """Wrapper for using a HierarchicalJEPA's latent space to train a world model.

    The hierarchical decoder is conditioned on a past frame. Decoding is fully
    parallelized over frames, all conditioned on the ground-truth first frame.
    """

    def __init__(self, jepa, mode: str = "eval", latent_normalization: str = "global"):
        super().__init__(
            jepa.encoder, mode=mode, latent_normalization=latent_normalization
        )
        training = mode == "train"
        self.decoder = jepa.hierarchical_decoder.train(training).requires_grad_(
            training
        )

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "HierarchicalJEPAWrapper requires past-conditioned decoding. "
            "Use decode() with images and latents."
        )

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        *,
        images: torch.Tensor,
        latents: torch.Tensor,
        conditioning: str = "first",
        **kwargs,
    ) -> torch.Tensor:
        """Decode latents conditioned on the ground-truth first frame, parallelized over frames.

        All T frames are decoded through the hierarchical decoder, including
        context frames; frame 0 is conditioned on itself.

        Args:
            z: Predicted latent video (B, T, H, W, C_latent) — WM convention.
            images: Ground-truth video (B, T, H_img, W_img, C_img) in [0, 1].
            latents: GT-encoded latent video (B, T, H, W, C_latent) — same
                     shape/scale as z; its first frame is passed as z_past.
            conditioning: Must be "first" (GT[0] is the context for all frames).

        Returns:
            Decoded video (B, T, H_img, W_img, C_img) in [0, 1].
        """
        if conditioning != "first":
            raise ValueError(f"Unknown conditioning mode: {conditioning!r}")
        B, T, *_ = z.shape
        z_bchw = self.denormalize(einops.rearrange(z, "b t h w c -> b t c h w"))
        gt_z_bchw = self.denormalize(
            einops.rearrange(latents, "b t h w c -> b t c h w")
        )
        images_bchw = einops.rearrange(images, "b t h w c -> b t c h w")

        z_t = z_bchw  # (B, T, C, h, w)
        s_past = images_bchw[:, :1].expand(-1, T, -1, -1, -1)
        z_past = gt_z_bchw[:, :1].expand(-1, T, -1, -1, -1)

        # Flatten (B, T) -> (B*T) for batched decode
        s_past_flat = einops.rearrange(s_past, "b t c h w -> (b t) c h w")
        z_past_flat = einops.rearrange(z_past, "b t c h w -> (b t) c h w")
        z_t_flat = einops.rearrange(z_t, "b t c h w -> (b t) c h w")

        decoded_flat = self.decoder(s_past_flat, z_past_flat, z_t_flat)
        return einops.rearrange(decoded_flat, "(b t) c h w -> b t h w c", b=B)

    @property
    def is_conditioned(self) -> bool:
        return True
