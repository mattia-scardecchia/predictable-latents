from abc import ABC, abstractmethod
import logging

import einops
import torch
from torch import nn
from diffusers import AutoencoderKL as SDVAE


class BaseAEWrapper(nn.Module, ABC):
    """Interface for latent space and decoder for use with a world model."""

    def __init__(self, latent_normalization: str = "global"):
        super().__init__()
        if latent_normalization not in ("global", "zscore", "zscore-channel"):
            raise ValueError(f"Unknown latent normalization: {latent_normalization!r}")
        self.latent_normalization = latent_normalization
        self.register_buffer("std_emb", torch.tensor(0.0))
        self.register_buffer("mean_emb", torch.tensor(0.0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # std_emb/mean_emb are registered as scalar placeholders, but
        # compute_embedding_stats() reassigns them with shape (C,H,W) or
        # (C,1,1) for zscore modes. Re-register with the checkpoint's shape
        # so load_state_dict can fill them without a shape mismatch error.
        for name in ("std_emb", "mean_emb"):
            key = prefix + name
            if key in state_dict and state_dict[key].shape != getattr(self, name).shape:
                self.register_buffer(name, torch.zeros_like(state_dict[key]))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @abstractmethod
    def encode_batch(
        self, x: torch.Tensor, sample: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pass

    @abstractmethod
    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _raw_encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Encode without normalization. Used by compute_embedding_stats."""
        pass

    def encode(
        self, x: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, "b t h w c -> (b t) c h w")
        z, kl = self.encode_batch(x, sample)
        z = einops.rearrange(z, "(b t) c h w -> b t h w c", b=B, t=T)
        return z, kl

    @torch.no_grad()
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        B, T, H, W, C = z.shape
        z = einops.rearrange(z, "b t h w c -> (b t) c h w")
        x_hat = self.decode_batch(z)
        x_hat = einops.rearrange(x_hat, "(b t) c h w -> b t h w c", b=B, t=T)
        return x_hat

    @property
    def is_conditioned(self) -> bool:
        return False

    @property
    @abstractmethod
    def latent_channels(self) -> int:
        pass

    @property
    def has_embedding_stats(self) -> bool:
        if self.latent_normalization in ("zscore", "zscore-channel"):
            return self.std_emb.dim() > 0
        return self.std_emb.item() != 0.0

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_embedding_stats:
            raise ValueError("Embedding stats have not been computed.")
        if self.latent_normalization == "global":
            return z / self.std_emb
        # zscore modes shift and scale
        return (z - self.mean_emb) / self.std_emb

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        if not self.has_embedding_stats:
            raise ValueError("Embedding stats have not been computed.")
        if self.latent_normalization == "global":
            return z * self.std_emb
        return z * self.std_emb + self.mean_emb

    @torch.no_grad()
    def compute_embedding_stats(self, data_loader: torch.utils.data.DataLoader):
        all_embeddings = []
        device = next(self.parameters()).device
        for batch in data_loader:
            B, T, C, H, W = batch["clip"].shape
            times = torch.randint(0, T, (B,))
            x = batch["clip"][torch.arange(B), times].to(device)
            emb = self._raw_encode_batch(x)
            all_embeddings.append(emb)
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, C, H, W)
        if self.latent_normalization == "zscore":
            self.mean_emb = all_embeddings.mean(dim=0)  # (C, H, W)
            self.std_emb = all_embeddings.std(dim=0) + 1e-6  # (C, H, W)
            logging.info(
                "Computed per-coordinate embedding stats: "
                f"mean in [{self.mean_emb.min():.4f}, {self.mean_emb.max():.4f}], "
                f"std in [{self.std_emb.min():.4f}, {self.std_emb.max():.4f}]"
            )
        elif self.latent_normalization == "zscore-channel":
            self.mean_emb = all_embeddings.mean(dim=(0, 2, 3), keepdim=True).squeeze(0)  # (C, 1, 1)
            self.std_emb = all_embeddings.std(dim=(0, 2, 3), keepdim=True).squeeze(0) + 1e-6  # (C, 1, 1)
            logging.info(
                "Computed per-channel embedding stats: "
                f"mean in [{self.mean_emb.min():.4f}, {self.mean_emb.max():.4f}], "
                f"std in [{self.std_emb.min():.4f}, {self.std_emb.max():.4f}]"
            )
        else:
            # global uses scalar stats; normalization divides by the std only
            self.mean_emb.fill_(all_embeddings.mean().item())
            self.std_emb.fill_((all_embeddings.std() + 1e-6).item())
            logging.info(
                f"Computed embedding stats: mean={self.mean_emb.item():.4f}, "
                f"std={self.std_emb.item():.4f}"
            )


class SDVAEWrapper(BaseAEWrapper):
    """Wrapper for using the diffusers.AutoencoderKL latent space to train a world model."""

    def __init__(self, vae: SDVAE, mode: str = "eval", latent_normalization: str = "global"):
        super().__init__(latent_normalization=latent_normalization)
        self.vae = vae
        training = mode == "train"
        self.training = training
        self.vae.train(training).requires_grad_(training)

    def _raw_encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Return the posterior mode of the pretrained VAE for images in [0, 1]."""
        return self.vae.encode(x * 2 - 1).latent_dist.mode()  # type: ignore

    def encode_batch(
        self, x: torch.Tensor, sample: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if sample:
            raise ValueError("World models encode the posterior mode; sampling is not supported")
        return self.normalize(self._raw_encode_batch(x)), None

    def decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        z = self.denormalize(z)
        x_hat = self.vae.decode(z, return_dict=False)[0]
        x_hat = (x_hat + 1) / 2
        return x_hat

    @property
    def latent_channels(self) -> int:
        return self.vae.config.latent_channels  # type: ignore
