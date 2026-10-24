from abc import ABC, abstractmethod
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from wm_repr.jepa.autoencoder import (
    BaseJEPADecoder,
    BaseJEPAEncoder,
    DecoderFromSDVAE,
    EncoderFromSDVAE,
    ProjectionHead,
)
from wm_repr.vae.losses import LPIPS

from .predictor import ConvPredictor, Predictor


def build_optimizer(cfg: DictConfig, param_groups: list[dict]):
    """AdamW with a linear warm-up from ``start_factor`` times the learning rate over ``steps``."""
    optimizer = torch.optim.AdamW(
        param_groups, **OmegaConf.to_container(cfg.training.optimizer.defaults, resolve=True)
    )
    start, steps = float(cfg.training.warmup.start_factor), int(cfg.training.warmup.steps)

    def factor(step):
        return start + (1.0 - start) * (step / steps) if step < steps else 1.0

    return [optimizer], [LambdaLR(optimizer, lr_lambda=factor)]


def build_predictor(cfg: DictConfig, latent_channels: int) -> nn.Module:
    """Build the action-conditioned latent predictor from scratch."""
    pred_cfg = cfg.model.predictor
    if pred_cfg.get("type", "conv") != "conv":
        raise ValueError("The paper uses the convolutional action-conditioned predictor")
    return ConvPredictor(
        latent_channels=latent_channels,
        action_dim=cfg.data.params.action_dim,
        d_model=pred_cfg.d_model,
        n_heads=pred_cfg.n_heads,
        num_layers=pred_cfg.num_layers,
        dropout=pred_cfg.dropout,
        max_actions=pred_cfg.max_actions,
        mlp_mult=pred_cfg.mlp_mult,
        action_encoder_layers=pred_cfg.action_encoder_layers,
        conv_kernel_size=pred_cfg.conv_kernel_size,
        norm_num_groups=pred_cfg.norm_num_groups,
    )


def build_rae(cfg: DictConfig):
    """Instantiate an RAE model from config."""
    from wm_repr.rae.rae import RAE

    rae_cfg = cfg.repr.rae
    params = OmegaConf.to_container(rae_cfg, resolve=True)
    return RAE(**params)


def _pretrained_vae(cfg: DictConfig) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(cfg.repr.ckpt, **cfg.repr.get("params", {}))
    vae.enable_gradient_checkpointing()
    vae.train(True).requires_grad_(True)
    return vae


def build_encoder(cfg: DictConfig) -> BaseJEPAEncoder:
    """The pretrained autoencoder's encoder."""
    return EncoderFromSDVAE(_pretrained_vae(cfg), cfg)


def build_decoder(cfg: DictConfig) -> BaseJEPADecoder:
    """The pretrained autoencoder's decoder."""
    return DecoderFromSDVAE(_pretrained_vae(cfg), cfg)


class BaseJEPA(pl.LightningModule, ABC):
    """Shared reconstruction loss and optimizer for ordinary-decoder objectives."""

    # Only the contrastive objective projects its queries and keys.
    uses_projection_head = False

    def __init__(
        self,
        cfg: DictConfig | dict,
        encoder: Optional[BaseJEPAEncoder] = None,
        predictor: Optional[Predictor] = None,
        decoder: Optional[BaseJEPADecoder] = None,
    ):
        super().__init__()

        # Save Config
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg

        # Encoder/Predictor/Decoder
        self.encoder = encoder if encoder else build_encoder(cfg)
        latent_channels: int = self.encoder.latent_channels
        self.head = ProjectionHead(cfg, latent_channels) if self.uses_projection_head else None
        self.predictor = (
            predictor if predictor else build_predictor(cfg, latent_channels)
        )
        self.decoder = decoder if decoder else build_decoder(cfg)
        self.lpips = LPIPS().eval()
        # Set only by the contrastive objective; configure_optimizers always keeps
        # its parameter group, empty for the other objectives.
        self.bilinear_W = None

        # The latent is the posterior mean; the KL objectives penalize the posterior's KL.
        self.kl_weight = cfg.model.encoder.get("kl_weight", 0.0)
        self.kl_without_sampling = cfg.model.encoder.get("kl_without_sampling", False)

        # Miscellanea
        self.automatic_optimization = False
        self.train_log_kwargs = dict(
            on_step=True, on_epoch=False, prog_bar=False, logger=True
        )
        self.val_log_kwargs = dict(
            on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

    def _encode(
        self, x: torch.Tensor, *, kl: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode to the posterior mean.

        Returns:
            (z, kl_loss_or_none): the KL of the posterior to the prior when ``kl`` is set and the
            objective penalizes it, otherwise None.
        """
        if kl and self.kl_without_sampling:
            return self.encoder.encode(x, sample=False, kl_of_mode=True)
        return self.encoder.encode(x, sample=False)

    def reconstruction_loss(self, input_embeddings, target_frames, split: str):
        recon = self.decoder(input_embeddings)
        recon_normed = recon * 2.0 - 1.0
        targets_normed = target_frames * 2.0 - 1.0
        pixel_loss = F.l1_loss(recon, target_frames)
        lpips_loss = self.lpips(recon_normed, targets_normed, reduction="mean")
        recon_loss = pixel_loss + lpips_loss
        losses = {
            f"{split}/recon_loss": recon_loss,
            f"{split}/pixel_loss": pixel_loss,
            f"{split}/lpips_loss": lpips_loss,
        }
        return losses, recon, recon_normed

    def decoder_loss(self, input_embeddings, target_frames, split: str):
        losses, recon, _ = self.reconstruction_loss(input_embeddings, target_frames, split)
        losses[f"{split}/dec_loss"] = losses[f"{split}/recon_loss"]
        return losses, recon

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return build_optimizer(self.cfg, [
            {"params": self.encoder.parameters()},
            {"params": self.head.parameters() if self.head is not None else []},
            {"params": self.predictor.parameters()},
            {"params": self.decoder.parameters()},
            {"params": [self.bilinear_W] if self.bilinear_W is not None else []},
        ])

    def on_fit_start(self):
        self.encoder.to(memory_format=torch.channels_last)
        self.decoder.to(memory_format=torch.channels_last)
        self.predictor.to(memory_format=torch.channels_last)
        if self.head is not None:
            self.head.to(memory_format=torch.channels_last)
        self._on_toggle_mode(train=True)

    def _on_toggle_mode(self, train: bool):
        self.lpips.eval()

    def on_train_start(self):
        self._on_toggle_mode(train=True)

    def on_train_epoch_start(self) -> None:
        self._on_toggle_mode(train=True)

    def on_validation_epoch_start(self) -> None:
        self._on_toggle_mode(train=False)
