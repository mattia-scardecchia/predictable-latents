"""Shared pretrained encoder, conditioned decoder, and optimization."""

from abc import abstractmethod
from copy import deepcopy
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from omegaconf import DictConfig, OmegaConf

from wm_repr.hierarchical.conv_decoder import PastConditionedConvDecoder
from wm_repr.hierarchical.vit_decoder import PastConditionedViTDecoder
from wm_repr.jepa.autoencoder import BaseJEPAEncoder, EncoderFromSDVAE
from wm_repr.jepa.jepa import build_optimizer, build_rae, build_predictor
from wm_repr.jepa.predictor import Predictor
from wm_repr.vae.losses import LPIPS


class BaseHierarchicalJEPA(pl.LightningModule):
    """Abstract base for hierarchical JEPA variants.

    Handles encoder, predictor, and hierarchical decoder,
    loss computation, optimizer setup, and lifecycle hooks.

    Subclasses must implement:
    - training_step
    - validation_step
    """

    def __init__(
        self,
        cfg: DictConfig | dict,
        encoder: Optional[BaseJEPAEncoder] = None,
        predictor: Optional[Predictor] = None,
        hierarchical_decoder: Optional[
            PastConditionedConvDecoder | PastConditionedViTDecoder
        ] = None,
    ):
        super().__init__()

        # Save Config
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True)})
        self.cfg = cfg

        # --- Build autoencoder (shared source for encoder and decoder) ---
        arch = cfg.repr.get("type", "hf-vae")

        if arch == "rae":
            rae = build_rae(cfg)
            vae = None  # RAE path doesn't use a shared VAE object
        else:
            rae = None
            vae = AutoencoderKL.from_pretrained(cfg.repr.ckpt, **cfg.repr.get("params", {}))

        # --- Encoder ---
        if encoder:
            self.encoder = encoder
        elif arch == "rae":
            from wm_repr.jepa.rae_autoencoder import EncoderFromRAE

            self.encoder = EncoderFromRAE(rae)
        else:
            encoder_vae = deepcopy(vae)
            encoder_vae.enable_gradient_checkpointing()
            encoder_vae.train(True).requires_grad_(True)
            self.encoder = EncoderFromSDVAE(encoder_vae, cfg)

        # --- Predictor ---
        latent_channels: int = self.encoder.latent_channels
        self.predictor = (
            predictor if predictor else build_predictor(cfg, latent_channels)
        )

        # --- Hierarchical Decoder ---
        if hierarchical_decoder:
            self.hierarchical_decoder = hierarchical_decoder
        elif arch == "rae":
            hier_dec_cfg = cfg.model.hierarchical_decoder
            vit_cfg = hier_dec_cfg.vit

            self.hierarchical_decoder = PastConditionedViTDecoder(
                rae=rae,
                context_dim=vit_cfg.context_dim,
                num_heads=hier_dec_cfg.num_heads,
                dropout=hier_dec_cfg.dropout,
                adapter_interval=vit_cfg.adapter_interval,
                pretrained_decoder=cfg.repr.rae.pretrained_decoder_path
                is not None,
            )
        else:
            hier_dec_cfg = cfg.model.hierarchical_decoder
            # Latent-first decoder with pretrained VAE + cross-attention
            self.hierarchical_decoder = PastConditionedConvDecoder(
                vae=deepcopy(vae),
                context_dim=hier_dec_cfg.get("conv", {}).get("context_dim", 512),
                num_heads=hier_dec_cfg.get("num_heads", 8),
                num_groups=hier_dec_cfg.get("num_groups", 32),
                dropout=hier_dec_cfg.get("dropout", 0.0),
            )

        # --- LPIPS (frozen) ---
        self.lpips = LPIPS().eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

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

    # -------------------------------------------------------------------------
    # Encoding helper
    # -------------------------------------------------------------------------

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

    def _decode(self, current, z_context, z_target, *, conditioned=True):
        """Decode with the past frame, or without it for the decoder interventions."""
        supports_unconditioned = getattr(
            self.hierarchical_decoder, "supports_unconditioned", False
        )
        if not conditioned:
            if not supports_unconditioned:
                raise ValueError("This decoder has no unconditioned forward path")
            return self.hierarchical_decoder(
                current, z_context, z_target, unconditioned=True
            )
        return self.hierarchical_decoder(current, z_context, z_target)

    # -------------------------------------------------------------------------
    # Loss methods
    # -------------------------------------------------------------------------

    def _reconstruction_loss(self, recon, target_frames, split: str):
        """Compute reconstruction loss (pixel + LPIPS)."""
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
        return losses, recon_normed

    def _decoder_loss(self, recon, target_frames, split: str):
        losses, _ = self._reconstruction_loss(recon, target_frames, split)
        losses[f"{split}/dec_loss"] = losses[f"{split}/recon_loss"]
        return losses, recon

    # -------------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def training_step(self, batch, batch_idx): ...

    @abstractmethod
    def validation_step(self, batch, batch_idx): ...

    # -------------------------------------------------------------------------
    # Optimizer and Scheduler
    # -------------------------------------------------------------------------

    def configure_optimizers(self):
        # The hierarchical decoder's frozen context encoder is left out.
        hier_dec_params = [
            p for p in self.hierarchical_decoder.parameters() if p.requires_grad
        ]
        return build_optimizer(self.cfg, [
            {"params": list(self.encoder.parameters())},
            {"params": list(self.predictor.parameters())},
            {"params": hier_dec_params},
        ])

    # -------------------------------------------------------------------------
    # Train/Eval Mode Handling
    # -------------------------------------------------------------------------

    def _on_toggle_mode(self, train: bool):
        self.lpips.eval()
        decoder = self.hierarchical_decoder
        if hasattr(decoder, "context_encoder"):
            ctx = decoder.context_encoder
            if hasattr(ctx, "encoder"):
                ctx.encoder.eval()

    def on_train_start(self):
        self._on_toggle_mode(train=True)

    def on_train_epoch_start(self) -> None:
        self._on_toggle_mode(train=True)

    def on_validation_epoch_start(self) -> None:
        self._on_toggle_mode(train=False)

    def on_fit_start(self):
        if self.cfg.repr.get("type", "hf-vae") != "rae":
            self.encoder.to(memory_format=torch.channels_last)
            self.hierarchical_decoder.to(memory_format=torch.channels_last)
        self.predictor.to(memory_format=torch.channels_last)
        self._on_toggle_mode(train=True)
