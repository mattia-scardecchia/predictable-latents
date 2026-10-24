"""
Hierarchical JEPA: JEPA loss + past-conditioned reconstruction.

Combines:
- JEPA loss: Latent-space dynamics prediction (encoder + predictor)
- Reconstruction loss: Past-conditioned pixel and LPIPS losses

Uses the TC dataset (current, future, other, actions) like LitTrueJEPA.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wm_repr.hierarchical.base import BaseHierarchicalJEPA
from wm_repr.hierarchical.conv_decoder import PastConditionedConvDecoder
from wm_repr.hierarchical.vit_decoder import PastConditionedViTDecoder
from wm_repr.jepa.autoencoder import BaseJEPAEncoder, HierarchicalJEPAWrapper
from wm_repr.jepa.predictor import Predictor
from wm_repr.jepa.registry import register


@register
class LitHierarchicalJEPA(BaseHierarchicalJEPA):
    """
    Hierarchical JEPA: JEPA loss + past-conditioned reconstruction.

    Supports two decoder types:
    - PastConditionedConvDecoder: FLUX decoder with cross-attention adapters
    - PastConditionedViTDecoder: RAE decoder with cross-attention adapters

    Both are called as decoder(s_past, z_past, z_t) and reconstruct the frame of
    z_t from z_t and the pixels of s_past; z_past is not used.

    Training objectives:
    - JEPA loss: L1(predictor(z, a), z') where z=encoder(s), z'=encoder(s')
    - Reconstruction loss: decoder(s, z, z') ≈ s' (pixel + LPIPS)
    """

    wrapper_cls = HierarchicalJEPAWrapper

    def __init__(
        self,
        cfg: DictConfig | dict,
        encoder: Optional[BaseJEPAEncoder] = None,
        predictor: Optional[Predictor] = None,
        hierarchical_decoder: Optional[
            PastConditionedConvDecoder | PastConditionedViTDecoder
        ] = None,
    ):
        super().__init__(cfg, encoder, predictor, hierarchical_decoder)
        self.future_pred_weight = self.cfg.model.encoder.get("future_pred_weight", 0.5)
        self.auxiliary_weight = self.future_pred_weight

    def _auxiliary_loss(self, z, z_future, batch, split):
        target = z_future.detach() if split == "train" else z_future
        loss = F.l1_loss(self.predictor(z, batch["actions"]), target)
        return loss, {f"{split}/latent_pred_loss": loss}

    # -------------------------------------------------------------------------
    # Training and Validation
    # -------------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        model_opt = self.optimizers()
        model_sched = self.lr_schedulers()

        # --- Model update ---
        self.toggle_optimizer(model_opt)

        # JEPA loss: predict future embedding
        z, kl_curr = self._encode(current)
        z_fut, kl_fut = self._encode(future)
        auxiliary_loss, auxiliary_metrics = self._auxiliary_loss(z, z_fut, batch, "train")

        # Reconstruction loss: decode the future frame from its embedding, conditioned
        # on the current frame's pixels; its gradients reach only the future encoding.
        # The latent-prediction loss detaches its target and reaches only the current
        # encoding (the inverse-dynamics loss of the subclass reaches both).
        recon = self._decode(current, z, z_fut)
        dec_losses, _ = self._decoder_loss(recon, future, split="train")

        # Combine losses (similar weighting to TrueJEPA)
        w = 2 * self.auxiliary_weight
        model_loss = w * auxiliary_loss + (2 - w) * dec_losses["train/dec_loss"]

        # KL loss
        kl_loss = None
        if kl_curr is not None:
            kl_terms = [kl_curr] + ([kl_fut] if kl_fut is not None else [])
            kl_loss = self.kl_weight * sum(kl_terms) / len(kl_terms)
            model_loss = model_loss + kl_loss

        model_opt.zero_grad()
        self.manual_backward(model_loss)
        self.clip_gradients(
            model_opt,
            gradient_clip_val=self.cfg.training.grad_clip_val,
            gradient_clip_algorithm="norm",
        )
        model_opt.step()
        model_sched.step()
        self.untoggle_optimizer(model_opt)

        # --- Logging ---
        self.log("train/loss", model_loss, **self.train_log_kwargs)
        self.log_dict(auxiliary_metrics, batch_size=current.shape[0], **self.train_log_kwargs)
        self.log("train/weighted_reconstruction_loss", (2 - w) * dec_losses["train/dec_loss"], **self.train_log_kwargs)
        if kl_loss is not None:
            self.log("train/kl_loss", kl_loss, **self.train_log_kwargs)
        self.log_dict(dec_losses, **self.train_log_kwargs)
        lr_model = model_opt.param_groups[0]["lr"]
        self.log("train/lr_model", lr_model, **self.train_log_kwargs)

    def validation_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)

        # Encode both frames
        z, _ = self._encode(current, kl=False)
        z_fut, _ = self._encode(future, kl=False)

        # JEPA loss
        auxiliary_loss, auxiliary_metrics = self._auxiliary_loss(z, z_fut, batch, "val")

        # Reconstruction loss
        recon = self._decode(current, z, z_fut)
        dec_losses, _ = self._decoder_loss(recon, future, split="val")

        # Combined loss
        w = 2 * self.auxiliary_weight
        model_loss = w * auxiliary_loss + (2 - w) * dec_losses["val/dec_loss"]

        self.log("val/loss", model_loss, **self.val_log_kwargs)
        self.log_dict(auxiliary_metrics, batch_size=current.shape[0], **self.val_log_kwargs)
        self.log("val/weighted_reconstruction_loss", (2 - w) * dec_losses["val/dec_loss"], **self.val_log_kwargs)
        self.log_dict(dec_losses, **self.val_log_kwargs)
