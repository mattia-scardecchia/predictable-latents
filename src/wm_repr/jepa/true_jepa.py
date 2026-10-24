from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from wm_repr.jepa.autoencoder import BaseJEPADecoder, BaseJEPAEncoder, JEPAWrapper
from wm_repr.jepa.jepa import BaseJEPA
from wm_repr.jepa.predictor import Predictor
from wm_repr.jepa.registry import register


@register
class LitTrueJEPA(BaseJEPA):
    """
    True JEPA. Computes future prediction loss in latent space.
    Given (s, a, s'):
        z = encoder(s), z' = predictor(z, a), z_fut = encoder(s')
        loss: dec(z) ≈ s  and  z' ≈ z_fut (latent space L1 loss, z_fut detached)
    Uses the TC dataset (current, future, other, actions); "other" is unused.
    """

    wrapper_cls = JEPAWrapper

    def __init__(
        self,
        cfg: DictConfig,
        encoder: Optional[BaseJEPAEncoder] = None,
        predictor: Optional[Predictor] = None,
        decoder: Optional[BaseJEPADecoder] = None,
    ):
        super().__init__(cfg, encoder, predictor, decoder)
        self.future_pred_weight = self.cfg.model.encoder.future_pred_weight

    def training_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]
        model_opt = self.optimizers()
        model_sched = self.lr_schedulers()

        # --- Encoder / Predictor / Decoder ---
        self.toggle_optimizer(model_opt)

        z, kl_curr = self._encode(current)
        z_pred = self.predictor(z, actions)

        # Encode the future frame without gradient: it is only a prediction target.
        with torch.no_grad():
            z_fut, kl_fut = self._encode(future)
        z_fut = z_fut.detach()

        # Current frame: pixel and LPIPS reconstruction losses.
        curr_dec_losses, curr_recon = self.decoder_loss(z, current, split="train")

        # Future frame: latent prediction loss (L1 in latent space)
        latent_pred_loss = F.l1_loss(z_pred, z_fut)

        # Combine
        w = 2 * self.future_pred_weight
        model_loss = (2 - w) * curr_dec_losses["train/dec_loss"] + w * latent_pred_loss

        # KL loss (only when encoder provides it)
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
        self.log("train/latent_pred_loss", latent_pred_loss, **self.train_log_kwargs)
        if kl_loss is not None:
            self.log("train/kl_loss", kl_loss, **self.train_log_kwargs)
        self.log_dict(curr_dec_losses, **self.train_log_kwargs)
        lr_model = model_opt.param_groups[0]["lr"]
        self.log("train/lr_model", lr_model, **self.train_log_kwargs)

    def validation_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]

        z, _ = self._encode(current, kl=False)
        z_pred = self.predictor(z, actions)
        z_fut, _ = self._encode(future, kl=False)

        curr_dec_losses, curr_recon = self.decoder_loss(
            z.detach(), current, split="val"
        )
        latent_pred_loss = F.l1_loss(z_pred.detach(), z_fut)

        w = 2 * self.future_pred_weight
        model_loss = (2 - w) * curr_dec_losses["val/dec_loss"] + w * latent_pred_loss

        self.log("val/loss", model_loss, **self.val_log_kwargs)
        self.log("val/latent_pred_loss", latent_pred_loss, **self.val_log_kwargs)
        self.log_dict(curr_dec_losses, **self.val_log_kwargs)
