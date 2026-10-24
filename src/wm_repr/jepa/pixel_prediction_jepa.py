from typing import Optional

import torch
from omegaconf import DictConfig

from wm_repr.jepa.autoencoder import BaseJEPADecoder, BaseJEPAEncoder, JEPAWrapper
from wm_repr.jepa.jepa import BaseJEPA
from wm_repr.jepa.predictor import Predictor
from wm_repr.jepa.registry import register


@register
class LitPixelPredictionJEPA(BaseJEPA):
    """
    Pixel future prediction. Computes future prediction loss in pixel space.
    Given (s, a, s'):
        z = encoder(s), z' = predictor(z, a)
        loss: dec(z) ≈ s  and  dec(z') ≈ s'
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
        self.future_recon_weight = self.cfg.model.encoder.future_recon_weight

    def training_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]
        model_opt = self.optimizers()
        model_sched = self.lr_schedulers()

        # --- Encoder / Predictor / Decoder ---
        self.toggle_optimizer(model_opt)

        z, kl = self._encode(current)
        z_pred = self.predictor(z, actions)

        # Current frame: pixel and LPIPS reconstruction losses.
        curr_dec_losses, curr_recon = self.decoder_loss(z, current, split="train")

        # Future frame: pixel and LPIPS losses on the predicted latent.
        fut_recon_losses, fut_recon, _ = self.reconstruction_loss(z_pred, future, "train_fut")

        # Combine
        w = 2 * self.future_recon_weight
        model_loss = (2 - w) * curr_dec_losses["train/dec_loss"] + w * fut_recon_losses[
            "train_fut/recon_loss"
        ]

        # KL loss
        kl_loss = None
        if kl is not None:
            kl_loss = self.kl_weight * kl
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
        if kl_loss is not None:
            self.log("train/kl_loss", kl_loss, **self.train_log_kwargs)
        self.log_dict(curr_dec_losses, **self.train_log_kwargs)
        self.log_dict(fut_recon_losses, **self.train_log_kwargs)
        lr_model = model_opt.param_groups[0]["lr"]
        self.log("train/lr_model", lr_model, **self.train_log_kwargs)

    def validation_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]

        z, _ = self._encode(current, kl=False)
        z_pred = self.predictor(z, actions)

        curr_dec_losses, curr_recon = self.decoder_loss(
            z.detach(), current, split="val"
        )
        fut_recon_losses, _, _ = self.reconstruction_loss(z_pred.detach(), future, "val_fut")

        w = 2 * self.future_recon_weight
        model_loss = (2 - w) * curr_dec_losses["val/dec_loss"] + w * fut_recon_losses[
            "val_fut/recon_loss"
        ]

        self.log("val/loss", model_loss, **self.val_log_kwargs)
        self.log_dict(curr_dec_losses, **self.val_log_kwargs)
        self.log_dict(fut_recon_losses, **self.val_log_kwargs)
