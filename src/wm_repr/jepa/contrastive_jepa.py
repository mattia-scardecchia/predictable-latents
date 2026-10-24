from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import Parameter

from wm_repr.jepa.autoencoder import BaseJEPADecoder, BaseJEPAEncoder, JEPAWrapper
from wm_repr.jepa.jepa import BaseJEPA
from wm_repr.jepa.predictor import Predictor
from wm_repr.jepa.queue import KeyQueue
from wm_repr.jepa.registry import register
from wm_repr.utils.training import update_ema


@register
class LitContrastiveJEPA(BaseJEPA):
    """
    Implements contrastive loss using a large queue of negative samples, a momentum key encoder, and a projection head.
    Provides training and validation step implementations.
    """

    wrapper_cls = JEPAWrapper
    uses_projection_head = True

    def __init__(
        self,
        cfg: DictConfig,
        encoder: Optional[BaseJEPAEncoder] = None,
        predictor: Optional[Predictor] = None,
        decoder: Optional[BaseJEPADecoder] = None,
    ):
        super().__init__(cfg, encoder, predictor, decoder)
        encoder_cfg = self.cfg.model.encoder
        # Identity initialization: training starts from dot-product logits.
        D = int(self.encoder.embedding_dim)
        self.bilinear_W = Parameter(torch.eye(D, D))
        self.contrastive_loss_weight = encoder_cfg.contrastive_loss_weight

        # EMA encoder
        self.ema_decay = encoder_cfg.ema.decay
        self.ema_encoder = deepcopy(self.encoder).eval()
        self.ema_head = deepcopy(self.head).eval()
        for param in self.ema_encoder.parameters():
            param.requires_grad = False
        for param in self.ema_head.parameters():
            param.requires_grad = False

        # Key Queue for negative samples
        self.key_queue = KeyQueue(
            queue_size=encoder_cfg.memory_bank.size,
            embedding_dim=self.encoder.embedding_dim,  # type: ignore
        )

    def contrastive_loss(self, current, actions, future, *, kl: bool = True):
        B = current.shape[0]
        curr_emb, kl_loss = self._encode(current, kl=kl)
        pred_fut_emb = self.predictor(curr_emb, actions)  # B, ...
        # The other futures in the batch are the in-batch negatives.
        with torch.no_grad():
            key = self.ema_head(self.ema_encoder(future))
        query = F.normalize(self.head(pred_fut_emb).flatten(start_dim=1), dim=-1)  # B, D
        key = F.normalize(key.flatten(start_dim=1), dim=-1)  # B, D

        # Retrieve more key embeddings from queue.
        bank = self.key_queue.get()  # K, D or empty
        if bank.numel() == 0:
            all_keys = key  # B, D
        else:
            all_keys = torch.cat([key, bank], dim=0)  # B + K, D

        # Bilinear logits with temperature 1, and labels
        logits = query @ self.bilinear_W @ all_keys.t()
        labels = torch.arange(B, device=logits.device)

        # Update queue (enqueue only future embeddings, skip during validation)
        if self.training:
            self.key_queue.enqueue(key.detach())

        contrastive_loss = F.cross_entropy(logits, labels)
        return contrastive_loss, curr_emb, pred_fut_emb, kl_loss

    def training_step(self, batch, batch_idx):
        """Manual optimization; does not support grad accumulation."""
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]
        model_opt = self.optimizers()
        model_sched = self.lr_schedulers()

        # Train encoder/predictor and decoder
        self.toggle_optimizer(model_opt)
        contrastive_loss, curr_emb, pred_fut_emb, kl = self.contrastive_loss(
            current, actions, future
        )
        dec_losses, curr_recon = self.decoder_loss(curr_emb, current, split="train")
        c = 2 * self.contrastive_loss_weight
        model_loss = c * contrastive_loss + (2 - c) * dec_losses["train/dec_loss"]

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
        update_ema(self.ema_encoder, self.encoder, self.ema_decay)
        update_ema(self.ema_head, self.head, self.ema_decay)
        self.untoggle_optimizer(model_opt)

        # Logging
        self.log("train/contrastive_loss", contrastive_loss, **self.train_log_kwargs)  # type: ignore
        self.log("train/loss", model_loss, **self.train_log_kwargs)  # type: ignore
        if kl_loss is not None:
            self.log("train/kl_loss", kl_loss, **self.train_log_kwargs)  # type: ignore
        self.log_dict(dec_losses, **self.train_log_kwargs)  # type: ignore
        lr_model = model_opt.param_groups[0]["lr"]
        self.log("train/lr_model", lr_model, **self.train_log_kwargs)  # type: ignore

    def validation_step(self, batch, batch_idx):
        current = batch["current"].to(memory_format=torch.channels_last)
        future = batch["future"].to(memory_format=torch.channels_last)
        actions = batch["actions"]
        contrastive_loss, z, zphat, _kl = self.contrastive_loss(
            current, actions, future, kl=False,
        )
        dec_losses, recon = self.decoder_loss(z.detach(), current, split="val")
        c = 2 * self.contrastive_loss_weight
        model_loss = c * contrastive_loss + (2 - c) * dec_losses["val/dec_loss"]

        self.log("val/contrastive_loss", contrastive_loss, **self.val_log_kwargs)  # type: ignore
        self.log("val/loss", model_loss, **self.val_log_kwargs)  # type: ignore
        self.log_dict(dec_losses, **self.val_log_kwargs)  # type: ignore

    def _on_toggle_mode(self, train: bool) -> None:
        self.ema_encoder.train(False)
        self.ema_head.train(False)
        return super()._on_toggle_mode(train)

    def on_fit_start(self) -> None:
        self.ema_encoder = self.ema_encoder.to(memory_format=torch.channels_last)  # type: ignore
        self.ema_head = self.ema_head.to(memory_format=torch.channels_last)  # type: ignore
        return super().on_fit_start()
