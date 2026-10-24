"""Diffusion-forcing training on a frozen visual representation."""

from copy import deepcopy
from pathlib import Path

import einops
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from wm_repr.utils.training import update_ema

from .diffusion import Diffusion
from .model import AttentionType, DiT, RotaryType


def relocate_assets(cfg, assets):
    """Relocate pretrained files while preserving the saved architecture settings."""
    root = Path(assets).expanduser().resolve()
    with open_dict(cfg):
        cfg.assets_root = str(root)
        if cfg.repr.get("type", "hf-vae") == "hf-vae":
            cfg.repr.ckpt = str(root / "flux1-schnell")
        elif cfg.repr.get("type") == "rae":
            from wm_repr.config import CONFIG_ROOT
            rae = cfg.repr.rae
            directory = root / "rae-dinov2-wreg-base"
            rae.encoder_config_path = str(directory / "dinov2")
            rae.encoder_params.dinov2_path = str(directory / "dinov2")
            rae.decoder_config_path = str(CONFIG_ROOT / "decoder" / "ViTXL")
            if rae.get("normalization_stat_path") is not None:
                rae.normalization_stat_path = str(directory / "stat.pt")
            if rae.get("pretrained_decoder_path") is not None:
                rae.pretrained_decoder_path = str(directory / "decoder_ViTXL_n08.pt")
    return cfg


def load_ae(cfg: DictConfig, mode: str = "eval"):
    """The pretrained FLUX autoencoder (``repr.type: hf-vae``) or a fine-tuned autoencoder checkpoint."""
    from diffusers import AutoencoderKL
    from wm_repr.vae.autoencoder import SDVAEWrapper

    normalization = cfg.training.get("latent_normalization", "global")
    if cfg.repr.get("type", "hf-vae") == "hf-vae":
        vae = AutoencoderKL.from_pretrained(cfg.repr.ckpt, **cfg.repr.get("params", {}))
        return SDVAEWrapper(vae, mode=mode, latent_normalization=normalization)
    ae = load_autoencoder_checkpoint(cfg.repr.ckpt, assets=cfg.get("assets_root"))
    return type(ae).wrapper_cls(ae, mode=mode, latent_normalization=normalization)


def load_autoencoder_checkpoint(path, assets=None):
    """Load a complete registered AE of the class the checkpoint records, rejecting every missing
    active parameter."""
    import wm_repr.hierarchical  # noqa: F401  (registers the conditioned representation classes)
    from wm_repr.jepa.registry import get_by_class_name
    saved = torch.load(path, map_location="cpu", weights_only=False)
    ae_cfg = OmegaConf.create(saved["hyper_parameters"]["cfg"])
    cls = get_by_class_name(ae_cfg.model.class_name)
    if assets:
        relocate_assets(ae_cfg, assets)
    ae = cls(ae_cfg)
    incompatible = ae.load_state_dict(saved["state_dict"], strict=False)
    missing_active = [key for key in incompatible.missing_keys if not key.startswith("lpips.")]
    if missing_active or incompatible.unexpected_keys:
        raise RuntimeError(f"Autoencoder checkpoint mismatch: missing={missing_active}, "
                           f"unexpected={incompatible.unexpected_keys}")
    # LPIPS is a fixed pretrained metric; model-only exports can omit its weights.
    return ae


def build_dit(cfg: DictConfig, in_channels: int) -> DiT:
    return DiT(
        in_channels=in_channels,
        patch_size=cfg.model.patch_size,
        dim=cfg.model.model_dim,
        num_layers=cfg.model.layers,
        num_heads=cfg.model.heads,
        action_dim=cfg.data.params.action_dim,
        max_frames=cfg.data.params.n_frames,
        cond_dropout=cfg.training.cond_dropout,
        rope_config={AttentionType.SPATIAL: RotaryType.PIXEL,
                     AttentionType.TEMPORAL: RotaryType.STANDARD},
        decoder_dim=cfg.model.get("decoder_dim"),
        decoder_depth=cfg.model.get("decoder_depth", 0),
        decoder_heads=cfg.model.get("decoder_heads"),
    )


def build_diffusion(cfg: DictConfig) -> Diffusion:
    return Diffusion(
        timesteps=cfg.model.timesteps,
        sampling_timesteps=cfg.model.sampling.sampling_timesteps,
        prediction_type=cfg.model.get("prediction_type", "v"),
        time_warp_gamma=cfg.model.get("time_warp_gamma", 1.0),
    )


class LitWorldModel(pl.LightningModule):
    """Train the DiT while keeping the representation and sampling EMA frozen."""

    def __init__(self, cfg, learning_rate: float, ae=None, dit=None, diffusion=None):
        super().__init__()
        cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else deepcopy(cfg)
        self.save_hyperparameters({"cfg": OmegaConf.to_container(cfg, resolve=True),
                                   "learning_rate": learning_rate})
        self.cfg = cfg
        self.learning_rate = learning_rate
        self.ema_decay = cfg.training.ema_decay
        self.ae = ae if ae is not None else load_ae(cfg)
        self.dit = dit if dit is not None else build_dit(cfg, self.ae.latent_channels)
        self.diffusion = diffusion if diffusion is not None else build_diffusion(cfg)
        self.ema_dit = deepcopy(self.dit).eval().requires_grad_(False)
        self.ae.eval().requires_grad_(False)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"params": list(self.dit.parameters()), "lr": self.learning_rate}],
            **self.cfg.training.optimizer,
        )
        warmup = self.cfg.training.get("warmup")
        if warmup is None or not warmup.get("enabled", False):
            return optimizer
        steps = int(warmup.warmup_steps)
        start = float(warmup.start_lr) / self.learning_rate if self.learning_rate else 0.0

        def multiplier(step):
            return 1.0 if step >= steps else start + (1.0 - start) * step / max(1, steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

    def _unpack_batch(self, batch):
        return einops.rearrange(batch["clip"], "b t c h w -> b t h w c"), batch["actions"]

    def training_step(self, batch, batch_idx):
        x, actions = self._unpack_batch(batch)
        with torch.no_grad():
            z, _ = self.ae.encode(x, sample=False)
        loss = self.diffusion.loss_fn(self.dit, z, actions=actions)
        self.log("train/loss", loss, on_step=True, on_epoch=False)
        return loss

    def optimizer_step(self, *args, **kwargs):
        result = super().optimizer_step(*args, **kwargs)
        update_ema(self.ema_dit, self.dit, self.ema_decay)
        return result

    @torch.no_grad()
    def _predict(self, x, actions):
        z, _ = self.ae.encode(x, sample=False)
        sampling = self.cfg.model.sampling
        predicted = self.diffusion.generate(
            self.ema_dit, z, actions,
            n_context_frames=sampling.n_context_frames, n_frames=z.shape[1],
            window_len=sampling.window_len, horizon=sampling.horizon,
            cfg_scale=sampling.cfg_scale,
        )
        frames = self.ae.decode(predicted, images=x, latents=z, conditioning="first")
        return z, predicted, frames

    def _eval_step(self, batch, prefix="val/"):
        from wm_repr.evaluation.rollout import compute_per_frame_metrics
        x, actions = self._unpack_batch(batch)
        z, predicted, frames = self._predict(x, actions)
        skip = self.cfg.model.sampling.n_context_frames if self.cfg.training.exclude_context_from_metrics else 0
        values = compute_per_frame_metrics(frames[:, skip:].float(), x[:, skip:].float(), self.lpips)
        metrics = {prefix + key: value.mean() for key, value in values.items()}
        metrics[prefix + "latent_mse"] = (predicted[:, skip:] - z[:, skip:]).square().mean()
        metrics[prefix + "loss"] = self.diffusion.loss_fn(self.ema_dit, z, actions=actions)
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        return {"targets": x, "samples": frames}

    @property
    def lpips(self):
        if not hasattr(self, "_lpips"):
            from wm_repr.vae.losses import LPIPS
            # Fixed metric initialization must not change paired diffusion noise.
            with torch.random.fork_rng(devices=[]):
                self._lpips = LPIPS().eval().to(self.device)
        return self._lpips

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, "val/")

    def train(self, mode=True):
        super().train(mode)
        self._on_toggle_mode(mode)
        return self

    def _on_toggle_mode(self, train):
        self.ae.eval()
        self.ema_dit.eval()
        if hasattr(self, "_lpips"):
            self._lpips.eval()

    def on_train_start(self):
        # A mid-epoch Lightning resume can skip on_train_epoch_start.
        self._on_toggle_mode(True)

    def on_train_epoch_start(self):
        self._on_toggle_mode(True)

    def on_validation_epoch_start(self):
        self._on_toggle_mode(False)

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get("state_dict", {})
        # The perceptual network is fixed and constructed lazily for evaluation.
        for key in list(state):
            if key.startswith(("lpips.", "_lpips.")):
                state.pop(key)


def load_checkpoint(path, device="cpu", ae_checkpoint=None, assets=None):
    """Load a trusted local checkpoint, optionally relocating its AE and pretrained assets."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(saved["hyper_parameters"]["cfg"])
    with open_dict(cfg):
        if ae_checkpoint:
            cfg.repr.ckpt = str(ae_checkpoint)
        if assets:
            relocate_assets(cfg, assets)
    model = LitWorldModel(cfg, saved["hyper_parameters"]["learning_rate"])
    model.on_load_checkpoint(saved)
    incompatible = model.load_state_dict(saved["state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint schema mismatch: {incompatible}")
    model.eval().to(device)
    if not model.ae.has_embedding_stats:
        raise ValueError("Checkpoint lacks latent normalization statistics. Compute and save them before evaluation.")
    return model
