"""WorldGym-compatible wrapper for wm-repr world models.

Adapts LitWorldModel to WorldGym's evaluation interface (reset / generate_chunk),
handling action dim padding, conditional vs unconditional decoders, latent
normalization, and precision differences.
"""

import logging

import einops
import torch

# Trigger @register decorators for all JEPA variants
import wm_repr.hierarchical as _hierarchical  # noqa: F401
import wm_repr.jepa as _jepa  # noqa: F401

from wm_repr.wm.world_model import load_checkpoint

log = logging.getLogger(__name__)


class WMReprWorldModel:

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        cfg_scale: float | None = None,
        chunk_size: int = 1,
        ae_ckpt_override: str | None = None,
        assets: str | None = None,
        sampling_steps: int | None = None,
    ):
        self.device = torch.device(device)

        model = load_checkpoint(checkpoint_path, self.device, ae_checkpoint=ae_ckpt_override, assets=assets)

        self.ae = model.ae
        self.model = model.ema_dit
        self.diffusion = model.diffusion
        if sampling_steps is not None:
            self.diffusion.sampling_timesteps = sampling_steps
        self.sampling_steps = self.diffusion.sampling_timesteps
        self.action_dim = self.model.action_dim
        self.is_conditioned = self.ae.is_conditioned
        self.cfg_scale = (
            cfg_scale
            if cfg_scale is not None
            else model.cfg.model.sampling.cfg_scale
        )
        self.chunk_size = chunk_size
        self.max_frames = self.model.max_frames

        # Mutable state, set in reset()
        self.xs = None
        self.actions = None
        self.curr_frame = 0
        self.initial_frame = None
        self.initial_latent_denorm = None

    @torch.no_grad()
    def reset(self, x: torch.Tensor):
        """Initialize with an observation frame.

        Args:
            x: (H, W, C) float tensor in [0, 1] on self.device.
        """
        x_5d = x[None, None]  # (1, 1, H, W, C)
        z, _ = self.ae.encode(x_5d)  # (1, 1, h, w, C_z) normalized
        self.xs = z
        self.actions = torch.zeros(
            (1, 1, self.action_dim), device=self.device
        )
        self.curr_frame = 1

        if self.is_conditioned:
            self.initial_frame = einops.rearrange(x, "h w c -> 1 c h w")
            z_bchw = einops.rearrange(z, "b t h w c -> (b t) c h w")
            self.initial_latent_denorm = self.ae.denormalize(z_bchw)

    @torch.no_grad()
    def generate_chunk(self, action_vec: torch.Tensor):
        """Generate next frame(s) conditioned on an action.

        Args:
            action_vec: (action_dim_in,) or (chunk_size, action_dim_in) tensor.

        Yields:
            (frame_index, decoded_frame) where decoded_frame is
            (1, 1, H, W, C) float in [0, 1].
        """
        # Pad action to model's expected dim
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)  # (1, D)
        if action_vec.shape[-1] < self.action_dim:
            pad = torch.zeros(
                *action_vec.shape[:-1],
                self.action_dim - action_vec.shape[-1],
                device=self.device,
            )
            action_vec = torch.cat([action_vec, pad], dim=-1)

        if self.actions.shape[1] != self.curr_frame:
            raise RuntimeError("The action buffer and current frame are inconsistent")
        new_actions = action_vec.to(self.actions).expand(self.chunk_size, self.action_dim)[None]
        self.actions = torch.cat((self.actions, new_actions), dim=1)
        end = self.curr_frame + self.chunk_size
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            self.xs = self.diffusion.generate(
                self.model,
                self.xs,
                self.actions,
                n_context_frames=self.curr_frame,
                n_frames=end,
                horizon=self.chunk_size,
                window_len=self.max_frames,
                cfg_scale=self.cfg_scale,
            )

        # Clean frames remain fixed during the remaining pyramid steps, so the
        # completed chunk can be decoded one frame at a time outside autocast.
        for frame_idx in range(self.curr_frame, end):
            yield frame_idx, self._decode_single_frame(frame_idx)
        self.curr_frame = end
        self._trim_buffers()

    def _trim_buffers(self):
        """Keep buffers bounded to max_frames to prevent OOM on long rollouts."""
        if self.curr_frame > self.max_frames:
            trim = self.curr_frame - self.max_frames
            self.xs = self.xs[:, trim:]
            self.actions = self.actions[:, trim:]
            self.curr_frame -= trim

    def _decode_single_frame(self, frame_idx: int) -> torch.Tensor:
        """Decode a single latent frame to pixels.

        Returns:
            (1, 1, H, W, C) float tensor in [0, 1].
        """
        z_norm = self.xs[:, frame_idx : frame_idx + 1]  # (1, 1, h, w, C_z)

        if not self.is_conditioned:
            return self.ae.decode(z_norm).float().clamp(0, 1)

        # Conditional decoder: call raw decoder with GT initial frame
        # ("first" conditioning). We bypass HierarchicalJEPAWrapper.decode()
        # because it expects full sequences; here we decode one frame at a time.
        z_bchw = einops.rearrange(z_norm, "b t h w c -> (b t) c h w")
        z_denorm = self.ae.denormalize(z_bchw)

        s_decoded = self.ae.decoder(
            self.initial_frame,
            self.initial_latent_denorm,
            z_denorm,
        )
        return einops.rearrange(s_decoded, "b c h w -> b 1 h w c").float().clamp(0, 1)
