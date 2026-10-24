# The DDIM sampling step adapts WorldGym's (https://github.com/world-model-eval/world-model-eval).
import torch
import torch.nn as nn
from types import SimpleNamespace

from wm_repr.third_party.worldgym.diffusion import Diffusion as _WorldGymDiffusion


class Diffusion(_WorldGymDiffusion):
    """WorldGym diffusion with a velocity loss, a velocity or clean-latent prediction, time
    warping and guidance."""

    # The conversions out of a clean-latent prediction divide by sigma, clipped from below.
    sigma_denom_clip = 0.05

    def __init__(
        self,
        timesteps: int = 1_000,
        sampling_timesteps: int = 10,
        prediction_type: str = "v",
        time_warp_gamma: float = 1.0,
    ) -> None:
        valid_types = {"x", "v"}
        if prediction_type not in valid_types:
            raise ValueError(
                f"prediction_type must be one of {valid_types}, got {prediction_type!r}"
            )
        if time_warp_gamma <= 0:
            raise ValueError(
                f"time_warp_gamma must be positive, got {time_warp_gamma!r}"
            )
        super().__init__(timesteps=timesteps, sampling_timesteps=sampling_timesteps)
        self.prediction_type = prediction_type
        self.time_warp_gamma = float(time_warp_gamma)

    def sigmoid_beta_schedule(
        self, timesteps: int, start: float = -3, end: float = 3, tau: float = 1
    ) -> torch.Tensor:
        # The adjacent survival-probability ratio cancels the normalization of
        # the sigmoid schedule (https://arxiv.org/abs/2212.11972). Keeping its
        # arithmetic in float64 avoids rounding before the cumulative product.
        fraction = torch.arange(timesteps + 1, dtype=torch.float64) / timesteps
        sigmoid = torch.sigmoid((start + (end - start) * fraction) / tau)
        survival = torch.sigmoid(torch.tensor(end / tau)) - sigmoid
        return (1 - survival[1:] / survival[:-1]).clamp(0, 0.999)

    def _expand_to_x_shape(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (B, T), x: (B, T, ...)
        return t.view(*t.shape, *((1,) * (x.ndim - 2)))

    def _uses_time_warp(self) -> bool:
        return self.time_warp_gamma != 1.0

    def _warp_fraction(self, u: torch.Tensor) -> torch.Tensor:
        """
        Monotone RAE-style warp on normalized time u in [0, 1]:
            u' = gamma * u / (1 + (gamma - 1) * u)
        gamma = 1 leaves normalized time unchanged.
        """
        if not self._uses_time_warp():
            return u
        gamma = self.time_warp_gamma
        return (gamma * u) / (1.0 + (gamma - 1.0) * u)

    def _warp_indices(self, t: torch.Tensor) -> torch.Tensor:
        """
        Apply the time warp to discrete timestep indices by mapping indices to
        normalized time in [0, 1], warping, and rounding back to integer indices.
        """
        if not self._uses_time_warp():
            return t

        if self.timesteps <= 1:
            return t

        u = t.to(dtype=torch.float32) / float(self.timesteps - 1)
        u = self._warp_fraction(u)
        t_warped = torch.round(u * float(self.timesteps - 1)).long()
        return t_warped.clamp_(0, self.timesteps - 1)

    def _sampling_noise_steps(self, device: torch.device) -> torch.Tensor:
        """
        Returns the discrete DDIM grid in diffusion-step indices.
        When time_warp_gamma == 1, the grid is uniformly spaced.
        """
        base_steps = torch.linspace(
            -1,
            self.timesteps - 1,
            steps=self.sampling_timesteps + 1,
            device=device,
            dtype=torch.long,
        )
        if not self._uses_time_warp():
            return base_steps

        warped_steps = base_steps.clone()
        nonnegative_mask = warped_steps >= 0
        warped_steps[nonnegative_mask] = self._warp_indices(
            warped_steps[nonnegative_mask]
        )
        return warped_steps

    def _get_alpha_sigma(
        self,
        t: torch.Tensor,
        x_like: torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns alpha_t and sigma_t broadcast to x_like's shape, where:
            x_t = alpha_t * x_0 + sigma_t * eps
            alpha_t = sqrt(alpha_bar_t), sigma_t = sqrt(1 - alpha_bar_t)
        """
        alpha_bar = (
            self.alphas_cumprod[t.reshape(-1)]
            .view(*t.shape, *((1,) * (x_like.ndim - 2)))
            .to(device=x_like.device, dtype=dtype)
        )
        alpha_bar = alpha_bar.clamp(0.0, 1.0)
        alpha = alpha_bar.sqrt()
        sigma = (1.0 - alpha_bar).clamp(min=0.0).sqrt()
        return alpha, sigma

    def _velocity_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        The velocity target under the current VP parameterization:
            x_t = alpha * x0 + sigma * noise
            v   = alpha * noise - sigma * x0
        """
        return alpha * noise.float() - sigma * x0.float()

    def _convert_model_output(
        self,
        model_out: torch.Tensor,
        x_t: torch.Tensor,
        alpha: torch.Tensor,
        sigma: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Given the model's direct output in self.prediction_type space, convert it to
        predictions in all three spaces: x, eps, v.
        The conversion algebra is carried out in float32, and the divisive
        conversions use aggressive denominator clipping.
        """
        x_t = x_t.float()
        model_out = model_out.float()
        alpha = alpha.float()
        sigma = sigma.float()

        if self.prediction_type == "x":
            sigma_d = sigma.clamp_min(self.sigma_denom_clip)
            pred_x = model_out
            pred_eps = (x_t - alpha * pred_x) / sigma_d
            pred_v = (alpha * x_t - pred_x) / sigma_d

        elif self.prediction_type == "v":
            pred_v = model_out
            pred_x = alpha * x_t - sigma * pred_v
            pred_eps = sigma * x_t + alpha * pred_v

        else:
            raise RuntimeError(f"Unknown prediction_type: {self.prediction_type}")

        return {"x": pred_x, "eps": pred_eps, "v": pred_v}

    def q_sample(
        self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        alpha, sigma = self._get_alpha_sigma(t, x, dtype=x.dtype)
        return alpha * x + sigma * noise

    def loss_fn(
        self, model: nn.Module, x: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Diffusion-forcing loss: an independent noise level per frame, with the
        squared error of the velocity averaged over all frames."""
        B, T = x.shape[:2]
        t = self._warp_indices(
            torch.randint(0, self.timesteps, (B, T), device=x.device, dtype=torch.long)
        )
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t, noise)

        model_out = model(x_t, t, actions)
        alpha, sigma = self._get_alpha_sigma(t, x_t, dtype=torch.float32)
        preds = self._convert_model_output(model_out, x_t, alpha, sigma)
        target = self._velocity_target(x, noise, alpha, sigma)

        elem = (preds["v"] - target).pow(2)
        return elem.sum() / elem.new_tensor(elem.numel())

    def ddim_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        t_idx: torch.Tensor,
        t_next_idx: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        # Derived from
        # https://github.com/buoyancy99/diffusion-forcing/blob/475e0bcab87545e48b24b39fb46a81fe59d80594/algorithms/diffusion_forcing/models/diffusion.py#L383
        x = x.contiguous()
        actions = actions.contiguous()
        t_idx = t_idx.contiguous()
        t_next_idx = t_next_idx.contiguous()

        B, T = x.shape[:2]

        sampling_noise_steps = self._sampling_noise_steps(x.device)
        t = sampling_noise_steps[t_idx]
        t_next = sampling_noise_steps[t_next_idx]

        clipped_t = torch.where(
            t < 0,
            torch.full_like(t, self.stabilization_level - 1, dtype=torch.long),
            t,
        )

        orig_x = x.clone().detach()
        scaled_context = self.q_sample(x, clipped_t, torch.zeros_like(x))
        x = torch.where(self._expand_to_x_shape(t, x) < 0, scaled_context, x)

        alpha, sigma = self._get_alpha_sigma(clipped_t, x, dtype=torch.float32)

        alpha_next = (
            torch.where(
                t_next < 0,
                torch.ones_like(t_next, dtype=torch.float32, device=x.device),
                self.alphas_cumprod[t_next.reshape(-1)]
                .view(B, T)
                .to(dtype=torch.float32, device=x.device),
            )
            .view(B, T, 1, 1, 1)
            .sqrt()
        )
        sigma_next = (1 - alpha_next.square()).sqrt()

        model_out_cond = model(x, clipped_t, actions)
        if cfg_scale != 1.0:
            model_out_null = model(x, clipped_t, model.get_null_cond(actions))
            model_out = model_out_null + cfg_scale * (model_out_cond - model_out_null)
        else:
            model_out = model_out_cond

        preds = self._convert_model_output(model_out, x, alpha, sigma)
        x_start = preds["x"]
        pred_noise = preds["eps"]

        # DDIM update
        x_pred = alpha_next * x_start + sigma_next * pred_noise

        x_pred = torch.where(
            (t == t_next).view(B, T, 1, 1, 1),
            orig_x,
            x_pred,
        )
        return x_pred

    def generate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        actions: torch.Tensor,
        n_context_frames: int = 1,
        n_frames: int = 1,
        horizon: int = 1,
        window_len: int | None = None,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        def step(model, latents, conditioning, current, following, cfg=1.0):
            return self.ddim_sample_step(
                model,
                latents,
                conditioning,
                current.to(latents.device),
                following.to(latents.device),
                cfg_scale=cfg,
            )

        # WorldGym constructs schedule indices on CPU and uses self.device for
        # new noise. A call-local view supports device moves without stale state.
        view = SimpleNamespace(
            device=x.device,
            generate_pyramid_scheduling_matrix=self.generate_pyramid_scheduling_matrix,
            ddim_sample_step=step,
        )
        return _WorldGymDiffusion.generate(
            view,
            model,
            x,
            actions,
            n_context_frames=n_context_frames,
            n_frames=n_frames,
            horizon=horizon,
            window_len=window_len,
            cfg=cfg_scale,
        )
