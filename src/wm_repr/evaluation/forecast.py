"""One-step latent forecasts."""
from __future__ import annotations
import numpy as np
import torch
import einops
from .rollout import autocast_context


@torch.no_grad()
def predict_next_latent(
    diffusion,
    model,
    context: torch.Tensor,  # (N, 1, H, W, C) context-frame latents
    actions: torch.Tensor,  # (N, 2, action_dim) the context action and the target's own action
    noise: torch.Tensor,    # (N, 1, H, W, C) initial state of the target frame
    cfg_scale: float = 1.0,
) -> torch.Tensor:          # (N, 1, H, W, C) predicted next-frame latents
    """Sample the next latent from one context frame with a fixed initial noise.

    The special case n_context_frames=1, n_frames=2, horizon=1 of
    ``wm.diffusion.Diffusion.generate``, with the noise supplied by the caller. The context frame
    stays at timestep index 0, which ``_sampling_noise_steps`` maps to the clean sentinel -1.
    """
    n = context.shape[0]
    device = context.device
    x_pred = torch.cat([context, noise], dim=1)  # (N, 2, H, W, C)
    scheduling_matrix = diffusion.generate_pyramid_scheduling_matrix(1).to(device)
    ctx_zeros = torch.zeros((n, 1), dtype=torch.long, device=device)
    for m in range(scheduling_matrix.shape[0] - 1):
        t = einops.repeat(scheduling_matrix[m], "t -> b t", b=n)
        t_next = einops.repeat(scheduling_matrix[m + 1], "t -> b t", b=n)
        t = torch.cat([ctx_zeros, t], dim=1)          # (N, 2)
        t_next = torch.cat([ctx_zeros, t_next], dim=1)
        x_pred = diffusion.ddim_sample_step(model, x_pred, actions, t, t_next, cfg_scale)
    return x_pred[:, -1:]


@torch.no_grad()
def run_forecastability(model, loader, *, device=None, precision="bf16-mixed", seed=42,
                        max_pairs=256):
    device = device or model.device
    model.eval()
    model.diffusion.sampling_timesteps = 10
    errors, target_sum, target_sq, count = [], 0.0, 0.0, 0
    tasks = []
    for batch in loader:
        x, actions = model._unpack_batch(batch)
        x, actions = x.to(device), actions.to(device)
        with autocast_context(device, precision):
            z, _ = model.ae.encode(x, sample=False)
            for row in range(len(z)):
                if len(errors) == max_pairs:
                    break
                context = z[row:row + 1, :1]
                target = z[row:row + 1, 1:2]
                generator = torch.Generator(device=device).manual_seed(seed + len(errors))
                noise = torch.randn(context.shape, device=device, generator=generator, dtype=context.dtype)
                predicted = predict_next_latent(model.diffusion, model.ema_dit, context,
                                                actions[row:row + 1, :2], noise, cfg_scale=1.0)
                errors.append(float((predicted.float() - target.float()).square().mean()))
                labels = batch.get("instruction") or [""] * len(z)
                tasks.append(labels[row] or "")
                flat = target.float()
                target_sum += float(flat.sum())
                target_sq += float(flat.square().sum())
                count += flat.numel()
        if len(errors) == max_pairs:
            break
    if not errors:
        raise ValueError("No transitions available for forecastability")
    variance = max(target_sq / count - (target_sum / count) ** 2, 0.0)
    normalized = np.asarray(errors) / max(variance, 1e-12)
    result = {
        "pairs_evaluated": len(errors), "target_latent_variance": variance,
        "latent_forecast_error": float(np.mean(errors)) / max(variance, 1e-12),
        "n_steps": 10, "cfg_scale": 1.0, "noise_seed": seed}
    if any(tasks):
        labels = np.asarray(tasks)
        result["per_task"] = {str(task): float(normalized[labels == task].mean()) for task in sorted(set(tasks))}
        result["equal_task_mean"] = float(np.mean(list(result["per_task"].values())))
    return {"mse": np.asarray(errors), "normalized_mse": normalized, "tasks": np.asarray(tasks)}, result

