"""Recorded-action rollouts with the paper's precision, context and metric conventions."""

from __future__ import annotations
import contextlib
from collections import defaultdict
import numpy as np
import torch
import einops
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_func


@torch.no_grad()
def compute_per_frame_metrics(
    samples: torch.Tensor,  # (B, T, H, W, C) float [0, 1]
    x: torch.Tensor,
    lpips_fn: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Returns a dict of (B, T) tensors: mse, ssim, lpips."""
    B, T, H, W, C = samples.shape
    device = samples.device
    mse_bt = torch.zeros(B, T, device=device)
    ssim_bt = torch.zeros(B, T, device=device)
    lpips_bt = torch.zeros(B, T, device=device)

    for t in range(T):
        s_t = einops.rearrange(samples[:, t], "b h w c -> b c h w")
        x_t = einops.rearrange(x[:, t], "b h w c -> b c h w")

        mse_bt[:, t] = ((s_t - x_t) ** 2).mean(dim=(1, 2, 3))
        ssim_bt[:, t] = ssim_func(s_t, x_t, data_range=1.0, reduction="none")
        lp = lpips_fn(2.0 * s_t - 1.0, 2.0 * x_t - 1.0, "none")  # (B, 1, 1, 1)
        lpips_bt[:, t] = lp.view(B)

    return {"mse": mse_bt, "ssim": ssim_bt, "lpips": lpips_bt}


@torch.no_grad()
def compute_latent_mse_per_frame(
    pred_latents: torch.Tensor,  # (B, T, H_z, W_z, C_z)
    gt_latents: torch.Tensor,
) -> torch.Tensor:  # (B, T)
    return ((pred_latents - gt_latents) ** 2).mean(dim=(-1, -2, -3))


def autocast_context(device, precision="bf16-mixed"):
    dtype = {"bf16-mixed": torch.bfloat16, "bf16": torch.bfloat16,
             "16-mixed": torch.float16, "16": torch.float16}.get(str(precision))
    if dtype is None or torch.device(device).type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


class DistributionMetrics:
    def __init__(self, device):
        from .video_metrics import FID, FVD, I3D
        self.i3d = I3D().eval().to(device)
        self.fid = FID().to(device)
        self.fvd = FVD().to(device)

    def update(self, generated, targets):
        from .video_metrics import extract_i3d_features
        generated = einops.rearrange(generated, "b t h w c -> b t c h w")
        targets = einops.rearrange(targets, "b t h w c -> b t c h w")
        fake = extract_i3d_features(self.i3d, generated)
        real = extract_i3d_features(self.i3d, targets)
        self.fvd.update(fake, real)
        self.fid.update(einops.rearrange(generated, "b t c h w -> (b t) c h w"),
                        einops.rearrange(targets, "b t c h w -> (b t) c h w"))

    def compute(self):
        return {"fid": float(self.fid.compute()), "fvd": float(self.fvd.compute())}


def decode(model, latents, images, encoded, chunk=1):
    """Decode ``chunk`` clips per call, or the whole batch in one call when ``chunk`` is None."""
    if chunk is None:
        return model.ae.decode(latents, images=images, latents=encoded, conditioning="first")
    return torch.cat([model.ae.decode(
        latents[i:i + chunk], images=images[i:i + chunk],
        latents=encoded[i:i + chunk], conditioning="first")
        for i in range(0, len(images), chunk)])


@torch.no_grad()
def run_inference(model, loader, *, horizon=8, n_steps=10, cfg_scale=3.0,
                  device=None, precision="bf16-mixed", compute_dist_metrics=False,
                  lpips_fn=None, actions_override=None, ceiling=False, decode_chunk=1):
    """Return per-clip/per-frame arrays and optional distribution metrics.

    Every array's second axis contains generated frames only. Decoding always uses
    the initial real frame. An action override has shape (clips, frames, actions).
    """
    device = device or model.device
    model.eval()
    model.diffusion.sampling_timesteps = int(n_steps)
    if lpips_fn is None:
        lpips_fn = model.lpips
    dist = DistributionMetrics(device) if compute_dist_metrics else None
    n_ctx = int(model.cfg.model.sampling.n_context_frames)
    store = defaultdict(list)
    tasks, stems = [], []
    offset = 0
    for batch in loader:
        x, actions = model._unpack_batch(batch)
        x = x.to(device)
        actions = actions.to(device)
        if actions_override is not None:
            actions = torch.as_tensor(actions_override[offset:offset + len(x)], device=device,
                                      dtype=actions.dtype)
        offset += len(x)
        with autocast_context(device, precision):
            z, _ = model.ae.encode(x, sample=False)
            predicted = z if ceiling else model.diffusion.generate(
                model.ema_dit, z, actions, n_context_frames=n_ctx, n_frames=z.shape[1],
                window_len=model.cfg.model.sampling.window_len, horizon=horizon,
                cfg_scale=cfg_scale)
        z, predicted = z.float(), predicted.float()
        frames = decode(model, predicted, x, z, decode_chunk)
        values = compute_per_frame_metrics(frames[:, n_ctx:], x[:, n_ctx:], lpips_fn)
        values["latent_mse"] = compute_latent_mse_per_frame(predicted[:, n_ctx:], z[:, n_ctx:])
        for key, value in values.items():
            store[key].append(value.float().cpu().numpy())
        store["actions_transformed"].append(actions.cpu().numpy())
        tasks.extend(batch.get("instruction") or [""] * len(x))
        stems.extend(batch.get("stem") or [""] * len(x))
        if dist is not None:
            dist.update(frames[:, n_ctx:], x[:, n_ctx:])
    if not store:
        raise ValueError("Evaluation loader produced no batches.")
    result = {key: np.concatenate(value) for key, value in store.items()}
    result["tasks"] = np.asarray([task or "" for task in tasks])
    result["trajectory_stem"] = np.asarray(stems)
    return result, dist.compute() if dist else None


def compute_stats(arrays):
    """Aggregate clips first; retain per-step counts for variable-length rollouts."""
    overall, per_frame = {}, {}
    for key in ("mse", "ssim", "lpips", "latent_mse"):
        if key not in arrays:
            continue
        a = arrays[key]
        valid = np.isfinite(a)
        count = valid.sum(axis=0)
        mean = np.divide(np.nansum(a, axis=0), count,
                         out=np.full(a.shape[1], np.nan), where=count > 0)
        variance = np.divide(np.nansum((a - mean) ** 2, axis=0), count,
                             out=np.full(a.shape[1], np.nan), where=count > 0)
        per_frame[key] = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
                          "sem": np.sqrt(variance / np.maximum(count, 1)).tolist(),
                          "n": count.tolist()}
        episode = np.nanmean(a, axis=1)
        overall[key] = {"mean": float(np.mean(episode)), "std": float(np.std(episode)),
                        "n_clips": len(episode)}
    result = {"overall": overall, "per_frame": per_frame}
    if "tasks" in arrays and any(arrays["tasks"]):
        result["per_task"] = {
            str(task): {key: float(np.nanmean(arrays[key][arrays["tasks"] == task]))
                       for key in overall}
            for task in sorted(set(arrays["tasks"]))}
        result["equal_task_mean"] = {
            key: float(np.mean([task[key] for task in result["per_task"].values()]))
            for key in overall}
    return result
