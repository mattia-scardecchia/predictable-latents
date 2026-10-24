"""Evaluate checkpoints on fixed Bridge or AutoEval clips."""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from wm_repr.utils.provenance import source_revision
from .data import make_dataset, make_loader, random_records, read_manifest
from .rollout import run_inference, compute_stats
from ..wm.world_model import load_checkpoint


def save(output, arrays, settings, distribution=None):
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "raw.npz", **arrays)
    result = compute_stats(arrays)
    result["eval_params"] = settings
    if settings.get("mode") == "drift":
        result["late_window"] = {
            "prediction_steps": [51, 59],
            "mean": {key: float(np.mean(np.asarray(value["mean"])[50:59]))
                     for key, value in result["per_frame"].items()},
            "counts": next(iter(result["per_frame"].values()))["n"][50:59],
        }
    if distribution:
        result["distribution"] = distribution
    (output / "stats.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=["id", "ood", "steps", "drift", "controls", "forecast"], default="id")
    parser.add_argument("--split", choices=["id", "ood"], default="id", help="Dataset for controls or forecasts")
    parser.add_argument("--subset", help="Name of the dataset subdirectory")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--ae-checkpoint", type=Path)
    parser.add_argument("--assets", type=Path, help="Asset root containing flux1-schnell or rae-dinov2-wreg-base")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--permutation-seed", type=int, default=0,
                        help="Fixed action donor assignment for --mode controls")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-counts", type=int, nargs="+", default=[2, 4, 6, 8, 10],
                        help="DDIM step counts for --mode steps")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--distribution-metrics", action="store_true")
    parser.add_argument("--max-clips", type=int,
                        help="Evaluate only N clips: the first N of --manifest, or N random clips "
                             "without one (quick test; not the paper protocol)")
    args = parser.parse_args()
    if args.manifest is None and args.max_clips is None:
        parser.error("Supply --manifest for the paper evaluation, or --max-clips for a quick test")
    if args.mode == "drift" and args.manifest is None:
        parser.error("Long rollouts require the fixed drift manifest")
    split = "ood" if args.mode == "ood" else args.split
    # The paper's evaluations ran float32 products in TF32 and decoded the whole batch in
    # distribution, four clips out of distribution and one clip per call in long rollouts;
    # with the same settings the per-clip values are reproduced exactly.
    torch.set_float32_matmul_precision("high")
    decode_chunk = 1 if args.mode == "drift" else 4 if split == "ood" else None
    if args.assets:
        os.environ.setdefault("WM_LPIPS_CACHE_ROOT", str(args.assets.resolve() / "lpips"))
        os.environ.setdefault("TORCH_HOME", str(args.assets.resolve() / "torch"))
        os.environ.setdefault("WM_I3D_PATH", str(args.assets.resolve() / "i3d_torchscript.pt"))
    model = load_checkpoint(args.checkpoint, args.device, args.ae_checkpoint, args.assets)
    dataset = make_dataset(model.cfg, args.data_root, split, args.subset)
    if args.manifest:
        records, digest = read_manifest(args.manifest, args.manifest_sha256)
    else:
        records, digest = random_records(dataset, args.max_clips), None
    if args.max_clips:
        records = records[:args.max_clips]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    (args.output / "provenance.json").write_text(json.dumps({
        "commit": source_revision(), "command": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "config": OmegaConf.to_container(model.cfg, resolve=True),
    }, indent=2) + "\n")
    seeds = args.seeds or ([42, 43, 44] if args.mode in {"id", "ood", "forecast"} else [42])
    settings = {"mode": args.mode, "split": split, "manifest_sha256": digest,
                "n_samples": len(records), "precision": args.precision,
                "checkpoint": str(args.checkpoint), "max_clips": args.max_clips}
    common = dict(device=args.device, precision=args.precision)
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        output = args.output / f"seed{seed}"
        params = dict(settings, seed=seed)
        if args.mode == "drift":
            by_length = {}
            for row in records:
                if row["start_frame"] != 0:
                    raise ValueError("Drift clips must start at frame zero")
                by_length.setdefault(min(60, row["video_length"]), []).append(row)
            arrays, ordered = {}, []
            for length, rows in sorted(by_length.items()):
                loader = make_loader(dataset, rows, args.batch_size, args.workers, length)
                values, _ = run_inference(model, loader, horizon=8, n_steps=args.steps,
                                          decode_chunk=decode_chunk, **common)
                for key in ("mse", "ssim", "lpips", "latent_mse"):
                    padded = np.full((len(rows), 59), np.nan, dtype=np.float32)
                    padded[:, :values[key].shape[1]] = values[key]
                    arrays.setdefault(key, []).append(padded)
                ordered.extend(rows)
            arrays = {key: np.concatenate(value) for key, value in arrays.items()}
            arrays["video_paths"] = np.asarray([row["video_path"] for row in ordered])
            arrays["lengths"] = np.asarray([row["video_length"] for row in ordered])
            save(output, arrays, params)
            continue
        # Forecasts score single transitions; every other mode keeps the manifest's clip length.
        frames = 2 if args.mode == "forecast" else None
        loader = make_loader(dataset, records, args.batch_size, args.workers, frames)
        if args.mode == "forecast":
            from .forecast import run_forecastability
            arrays, result = run_forecastability(model, loader, seed=seed, **common)
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / "raw.npz", **arrays)
            (output / "stats.json").write_text(json.dumps(dict(result, eval_params=params), indent=2) + "\n")
            continue
        if args.mode == "controls":
            from .controls import run_controls
            values, donors = run_controls(model, loader, permutation_seed=args.permutation_seed,
                                         within_task=split == "ood",
                                         horizon=1 if split == "ood" else 8, n_steps=args.steps,
                                         decode_chunk=decode_chunk, **common)
            for name, arrays in values.items():
                arrays["donor_index"] = donors
                arrays["dataset_index"] = np.asarray([row.get("dataset_index", i) for i, row in enumerate(records)])
                save(output / name, arrays, dict(params, permutation_seed=args.permutation_seed))
            continue
        steps = args.step_counts if args.mode == "steps" else [args.steps]
        for count in steps:
            # Match diffusion noise across sampling step counts and across models.
            torch.manual_seed(seed)
            values, dist = run_inference(model, loader, horizon=1 if split == "ood" or args.mode == "steps" else 8,
                                        n_steps=count, compute_dist_metrics=args.distribution_metrics,
                                        decode_chunk=decode_chunk, **common)
            values["dataset_index"] = np.asarray([row.get("dataset_index", i) for i, row in enumerate(records)])
            save(output / f"steps{count}", values, dict(params, n_steps=count), dist)


if __name__ == "__main__":
    main()
