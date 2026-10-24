"""Fixed-pair autoencoder reconstruction, decoder interventions, and latent dumps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
import csv
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from wm_repr.analysis.latent_bases import turning_angles_tensor
from wm_repr.data.dataset import OpenXMP4FullDataset, OpenXMP4TCDataset
from wm_repr.hierarchical.base import BaseHierarchicalJEPA
from wm_repr.jepa.predictor import InverseDynamicsHead
from wm_repr.utils.provenance import source_revision
from wm_repr.wm.world_model import load_autoencoder_checkpoint
from .data import make_loader, read_manifest
from .rollout import autocast_context


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _per_example_metrics(model, reconstruction, target):
    pixel_l1 = (reconstruction - target).abs().flatten(1).mean(1)
    lpips = model.lpips(
        reconstruction * 2.0 - 1.0,
        target * 2.0 - 1.0,
        reduction="none",
    ).flatten(1).mean(1)
    return pixel_l1, lpips, pixel_l1 + lpips


def _aggregate(rows: list[dict]) -> dict:
    by_mode = defaultdict(lambda: defaultdict(list))
    metrics = (
        "pixel_l1",
        "lpips",
        "reconstruction_loss",
        "latent_prediction_l1",
        "normalized_latent_prediction_l1",
        "target_latent_mean_abs",
    )
    for row in rows:
        for metric in metrics:
            if metric in row:
                by_mode[row["mode"]][metric].append(float(row[metric]))

    summary = {}
    for mode, values in by_mode.items():
        summary[mode] = {"examples": len(values["pixel_l1"])}
        for metric, samples in values.items():
            tensor = torch.tensor(samples, dtype=torch.float64)
            summary[mode][metric] = {
                "mean": tensor.mean().item(),
                "sample_std": tensor.std(unbiased=True).item()
                if len(tensor) > 1
                else 0.0,
                "naive_example_se": tensor.std(unbiased=True).item() / len(tensor) ** 0.5
                if len(tensor) > 1
                else 0.0,
            }
    return summary


class EvaluationPairs(Dataset):
    """Replay saved pair identities without depending on loader worker order."""

    def __init__(self, dataset, records):
        self.dataset, self.records = dataset, records
        if len({row["video_path"] for row in records}) != len(records):
            raise ValueError("The fixed-pair manifest must contain each video exactly once")
        paths = {str(p.relative_to(dataset.root_path)): i for i, p in enumerate(dataset.video_paths)}
        self.indices = []
        for row in records:
            index = paths[row["video_path"]]
            if int(dataset.video_lengths[index]) != int(row["video_length"]):
                raise ValueError(f"Video length differs from manifest: {row['video_path']}")
            current, future = int(row["current_index"]), int(row["future_index"])
            if not 0 <= current < future < dataset.video_lengths[index] or future - current != 1:
                raise ValueError("The fixed-pair reconstruction protocol requires an exact one-frame gap")
            self.indices.append(index)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        dataset = copy(self.dataset)
        dataset._sample_indices = lambda *_: (int(row["current_index"]), int(row["future_index"]), None)
        return dataset[self.indices[index]]


def _dataset_options(model, data_root, split="test"):
    params = model.cfg.data.params
    return dict(root_path=str(data_root), split=split, input_h=int(params.input_h),
                input_w=int(params.input_w), action_dim=int(params.action_dim),
                subset_names="bridge_v2", seed=42)


@torch.inference_mode()
def evaluate_pairs(model, loader, records, *, interventions=False, device="cpu",
                   precision="32-true", shuffle_seed=42):
    """Evaluate batch-local context shuffling and per-example metrics."""
    model.eval()
    rows, offset = [], 0
    generator = torch.Generator().manual_seed(shuffle_seed)
    for batch in loader:
        current, future = batch["current"].to(device), batch["future"].to(device)
        actions = {key: value.to(device) for key, value in batch["actions"].items()}
        with autocast_context(device, precision):
            z, _ = model._encode(current, kl=False)
            z_future, _ = model._encode(future, kl=False)
            auxiliary = {}
            if not isinstance(model.predictor, InverseDynamicsHead):
                prediction_l1 = (model.predictor(z, actions) - z_future).abs().flatten(1).mean(1)
                magnitude = z_future.abs().flatten(1).mean(1)
                auxiliary = {"latent_prediction_l1": prediction_l1,
                             "target_latent_mean_abs": magnitude,
                             "normalized_latent_prediction_l1": prediction_l1 / magnitude.clamp_min(1e-12)}
            permutation = None
            if isinstance(model, BaseHierarchicalJEPA):
                trained_mode = "conditioned"
                if interventions:
                    permutation = torch.randperm(len(current), generator=generator).to(device)
                    reconstructions = {
                        "conditioned": model._decode(current, z, z_future, conditioned=True),
                        "conditioning_disabled": model._decode(current, z, z_future, conditioned=False),
                        "shuffled_past": model._decode(current[permutation], z[permutation], z_future, conditioned=True),
                    }
                else:
                    reconstructions = {trained_mode: model._decode(current, z, z_future)}
                target = future
            else:
                if interventions:
                    raise ValueError("Decoder interventions require a conditioned autoencoder")
                reconstructions, target, trained_mode = {"ordinary": model.decoder(z)}, current, "ordinary"
            for mode, reconstruction in reconstructions.items():
                pixel, lpips, reconstruction_loss = _per_example_metrics(model, reconstruction, target)
                for i in range(len(current)):
                    record = records[offset + i]
                    donor = offset + int(permutation[i]) if mode == "shuffled_past" else offset + i
                    rows.append(dict(record, mode=mode, is_trained_mode=mode == trained_mode,
                                     donor_index=donor, donor_video_path=records[donor]["video_path"],
                                     pixel_l1=float(pixel[i]), lpips=float(lpips[i]),
                                     reconstruction_loss=float(reconstruction_loss[i]),
                                     **{key: float(value[i]) for key, value in auxiliary.items()}))
        offset += len(current)
    if offset != len(records) or not rows:
        raise ValueError(f"Evaluated {offset} pairs for a {len(records)}-pair manifest")
    return rows


def run_reconstruction(model, args):
    dataset = OpenXMP4TCDataset(**_dataset_options(model, args.data_root),
                               sample_other=False,
                               fixed_eval_pairs=True)
    records = read_manifest(args.manifest, args.manifest_sha256)[0] if args.manifest else list(dataset.fixed_pair_manifest())
    if args.max_examples is not None:
        records = records[:args.max_examples]
    elif len(records) != 6872:
        raise ValueError(f"Paper reconstruction evaluation requires all 6872 fixed pairs; found {len(records)}. "
                         "Use --max-examples for a quick test (not the paper protocol).")
    _write_manifest(args.output, records)
    selected = EvaluationPairs(dataset, records)
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.workers, collate_fn=dataset.get_collate_fn(),
                        persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(42))
    rows = evaluate_pairs(model, loader, records, interventions=args.mode == "interventions",
                          device=args.device, precision=args.precision, shuffle_seed=args.seed)
    with (args.output / "per_example.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shuffled = [row for row in rows if row["mode"] == "shuffled_past"]
    return dict(examples=len(records), aggregates=_aggregate(rows),
                shuffle_protocol="seeded permutation within each evaluation batch; fixed points retained",
                shuffle_batch_size=args.batch_size,
                shuffled_fixed_points=sum(row["video_path"] == row["donor_video_path"] for row in shuffled))


def _write_manifest(output, records):
    (output / "manifest.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))


def geometry_records(dataset):
    """First 100 valid videos with reproducible seed-42 random frame offsets."""
    return list(dataset.fixed_clip_manifest())[:100]


@torch.inference_mode()
def dump_autoencoder_latents(model, loader, records, output, *, wrapper, device="cpu", precision="32-true"):
    """Write the arrays read by `wm_repr.analysis.predictability` and `wm_repr.analysis.structure`."""
    output, total, arrays, offset = Path(output), len(records), None, 0
    for batch in loader:
        clips = batch["clip"].to(device)
        b, t = clips.shape[:2]
        with autocast_context(device, precision):
            raw = model.encoder(einops.rearrange(clips, "b t c h w -> (b t) c h w"))
            normalized = wrapper.normalize(raw) if wrapper is not None else raw
        raw = einops.rearrange(raw.float(), "(b t) c h w -> b t h w c", b=b, t=t)
        normalized = einops.rearrange(normalized.float(), "(b t) c h w -> b t h w c", b=b, t=t)
        actions = batch["actions"].numpy()
        if arrays is None:
            layouts = {"latents": (np.float16, (total, t, *raw.shape[2:])),
                       "raw_latents": (np.float16, (total, t, *raw.shape[2:])),
                       "actions": (np.float32, (total, t, actions.shape[-1])),
                       "dataset_index": (np.int64, (total,)), "start_frame": (np.int64, (total,)),
                       "pixel_turning_angles": (np.float32, (total, t - 2))}
            arrays = {key: np.lib.format.open_memmap(output / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
                      for key, (dtype, shape) in layouts.items()}
        part = slice(offset, offset + b)
        arrays["latents"][part] = normalized.cpu().numpy().astype(np.float16)
        arrays["raw_latents"][part] = raw.cpu().numpy().astype(np.float16)
        arrays["actions"][part] = actions
        arrays["pixel_turning_angles"][part] = turning_angles_tensor(clips.flatten(2)).cpu().numpy()
        arrays["dataset_index"][part] = [r["dataset_index"] for r in records[part]]
        arrays["start_frame"][part] = [r["start_frame"] for r in records[part]]
        offset += b
        (output / "dump_record.json").write_text(json.dumps({"status": "partial", "clips_done": offset}))
    if arrays is None or offset != total:
        raise ValueError("The latent dump did not cover its complete manifest")
    for value in arrays.values():
        value.flush()
    record = dict(status="complete", clips_done=offset, n_clips=total, n_frames=t,
                  n_transitions=offset * (t - 1), latent_shape=list(raw.shape[2:]),
                  latent_layout="clip, frame, height, width, channel",
                  latent_normalization=wrapper.latent_normalization if wrapper else "none",
                  raw_latents_normalization="none", precision=precision, label=output.name)
    (output / "dump_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run_dump(model, args):
    common = dict(n_frames=10)
    geometry = args.mode == "geometry"
    dataset = OpenXMP4FullDataset(**_dataset_options(model, args.data_root), **common, fixed_eval_clips=True,
                                  max_videos=1024 if geometry else None)
    records = read_manifest(args.manifest, args.manifest_sha256)[0] if args.manifest else (
        geometry_records(dataset) if geometry else list(dataset.fixed_clip_manifest()))
    if args.max_examples is not None:
        records = records[:args.max_examples]
    elif len(records) != (100 if geometry else 2000):
        raise ValueError(f"This paper analysis requires {100 if geometry else 2000} clips; found {len(records)}")
    _write_manifest(args.output, records)
    loader = make_loader(dataset, records, args.batch_size, args.workers)
    wrapper = None
    normalization_record = {"method": args.normalization}
    if args.normalization != "none":
        wrapper = type(model).wrapper_cls(model, mode="eval", latent_normalization=args.normalization)
        train_dataset = OpenXMP4FullDataset(**_dataset_options(model, args.data_root, "train"), **common)
        train_loader = DataLoader(train_dataset, batch_size=args.stats_batch_size, shuffle=True,
                                 drop_last=True, num_workers=0, collate_fn=train_dataset.get_collate_fn(),
                                 generator=torch.Generator().manual_seed(args.seed))
        if len(train_loader) == 0:
            raise ValueError("The training split is smaller than --stats-batch-size")
        torch.manual_seed(args.seed)
        wrapper.compute_embedding_stats(itertools.islice(train_loader, 100))
        if not torch.isfinite(wrapper.std_emb).all() or not (wrapper.std_emb > 0).all():
            raise ValueError("Training embedding statistics are not finite and positive")
        np.savez(args.output / "normalization.npz", mean=wrapper.mean_emb.cpu().numpy(),
                 std=wrapper.std_emb.cpu().numpy())
        normalization_record.update(split="train", maximum_batches=100, batch_size=args.stats_batch_size,
                                    batches=min(100, len(train_loader)), seed=args.seed,
                                    training_videos=len(train_dataset), workers=0, drop_last=True)
    record = dump_autoencoder_latents(model, loader, records, args.output, wrapper=wrapper,
                                      device=args.device, precision=args.precision)
    record["normalization_estimation"] = normalization_record
    if geometry:
        record["geometry_sampling"] = {
            "video_selection": "first 100 valid videos from the first 1024 sorted test paths",
            "frames_per_video": 10, "offsets": "uniform random start; per-video SeedSequence([42, index])",
            "manifest_supplied": args.manifest is not None,
        }
    (args.output / "dump_record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["reconstruction", "interventions", "dump", "geometry"], default="reconstruction")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--max-examples", type=int,
                        help="Evaluate only the first N examples (quick test; not the paper protocol)")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalization", choices=["zscore-channel", "global", "none"])
    parser.add_argument("--stats-batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    is_dump = args.mode in {"dump", "geometry"}
    args.batch_size = args.batch_size or (8 if is_dump else 16)
    args.precision = args.precision or ("bf16-mixed" if is_dump else "32-true")
    args.normalization = args.normalization or ("none" if args.mode == "geometry" else "zscore-channel")
    if args.mode == "dump" and args.manifest is None and args.max_examples is None:
        parser.error("Autoencoder latent dumps require the fixed 2000-clip --manifest, or --max-examples for a quick test")
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists() or (args.output / "dump_record.json").exists():
        raise FileExistsError(f"Use a new output directory: {args.output}")
    os.environ.setdefault("WM_LPIPS_CACHE_ROOT", str(args.assets.resolve() / "lpips"))
    os.environ.setdefault("TORCH_HOME", str(args.assets.resolve() / "torch"))
    commit = source_revision()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if is_dump:
        torch.set_float32_matmul_precision("high")
    model = load_autoencoder_checkpoint(args.checkpoint, args.assets).to(args.device).eval()
    summary = run_dump(model, args) if is_dump else run_reconstruction(model, args)
    summary.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=_sha256(args.checkpoint),
                   commit=commit, command=sys.argv, mode=args.mode, precision=args.precision,
                   device=args.device, seed=args.seed, max_examples=args.max_examples,
                   manifest_sha256=_sha256(args.output / "manifest.jsonl"))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "max_examples": args.max_examples}))


if __name__ == "__main__":
    main()
