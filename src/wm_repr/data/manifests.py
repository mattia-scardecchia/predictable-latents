#!/usr/bin/env python3
"""Build the paper evaluation manifests for Bridge and AutoEval."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from wm_repr.utils.provenance import source_revision
from wm_repr.data.dataset import OpenXMP4FullDataset
from wm_repr.data.autoeval_dataset import AutoEvalMP4Dataset, _task_from_stem


def write_jsonl(path: Path, records: list[dict]) -> dict:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "entries": len(records),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def clip_record(dataset: OpenXMP4FullDataset, idx: int) -> dict:
    path = dataset.video_paths[idx]
    length = int(dataset.video_lengths[idx])
    return {
        "dataset_index": int(idx),
        "video_path": str(path.relative_to(dataset.root_path)),
        "video_length": length,
        "start_frame": dataset._sample_start_frame(idx, length),
        "clip_len": int(dataset.clip_len),
        "n_frames": int(dataset.n_frames),
        # Part of the manifest format; clips are always consecutive frames.
        "frame_skip": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--autoeval-root", type=Path)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--clip-seed", type=int, default=42)
    parser.add_argument("--source-commit", default=source_revision())
    parser.add_argument("--small", action="store_true",
                        help="Accept datasets smaller than the paper's (quick test; not the paper protocol)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = OpenXMP4FullDataset(
        root_path=args.data_root,
        input_h=256,
        input_w=256,
        n_frames=10,
        action_dim=11,
        subset_names="bridge_v2",
        split="test",
        seed=args.clip_seed,
        fixed_eval_clips=True,
    )
    if len(dataset) < 2_000 and not args.small:
        raise RuntimeError(f"Need at least 2,000 eligible Bridge test videos; found {len(dataset)}")
    rng = np.random.default_rng(args.selection_seed)
    permutation = rng.permutation(len(dataset))
    id_indices = permutation[:2_000].tolist()

    files = {}
    files["id_2000"] = write_jsonl(
        args.output / "bridge_id_2000.jsonl",
        [clip_record(dataset, int(idx)) for idx in id_indices],
    )
    files["steps_512"] = write_jsonl(
        args.output / "bridge_steps_512.jsonl",
        [clip_record(dataset, int(idx)) for idx in id_indices[:512]],
    )
    files["control_256"] = write_jsonl(
        args.output / "bridge_control_256.jsonl",
        [clip_record(dataset, int(idx)) for idx in id_indices[:256]],
    )

    eligible = np.flatnonzero(np.asarray(dataset.video_lengths) >= 20)
    tail = np.flatnonzero(np.asarray(dataset.video_lengths) >= 60)
    if len(tail) != 224 and not args.small:
        raise RuntimeError(
            f"Protocol expects 224 Bridge test videos with length >=60; found {len(tail)}"
        )
    uniform = rng.choice(eligible, size=min(512, len(eligible)) if args.small else 512, replace=False)
    tail_set = set(int(idx) for idx in tail)
    uniform_set = set(int(idx) for idx in uniform)
    drift_indices = sorted(tail_set | uniform_set)
    drift_records = []
    for idx in drift_indices:
        if idx in tail_set and idx in uniform_set:
            stratum = "tail+uniform"
        elif idx in tail_set:
            stratum = "tail"
        else:
            stratum = "uniform"
        record = clip_record(dataset, idx)
        record["start_frame"] = 0
        record["stratum"] = stratum
        drift_records.append(record)
    files["drift"] = write_jsonl(
        args.output / "bridge_drift.jsonl", drift_records
    )

    ood_summary = None
    if args.autoeval_root is not None:
        ood = AutoEvalMP4Dataset(
            root_path=args.autoeval_root,
            subset_names="autoeval_rollout_trajectories_splitted_4_all",
            split="test",
            input_h=256,
            input_w=256,
            n_frames=30,
            action_dim=11,
            force_start_idx=20,
            seed=args.clip_seed,
        )
        if len(ood) != 479 and not args.small:
            raise RuntimeError(f"Protocol expects 479 AutoEval trajectories; found {len(ood)}")
        ood_records = []
        by_task: dict[str, list[dict]] = {}
        for idx, (path, length) in enumerate(zip(ood.video_paths, ood.video_lengths)):
            task = _task_from_stem(path.stem)
            if task is None:
                raise RuntimeError(f"Could not parse task from {path}")
            record = {
                "dataset_index": idx,
                "stem": path.stem,
                "video_path": str(path.relative_to(ood.root_path)),
                "video_length": int(length),
                "start_frame": 20,
                "n_frames": 30,
                "task": task,
            }
            ood_records.append(record)
            by_task.setdefault(task, []).append(record)
        files["ood_all_479"] = write_jsonl(
            args.output / "autoeval_all_479.jsonl", ood_records
        )
        balanced_256 = []
        for task in sorted(by_task):
            task_records = by_task[task]
            task_order = rng.permutation(len(task_records))
            if len(task_records) < 64 and not args.small:
                raise RuntimeError(f"Task {task!r} has only {len(task_records)} trajectories")
            balanced_256.extend(task_records[int(i)] for i in task_order[:64])
        files["ood_control_256"] = write_jsonl(
            args.output / "autoeval_control_256.jsonl", balanced_256
        )

        ood_summary = {task: len(records) for task, records in sorted(by_task.items())}

    index = {
        "schema_version": 1,
        "small": args.small,
        "dataset": "bridge_v2/test",
        "dataset_root": str(args.data_root),
        "eligible_clips": len(dataset),
        "selection_seed": args.selection_seed,
        "clip_seed": args.clip_seed,
        "source_commit": args.source_commit,
        "drift_tail_count": len(tail_set),
        "drift_uniform_count": len(uniform_set),
        "drift_overlap_count": len(tail_set & uniform_set),
        "files": files,
        "ood_task_counts": ood_summary,
    }
    index_path = args.output / "index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(index_path)


if __name__ == "__main__":
    main()
