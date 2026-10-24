"""Fixed evaluation clips, keyed by relative file path and checked against lengths."""
from __future__ import annotations
from copy import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class EvaluationClips(Dataset):
    def __init__(self, dataset, records, n_frames=None):
        self.dataset = dataset
        self.records = records
        self.n_frames = n_frames
        paths = {str(p.relative_to(dataset.root_path)): i for i, p in enumerate(dataset.video_paths)}
        self.indices = []
        for row in records:
            idx = paths[row["video_path"]]
            if int(dataset.video_lengths[idx]) != int(row["video_length"]):
                raise ValueError(f"Video length differs from manifest: {row['video_path']}")
            self.indices.append(idx)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        ds = copy(self.dataset)
        n_frames = self.n_frames or int(row.get("n_frames", 10))
        ds.n_frames = n_frames
        ds.clip_len = n_frames
        ds.fixed_eval_clips = False
        start = int(row["start_frame"])
        if hasattr(ds, "force_start_idx"):
            ds.force_start_idx = start
        else:
            ds._sample_start_frame = lambda *_: start
        return ds[self.indices[index]]

    def get_collate_fn(self):
        return self.dataset.get_collate_fn()


def read_manifest(path, expected_sha256=None):
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Manifest checksum mismatch: {digest} != {expected_sha256}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty evaluation manifest")
    return rows, digest


def make_dataset(cfg, data_root, split="id", subset=None):
    from wm_repr.data.dataset import OpenXMP4FullDataset
    from wm_repr.data.autoeval_dataset import AutoEvalMP4Dataset
    p = cfg.data.params
    common = dict(root_path=str(data_root), split="test", input_h=int(p.input_h),
                  input_w=int(p.input_w), action_dim=int(p.action_dim), seed=42)
    if split == "ood":
        return AutoEvalMP4Dataset(
            **common, subset_names=subset or "autoeval_rollout_trajectories_splitted_4_all",
            n_frames=30, force_start_idx=20)
    return OpenXMP4FullDataset(**common, subset_names=subset or "bridge_v2",
                              n_frames=10, fixed_eval_clips=True)


def random_records(dataset, maximum):
    """Manifest rows for `maximum` clips drawn at random (seed 42), for a quick test without a manifest."""
    rng = np.random.default_rng(42)
    selected = rng.permutation(len(dataset))[:maximum]
    return [{"dataset_index": int(i), "video_path": str(dataset.video_paths[i].relative_to(dataset.root_path)),
             "video_length": int(dataset.video_lengths[i]),
             "start_frame": int(dataset.force_start_idx) if hasattr(dataset, "force_start_idx")
             else dataset._sample_start_frame(int(i), dataset.video_lengths[i]),
             "n_frames": int(dataset.n_frames)} for i in selected]


def make_loader(dataset, records, batch_size=8, workers=0, n_frames=None):
    selected = EvaluationClips(dataset, records, n_frames=n_frames)
    return DataLoader(selected, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=workers, collate_fn=selected.get_collate_fn(),
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(42))
