"""AutoEval trajectories with Bridge-compatible action scaling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import einops
import numpy as np
import torch

from .dataset import OpenXMP4FullDataset, openxmp4_full_collate_fn

# The four AutoEval tasks (must match the filename prefixes on disk).
AUTOEVAL_TASKS: tuple[str, ...] = (
    "close the drawer",
    "open the drawer",
    "put the eggplant in the blue sink",
    "put the eggplant in the yellow basket",
)

# Translation and rotation scaling leave a 0.01 margin inside the action bounds.
_XYZ_SCALE, _XYZ_CLIP = 35.0, 1.74
_RPY_SCALE, _RPY_CLIP = 5.6, 1.39


def autoeval_full_collate_fn(batch):
    """Like ``openxmp4_full_collate_fn`` but carries the per-trajectory source
    ``stem`` (for naming saved gt/pred videos)."""
    core = [(clip, actions, instruction) for clip, actions, instruction, _ in batch]
    out = openxmp4_full_collate_fn(core)
    out["stem"] = [stem for *_, stem in batch]  # list[str]
    return out


def _transform_autoeval_actions(
    raw: np.ndarray,  # (T, >=7) raw bridge 7-DoF [Δx,Δy,Δz,Δr,Δp,Δy,gripper]
    action_dim: int,
) -> torch.Tensor:
    """Raw AutoEval 7-DoF → the world model's normalized (T, action_dim) action.

    xyz/rpy: Bridge action scaling.
    gripper: AutoEval is continuous with ``+1 = open``, the opposite sign of Bridge's
      ``-1 = open``. It is binarized to match the binary Bridge training actions:
      ``+1 if raw < 0.5 else -1`` (low→closed→+1, high→open→-1).
    Remaining dims are zero-padded; index ``action_dim-1`` stays 0 (null flag).
    """
    if action_dim <= 7:
        raise ValueError(
            f"AutoEval action_dim must be > 7 to leave the bridge 7-DoF + a "
            f"null-action slot at index -1; got action_dim={action_dim}."
        )
    raw = np.asarray(raw, dtype=np.float32)
    xyz = np.clip(_XYZ_SCALE * raw[:, 0:3], -_XYZ_CLIP, _XYZ_CLIP)
    rpy = np.clip(_RPY_SCALE * raw[:, 3:6], -_RPY_CLIP, _RPY_CLIP)
    grip = np.where(raw[:, 6] < 0.5, 1.0, -1.0).astype(np.float32)

    seven = np.concatenate([xyz, rpy, grip[:, None]], axis=1)  # (T, 7)
    pad = np.zeros((seven.shape[0], action_dim - 7), dtype=np.float32)  # incl. null slot
    actions = np.concatenate([seven, pad], axis=1)  # (T, action_dim), index -1 = 0
    return torch.from_numpy(actions).float()


def _task_from_stem(stem: str) -> str | None:
    """Derive the task name from ``<task>_<timestamp>`` by prefix-matching the
    known task set (robust to the timestamp format)."""
    for task in AUTOEVAL_TASKS:
        if stem.startswith(task):
            return task
    return None


class AutoEvalMP4Dataset(OpenXMP4FullDataset):
    """AutoEval real-robot trajectories for action-replay OOD WM evaluation.

    Returns ``(clip, actions, instruction, stem)`` and uses
    :func:`autoeval_full_collate_fn`. A deterministic ``force_start_idx`` replaces
    the base class's random start so every run rolls out from the same frame.
    """

    def __init__(
        self,
        root_path: str | Path,
        input_h: int = 256,
        input_w: int = 256,
        n_frames: int = 30,
        action_dim: int = 11,
        split: str = "test",
        subset_names: Sequence[str] | str = "autoeval_rollout_trajectories_splitted_4_all",
        max_videos: int | None = None,
        seed: int = 42,
        force_start_idx: int = 20,
    ) -> None:
        if split != "test":
            raise ValueError(f"AutoEvalMP4Dataset is test-only; got split={split!r}.")
        self.force_start_idx = int(force_start_idx)
        super().__init__(
            root_path=root_path,
            input_h=input_h,
            input_w=input_w,
            n_frames=n_frames,
            action_dim=action_dim,
            split=split,
            subset_names=subset_names,
            max_videos=max_videos,
            seed=seed,
        )

    def _check_video_for_indexing(self, mp4_path: Path):
        """Require a known task and a trajectory long enough to roll out from ``force_start_idx``."""
        task = _task_from_stem(mp4_path.stem)
        if task is None:
            logging.warning(f"Could not parse task from {mp4_path.name}; skipping.")
            return False, None
        action_path = mp4_path.with_name(mp4_path.stem + "_actions.npz")
        try:
            with np.load(action_path) as f:
                length = int(f["actions"].shape[0])
        except Exception as e:  # noqa: BLE001
            logging.warning(f"Error loading {action_path}: {e}")
            return False, None
        # Need a full clip starting at force_start_idx.
        if self.force_start_idx + self.clip_len > length:
            return False, length
        return True, length

    def _load_actions(self, video_path: Path, start_frame: int, clip_len: int):
        if self.action_dim <= 0:
            return None
        action_path = video_path.with_name(video_path.stem + "_actions.npz")
        with np.load(action_path) as f:
            raw = f["actions"][start_frame : start_frame + clip_len]
        return _transform_autoeval_actions(raw, self.action_dim)

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        video_length = self.video_lengths[idx]
        # Deterministic start (clamped so the clip always fits).
        start_frame = int(min(self.force_start_idx, max(0, video_length - self.clip_len)))
        clip, start_frame = self._sample_clip(video_path, self.clip_len, start_frame)
        actions = self._load_actions(video_path, int(start_frame), self.clip_len)

        assert len(clip) == self.n_frames, (len(clip), self.n_frames)
        assert actions is None or len(actions) == self.n_frames

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)

        return clip, actions, _task_from_stem(video_path.stem), video_path.stem

    @staticmethod
    def get_collate_fn():
        return autoeval_full_collate_fn
