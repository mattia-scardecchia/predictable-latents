"""Bridge V2 videos as fixed-length clips (world models) or frame pairs (autoencoders)."""
import logging
import sys
from pathlib import Path
from typing import Sequence

import einops
import numpy as np
import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

# NOTE: convention is shape (B, T, C, H, W) and range [0,1] for batches of frames

# NOTE: We define the collate_fn's at module level so they can be pickled when using
# multiprocessing start methods like 'forkserver' or 'spawn'.


def openxmp4_full_collate_fn(batch):
    """Collate function for OpenXMP4FullDataset.
    Assumes that either all samples have actions or none have them.
    """
    clip, actions, instruction = zip(*batch)
    clip = torch.stack(clip, dim=0)
    actions = None if actions[0] is None else torch.stack(actions, dim=0)
    # Bridge clips carry no instruction (None) and AutoEval clips their task; a batch without any
    # instruction collapses to None.
    instruction = None if all(i is None for i in instruction) else list(instruction)
    return {
        "clip": clip,  # (B, T, C, H, W)
        "actions": actions,  # (B, T, D) or None
        "instruction": instruction,  # list[str] or None
    }


def openxmp4_tc_collate_fn(batch):
    """Collate function for OpenXMP4TCDataset."""
    current, future, other, actions = zip(*batch)
    current = torch.stack(current, dim=0)
    future = torch.stack(future, dim=0)
    if other[0] is None:
        other = None
    else:
        other = torch.stack(other, dim=0)
    if actions[0] is None:
        actions = None
    else:
        seq, mask = _pad_tensors_for_batching(list(actions))
        actions = {"seq": seq, "mask": mask}
    return {
        "current": current,  # (B, C, H, W)
        "future": future,  # (B, C, H, W)
        "other": other,  # (B, C, H, W) or None
        "actions": actions,  # seq: (B, T, D), mask: (B, T) (or None)
    }


def _pad_tensors_for_batching(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of variable-length action sequences to the same length.

    :param tensors: Sequence of tensors of shape (T_b, D)
    :returns: Tuple of the padded sequences and a boolean mask
    :returns seq: Tensor of shape (B, T_max, D) with zero padding
    :returns mask: Boolean tensor of shape (B, T_max); True for valid steps (t < T_b)
    """
    lengths = torch.as_tensor([e.size(0) for e in tensors], dtype=torch.int64)  # (B,)
    seq = pad_sequence(tensors, batch_first=True, padding_value=0.0)  # (B, T_max, D)
    T_max = seq.size(1)
    mask = torch.arange(T_max, device=seq.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )  # (B, T_max)
    return seq, mask


class OpenXMP4BaseDataset(Dataset):
    def __init__(
        self,
        root_path: str | Path,
        input_h: int,
        input_w: int,
        n_frames: int,  # for filtering
        subset_names: Sequence[str] | str,
        action_dim: int = 10,
        split: str = "train",
        max_videos: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root_path = Path(root_path)
        if isinstance(subset_names, str):
            subset_names = subset_names.split(",")
        self.subset_names = list(subset_names)

        self.n_frames = int(n_frames)
        self.clip_len = self.n_frames
        self.action_dim = int(action_dim)
        self.transform = transforms.Resize((int(input_h), int(input_w)))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.video_paths, self.video_lengths = self._index_videos(split, max_videos)
        if not self.video_paths:
            raise RuntimeError(
                f"No valid videos found in {self.root_path} for subsets {self.subset_names}"
            )

    def _index_videos(
        self, split: str, max_videos: int | None
    ) -> tuple[list[Path], list[int]]:
        """Enumerate videos + their frame lengths.

        Layout: ``root_path/{subset}/{split}/*.mp4``, each with a sibling ``{stem}_actions.npz``.
        """
        video_paths: list[Path] = []
        video_lengths: list[int] = []
        for name in self.subset_names:
            num_is_ok, num_total = 0, 0
            subset_dir = self.root_path / name / split
            mp4_paths = sorted(subset_dir.glob("*.mp4"))
            if max_videos is not None:
                mp4_paths = mp4_paths[:max_videos]
            for mp4_path in tqdm(
                mp4_paths,
                desc=f"Indexing {name} {split} videos",
                disable=not sys.stderr.isatty(),
            ):
                is_ok, length = self._check_video_for_indexing(mp4_path)
                num_total += 1
                if not is_ok:
                    continue
                num_is_ok += 1
                assert length is not None
                video_paths.append(mp4_path)
                video_lengths.append(length)
            if num_total > 0:
                logging.info(
                    f"Subset {name} {split}: {num_is_ok}/{num_total} videos included after indexing ({num_is_ok / num_total * 100:.2f}%)."
                )
        return video_paths, video_lengths

    def _check_video_for_indexing(self, mp4_path: Path):
        """
        :returns: (is_ok: bool, length: int | None)
        :returns is_ok: whether the video should be included
        :returns length: number of video frames, or None if loading failed
        """
        # try loading actions
        action_path = mp4_path.with_name(mp4_path.stem + "_actions.npz")
        try:
            actions = np.load(action_path)["arr_0"]
        except Exception as e:
            logging.warning(f"Error loading {action_path}: {e}")
            return False, None
        length = actions.shape[0]

        # filter by length
        if length < self.clip_len:
            return False, length

        return True, length

    def _load_actions(self, video_path: Path, start_frame: int, clip_len: int):
        if self.action_dim <= 0:
            return None
        action_path = video_path.with_name(video_path.stem + "_actions.npz")
        actions = np.load(action_path)["arr_0"]
        actions = actions[start_frame : start_frame + clip_len]
        if actions.shape[1] < self.action_dim:
            padding = np.zeros(
                (actions.shape[0], self.action_dim - actions.shape[1]),
                dtype=actions.dtype,
            )
            actions = np.concatenate([actions, padding], axis=1)
        else:
            raise ValueError(
                f"action_dim={self.action_dim} must exceed the {actions.shape[1]} stored action "
                "columns, to leave a slot for the null-action flag."
            )
        actions = torch.from_numpy(actions).float()
        return actions

    def _sample_clip(self, video_path: Path, clip_len: int, start_frame: int):
        video = EncodedVideo.from_path(str(video_path), decode_audio=False)
        fps = video._container.streams.video[0].guessed_rate
        start_sec = start_frame / fps
        end_sec = (start_frame + clip_len) / fps
        clip = video.get_clip(start_sec=start_sec, end_sec=end_sec)["video"]
        clip = einops.rearrange(clip, "c t h w -> t h w c")
        return clip, start_frame

    def __len__(self) -> int:
        return len(self.video_paths)


class OpenXMP4FullDataset(OpenXMP4BaseDataset):
    def __init__(
        self,
        split: str = "train",
        max_videos: int | None = None,
        fixed_eval_clips: bool = False,
        **kwargs,
    ) -> None:
        self.fixed_eval_clips = bool(fixed_eval_clips and split == "test")
        super().__init__(split=split, max_videos=max_videos, **kwargs)

    def _sample_start_frame(self, idx: int, video_length: int) -> int:
        rng = self.rng
        if self.fixed_eval_clips:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        return int(rng.integers(0, video_length - self.clip_len + 1))

    def fixed_clip_manifest(self):
        """Return deterministic validation clips for provenance and paired evaluation."""
        if not self.fixed_eval_clips:
            raise RuntimeError("fixed_clip_manifest requires fixed_eval_clips=True on test")
        for idx, (video_path, video_length) in enumerate(
            zip(self.video_paths, self.video_lengths)
        ):
            yield {
                "dataset_index": idx,
                "video_path": str(video_path.relative_to(self.root_path)),
                "video_length": int(video_length),
                "start_frame": self._sample_start_frame(idx, video_length),
                "clip_len": int(self.clip_len),
                "n_frames": int(self.n_frames),
                # Part of the manifest format; clips are always consecutive frames.
                "frame_skip": 1,
            }

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        video_length = self.video_lengths[idx]
        start_frame = self._sample_start_frame(idx, video_length)
        clip, start_frame = self._sample_clip(
            video_path, self.clip_len, int(start_frame)
        )
        actions = self._load_actions(video_path, int(start_frame), self.clip_len)

        assert len(clip) == self.n_frames
        assert actions is None or len(actions) == self.n_frames

        clip = clip.float() / 255.0
        clip = einops.rearrange(clip, "t h w c -> t c h w")
        clip = self.transform(clip)
        return clip, actions, None

    @staticmethod
    def get_collate_fn():
        return openxmp4_full_collate_fn


class OpenXMP4TCDataset(OpenXMP4BaseDataset):
    """Pairs of consecutive frames, with a third frame adjacent to the pair when ``sample_other``."""

    def __init__(
        self,
        root_path: str | Path,
        input_h: int,
        input_w: int,
        subset_names: Sequence[str] | str,
        sample_other: bool = True,
        action_dim: int = 10,
        split: str = "train",
        max_videos: int | None = None,
        seed: int = 42,
        fixed_eval_pairs: bool = False,
    ) -> None:
        super().__init__(
            root_path=root_path,
            input_h=input_h,
            input_w=input_w,
            # Index only videos of at least five frames with a third frame, three without.
            n_frames=5 if sample_other else 3,
            action_dim=action_dim,
            split=split,
            subset_names=subset_names,
            max_videos=max_videos,
            seed=seed,
        )
        self.sample_other = sample_other
        self.seed = int(seed)
        self.fixed_eval_pairs = bool(fixed_eval_pairs and split == "test")
        del self.n_frames
        del self.clip_len

    def _sample_triplet_indices(
        self, video_length: int, rng: np.random.Generator | None = None
    ):
        """Sample three consecutive frames (idx_curr, idx_fut, idx_other)."""
        rng = self.rng if rng is None else rng
        t1 = rng.integers(0, video_length - 2)
        t2 = t1 + 1
        t3 = t2 + 1
        # Randomly assign roles; ensure current < future.
        other_choice = rng.integers(1, 4)
        if other_choice == 1:
            return t2, t3, t1
        elif other_choice == 2:
            return t1, t3, t2
        else:
            return t1, t2, t3

    def _sample_pair_indices(
        self, video_length: int, rng: np.random.Generator | None = None
    ):
        """Sample two consecutive frames (idx_curr, idx_fut)."""
        rng = self.rng if rng is None else rng
        t1 = rng.integers(0, video_length - 1)
        return t1, t1 + 1

    def _sample_indices(self, idx: int, video_length: int):
        rng = self.rng
        if self.fixed_eval_pairs:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        if self.sample_other:
            return self._sample_triplet_indices(video_length, rng)
        idx_curr, idx_fut = self._sample_pair_indices(video_length, rng)
        return idx_curr, idx_fut, None

    def fixed_pair_manifest(self):
        """Return the deterministic evaluation sampling decisions for provenance."""
        if not self.fixed_eval_pairs:
            raise RuntimeError("fixed_pair_manifest requires fixed_eval_pairs=True on test")
        for idx, (video_path, video_length) in enumerate(
            zip(self.video_paths, self.video_lengths)
        ):
            idx_curr, idx_fut, idx_other = self._sample_indices(idx, video_length)
            yield {
                "dataset_index": idx,
                "video_path": str(video_path.relative_to(self.root_path)),
                "video_length": int(video_length),
                "current_index": int(idx_curr),
                "future_index": int(idx_fut),
                "other_index": None if idx_other is None else int(idx_other),
            }

    def __getitem__(self, idx: int):
        video_path = self.video_paths[idx]
        video_length = self.video_lengths[idx]

        idx_curr, idx_fut, idx_other = self._sample_indices(idx, video_length)
        if idx_other is not None:
            start = min(idx_curr, idx_fut, idx_other)
            end = max(idx_curr, idx_fut, idx_other)
        else:
            start, end = idx_curr, idx_fut

        clip, _ = self._sample_clip(video_path, end - start + 1, start)
        frames = [clip[idx_curr - start], clip[idx_fut - start]]
        if idx_other is not None:
            frames.append(clip[idx_other - start])
        actions = self._load_actions(video_path, idx_curr, idx_fut - idx_curr)

        stacked = torch.stack(frames, dim=0)
        stacked = stacked.float() / 255.0
        stacked = einops.rearrange(stacked, "t h w c -> t c h w")
        stacked = self.transform(stacked)
        frame_other = stacked[2] if idx_other is not None else None
        return stacked[0], stacked[1], frame_other, actions

    @staticmethod
    def get_collate_fn():
        return openxmp4_tc_collate_fn
