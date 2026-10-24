"""Convert Bridge V2 RLDS episodes to the video/action layout used by the paper."""

import argparse
import json
from pathlib import Path

import numpy as np


def bridge_actions(raw):
    """Map raw Bridge actions to ten stored columns; the loader adds the null flag."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 7:
        raise ValueError(f"Expected (frames, 7) actions, got {raw.shape}")
    actions = np.zeros((len(raw), 10), dtype=np.float32)
    actions[:, :3] = np.clip((raw[:, :3] + .05) / .1 * 3.5 - 1.75, -1.74, 1.74)
    actions[:, 3:6] = np.clip((raw[:, 3:6] + .25) / .5 * 2.8 - 1.4, -1.39, 1.39)
    actions[:, 6] = -(np.rint(raw[:, 6]) * 2 - 1)
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder-dir", required=True,
                        help="Directory containing the official Bridge V2 TFDS builder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    args = parser.parse_args()
    import tensorflow_datasets as tfds
    import torch
    from torchvision.io import write_video
    from tqdm import tqdm

    builder = tfds.builder_from_directory(args.builder_dir)
    splits = list(builder.info.splits)
    if "train" not in splits or not any(split != "train" for split in splits):
        raise ValueError("Use Bridge V2 with its published train/validation splits; no random split is substituted")
    counts = {"train": 0, "test": 0}
    records = []
    instructions = {"train": {}, "test": {}}
    for split in splits:
        destination = "train" if split == "train" else "test"
        target = args.output / "bridge_v2" / destination
        target.mkdir(parents=True, exist_ok=True)
        for source_index, episode in enumerate(tqdm(tfds.as_numpy(builder.as_dataset(split=split)), desc=split)):
            if args.max_episodes is not None and source_index >= args.max_episodes:
                break
            steps = list(episode["steps"])
            if not steps:
                raise ValueError(f"Empty episode at {split}:{source_index}")
            name = f"{counts[destination]:09d}"
            video_path = target / f"{name}.mp4"
            action_path = target / f"{name}_actions.npz"
            if video_path.exists() or action_path.exists():
                raise FileExistsError(f"Conversion output already exists: {video_path}")
            frames = np.stack([s["observation"]["image_0"] for s in steps])
            actions = bridge_actions(np.stack([s["action"] for s in steps]))
            write_video(str(video_path), torch.from_numpy(frames), fps=5)
            np.savez(action_path, actions)
            instruction = steps[0].get("language_instruction", b"")
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")
            instructions[destination][name] = str(instruction)
            records.append({"source_split": split, "source_index": source_index,
                            "video": str(video_path.relative_to(args.output)), "frames": len(frames)})
            counts[destination] += 1
    for split, values in instructions.items():
        (args.output / "bridge_v2" / split / "instructions.json").write_text(json.dumps(values))
    (args.output / "bridge_v2" / "conversion_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
