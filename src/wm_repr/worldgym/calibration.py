"""Prepare human-labelled real episodes for judging and summarize their accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from wm_repr.utils.provenance import source_revision


def load_selection(path):
    selection = json.loads(Path(path).read_text())
    episodes = selection["episodes"]
    stems = [r["stem"] for r in episodes]
    if len(set(stems)) != len(stems):
        raise ValueError("Episode stems must be unique")
    for row in episodes:
        if Path(row["stem"]).name != row["stem"] or row["label"] not in (0, 1):
            raise ValueError("Each episode needs a filename stem and binary human label")
    return selection


def resample_video(video):
    if len(video) < 1:
        raise ValueError("Cannot resample an empty episode")
    indices = np.round(np.linspace(0, len(video) - 1, 41)).astype(int)
    return np.ascontiguousarray(video[indices], dtype=np.uint8), indices[::20].tolist()


def read_rgb(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return np.stack(frames)


def prepare(manifest, video_root, output):
    selection = load_selection(manifest)
    output = Path(output)
    (output / "videos").mkdir(parents=True, exist_ok=True)
    rows = []
    for item in selection["episodes"]:
        video = read_rgb(Path(video_root) / (item["stem"] + ".mp4"))
        sampled, indices = resample_video(video)
        if len(video) != item["video_frames"] or indices != item["source_frames"]:
            raise ValueError(f"{item['stem']}: frame count or selected frames differ from the manifest")
        relative = "videos/" + item["stem"] + ".npz"
        np.savez_compressed(output / relative, video=sampled)
        # Human labels remain separate from the requests sent to the judge.
        rows.append({"trial_png": item["stem"], "instruction": item["task"],
                     "partial_criteria": None, "task_key": item["task"],
                     "task_display": item["task"], "retry_idx": 0,
                     "video": relative, "shape": list(sampled.shape),
                     "sha256": hashlib.sha256(sampled.tobytes()).hexdigest()})
    (output / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    metadata = {"source": "human_labelled_real_episodes", "episodes": len(rows), "retries": 1,
                "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                "protocol": selection["protocol"], "commit": source_revision(), "command": sys.argv}
    (output / "rollout_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def summarize(manifest, judgements):
    selection = load_selection(manifest)
    rows = [json.loads(line) for line in Path(judgements).read_text().splitlines() if line.strip()]
    by_stem = {r["trial_png"]: r for r in rows}
    expected = {r["stem"] for r in selection["episodes"]}
    if len(by_stem) != len(rows) or set(by_stem) != expected:
        raise ValueError("Judgements must contain each selected episode exactly once")
    expected_n = selection["protocol"]["responses_per_episode"]
    comparisons = []
    for item in selection["episodes"]:
        row = by_stem[item["stem"]]
        if row["vote"] not in (0, 1) or len(row["parsed"]) != expected_n:
            raise ValueError("Calibration requires a binary vote and the recorded response count")
        comparisons.append({"stem": item["stem"], "task": item["task"],
                            "label": item["label"], "vote": row["vote"],
                            "correct": row["vote"] == item["label"],
                            "unparsed_responses": sum(v is None for v in row["parsed"])})

    def counts(items):
        total = len(items)
        correct = sum(r["correct"] for r in items)
        return {"episodes": total, "correct": correct, "accuracy": correct / total if total else None,
                "true_positive": sum(r["label"] == 1 and r["vote"] == 1 for r in items),
                "false_negative": sum(r["label"] == 1 and r["vote"] == 0 for r in items),
                "true_negative": sum(r["label"] == 0 and r["vote"] == 0 for r in items),
                "false_positive": sum(r["label"] == 0 and r["vote"] == 1 for r in items),
                "unparsed_responses": sum(r["unparsed_responses"] for r in items)}

    return {"overall": counts(comparisons),
            "by_task": {task: counts([r for r in comparisons if r["task"] == task])
                        for task in sorted({r["task"] for r in comparisons})},
            "per_episode": comparisons, "protocol": selection["protocol"],
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            "judgements_sha256": hashlib.sha256(Path(judgements).read_bytes()).hexdigest(),
            "commit": source_revision(), "command": sys.argv}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Resample real episodes into the saved-rollout format")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--video-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summary", help="Compare saved votes with human binary labels")
    summary.add_argument("--manifest", type=Path, required=True)
    summary.add_argument("--judgements", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.manifest, args.video_root, args.output)
    else:
        result = summarize(args.manifest, args.judgements)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("overall", {"episodes": result.get("episodes")}), indent=2))


if __name__ == "__main__":
    main()
