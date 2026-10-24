"""Judge saved WorldGym rollout cells, retaining the upstream prompt and voting rule."""

import argparse
import hashlib
import json
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from .claude_judge import ClaudeCLI, cli_version
from . import scorer as worldgym
from wm_repr.utils.provenance import source_revision


def rollout_items(rollout_dir):
    rows = [json.loads(l) for l in (rollout_dir / "rollouts.jsonl").read_text().splitlines()]
    meta = json.loads((rollout_dir / "rollout_meta.json").read_text())
    items = []
    for r in rows:
        items.append({"id": r["video"], "trial": {k: r[k] for k in
                                                  ("trial_png", "instruction", "partial_criteria",
                                                   "task_key", "task_display")},
                      "retry_idx": r["retry_idx"], "video_file": str(rollout_dir / r["video"]),
                      "sha256": r["sha256"]})
    return items, meta


def load_video(item):
    path = item["video_file"]
    video = np.load(path)["video"]
    if hashlib.sha256(np.ascontiguousarray(video).tobytes()).hexdigest() != item["sha256"]:
        raise RuntimeError(f"{path}: frames differ from the ones saved at rollout time")
    return video


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rollout-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, help="default: the rollout dir")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--n", type=int, default=3, help="independent judge samples per rollout")
    p.add_argument("--parallel", type=int, default=2, help="rollouts judged at once (n calls each)")
    args = p.parse_args()

    items, meta = rollout_items(args.rollout_dir)
    out = args.output or args.rollout_dir
    for d in ("judge_raw", "judge_workdir"):
        (out / d).mkdir(parents=True, exist_ok=True)
    judge = {"model": args.model, "effort": args.effort, "n": args.n, "system_prompt": "",
             "claude_cli": cli_version(), "commit": source_revision(), "command": sys.argv}
    (out / "judge_meta.json").write_text(json.dumps({**judge, "input": meta}, indent=2))

    path = out / "judgements.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} rollouts, {len(done)} already judged, {len(todo)} to judge", flush=True)
    lock = threading.Lock()

    def judge_one(item):
        video = load_video(item)
        client = ClaudeCLI(args.model, args.effort, out / "judge_workdir", out / "judge_raw",
                           item["id"].replace("/", "__").replace(":", "_"))
        vote, per_call = worldgym.predict(video, item["trial"], n=args.n, return_raw=True, client=client)
        row = {"id": item["id"], **item["trial"], "retry_idx": item["retry_idx"], "vote": vote,
               "parsed": [c["parsed_score"] for c in per_call],
               "request_sha256": hashlib.sha256(json.dumps(client.request).encode()).hexdigest(),
               "samples": client.calls}
        with lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    with ThreadPoolExecutor(args.parallel) as pool:
        pending, queue = set(), list(todo)
        while queue or pending:
            while queue and len(pending) < args.parallel:
                pending.add(pool.submit(judge_one, queue.pop(0)))
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in finished:
                row = f.result()  # a failed rollout stops the job; finished ones are kept
                print(f"{row['id']} | vote={row['vote']} | parsed={row['parsed']}", flush=True)

    rows = {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}
    if len(rows) < len(items):
        print(f"{len(items) - len(rows)} rollouts not judged; results.json not written", flush=True)
        sys.exit(1)
    per_trial = [{"task_key": it["trial"]["task_key"], "task_display": it["trial"]["task_display"],
                  "score": float(rows[it["id"]]["vote"])} for it in items]
    per_scorer_call = [{"id": it["id"], "task_key": it["trial"]["task_key"],
                        "trial_png": it["trial"]["trial_png"], "retry_idx": it["retry_idx"],
                        "scorer_call_idx": k, "score": rows[it["id"]]["parsed"][k],
                        "rationale": s["text"], "judge_meta": {x: s[x] for x in
                                                               ("model", "usage", "wall_s", "attempt", "raw")}}
                       for it in items for k, s in enumerate(rows[it["id"]]["samples"])]
    payload = {**meta, "scorer_n": args.n, "judge": judge, "per_trial": per_trial,
               "aggregate": worldgym.aggregate_model_results(per_trial), "per_scorer_call": per_scorer_call}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    worldgym.print_results_table(payload["aggregate"])


if __name__ == "__main__":
    main()
