"""Summarize success rates and their agreement with real-world success from judged rollout cells."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .bootstrap import load_ckpt_cells, bootstrap_correlations, bootstrap_delta_r
from .reference import aggregate_policy


def starting_frame_scores(data):
    """Average retries within each starting observation before estimating uncertainty."""
    retries = int(data.get("retries", 1))
    n = int(data.get("scorer_n", 1))
    calls = data.get("per_scorer_call")
    if calls is not None and len(calls) != n * len(data["per_trial"]):
        raise ValueError("Per-scorer rows do not match the rollout count")
    counts, groups = {}, {}
    for index, row in enumerate(data["per_trial"]):
        task = row["task_key"]
        position = counts.get(task, 0)
        counts[task] = position + 1
        trial = position // retries
        if calls is not None:
            call = calls[index * n]
            if call["task_key"] != task or call["retry_idx"] != position % retries:
                raise ValueError("Rollout rows must be ordered by task, starting frame and retry")
            trial = call["trial_png"]
        groups.setdefault((task, trial), []).append(float(row["score"]))
    if any(len(values) != retries for values in groups.values()):
        raise ValueError("Incomplete retries for a starting observation")
    return np.asarray([np.mean(values) for values in groups.values()])


def summarize(paths, samples=10000):
    cells = load_ckpt_cells([str(path) for path in paths])
    pooled = bootstrap_correlations(cells, samples=samples, seed=0)
    result = {"pearson": pooled["pearson"], "pearson_se": pooled["pearson_se"], "per_policy": {}}
    real, simulated = [], []
    for path in paths:
        data = json.loads(Path(path).read_text())
        policy = data["policy"]
        if policy in result["per_policy"]:
            raise ValueError(f"More than one results cell for policy {policy}")
        x, y, labels = aggregate_policy(data, policy)
        real.extend(x)
        simulated.extend(y)
        trials = starting_frame_scores(data)
        correlations = bootstrap_correlations(load_ckpt_cells([str(path)]), samples=samples, seed=0)
        result["per_policy"][policy] = {
            "success_rate": float(trials.mean()),
            "success_rate_sem": float(trials.std(ddof=1) / np.sqrt(len(trials))) if len(trials) > 1 else None,
            "n_starting_frames": len(trials), "n_rollouts": len(data["per_trial"]),
            "pearson": correlations["pearson"], "pearson_se": correlations["pearson_se"],
            "spearman": float(spearmanr(x, y).statistic), "spearman_se": correlations["spearman_se"],
            "task_labels": labels, "real_success_percent": x, "model_success_percent": y,
        }
    result["spearman"] = float(spearmanr(real, simulated).statistic)
    result["spearman_se"] = pooled["spearman_se"]
    result["bootstrap"] = {"unit": "task", "samples": samples, "seed": 0, "n_tasks": pooled["n_labels"]}
    return result, cells


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True, type=Path,
                   help="One results.json per policy, all from the same world model and step count")
    p.add_argument("--compare", nargs="+", type=Path, help="Reference world model, same policy/task cells")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--samples", type=int, default=10000)
    args = p.parse_args()
    result, cells = summarize(args.results, args.samples)
    if args.compare:
        reference = load_ckpt_cells([str(path) for path in args.compare])
        result["difference"] = bootstrap_delta_r(reference, cells, samples=args.samples, seed=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
