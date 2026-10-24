"""Task bootstrap for the agreement between world-model and real-world policy success."""

import json

from pathlib import Path

import numpy as np

from scipy.stats import pearsonr, spearmanr

from .reference import REAL_WORLD, COMPOUND_TASKS, TRIALS_PER_TASK

PAPER_POLICIES = ("openvla", "octo", "rt1")


def load_ckpt_cells(results_paths: list[str]) -> dict:
    """Load N results.json files (one per policy for ONE ckpt).

    Returns: cells[label][policy] -> {
        "rows":     np.ndarray of shape (n_rows,) with float scores,
        "sub_keys": tuple of sub_task_keys present in this cell (sorted),
    }
    A compound task joins its two sub-tasks under one label. Filters to PAPER_POLICIES and to
    tasks present in REAL_WORLD[policy].
    """
    cells: dict = {}

    for path in results_paths:
        data = json.loads(Path(path).read_text())
        policy = data.get("policy")
        if policy not in PAPER_POLICIES:
            continue

        rows_by_task: dict[str, list[float]] = {}
        for row in data["per_trial"]:
            tk = row["task_key"]
            if tk not in REAL_WORLD[policy]:
                continue
            rows_by_task.setdefault(tk, []).append(float(row["score"]))

        for tk, scores in rows_by_task.items():
            label = COMPOUND_TASKS.get(tk, tk.split("/")[-1])
            cell = cells.setdefault(label, {}).setdefault(policy, {"rows": [], "sub_keys": set()})
            cell["rows"].extend(scores)
            cell["sub_keys"].add(tk)

    for label in cells:
        for cell in cells[label].values():
            cell["rows"] = np.asarray(cell["rows"], dtype=np.float64)
            cell["sub_keys"] = tuple(sorted(cell["sub_keys"]))
    return cells


def _real_pct(policy: str, sub_keys: tuple) -> float:
    """Real-world success rate (%) for a (label, policy) cell, from its sub_task_key composition."""
    real_table = REAL_WORLD[policy]
    succ = sum(real_table[stk] for stk in sub_keys)
    return 100.0 * succ / (TRIALS_PER_TASK * len(sub_keys))


def _pairs_from_cells(cells: dict, labels) -> tuple[np.ndarray, np.ndarray]:
    """The (real_pct, wm_pct) vectors over the (labels × paper_policies) cells."""
    real_pcts: list[float] = []
    wm_pcts: list[float] = []
    for label in labels:
        for policy in PAPER_POLICIES:
            cell = cells.get(label, {}).get(policy)
            if cell is None:
                continue
            real_pcts.append(_real_pct(policy, cell["sub_keys"]))
            wm_pcts.append(100.0 * cell["rows"].mean())
    return np.asarray(real_pcts), np.asarray(wm_pcts)


def _correlation(cells: dict, labels, statistic=pearsonr) -> float:
    real, wm = _pairs_from_cells(cells, labels)
    if len(real) < 2:
        return float("nan")
    if real.std() == 0 or wm.std() == 0:
        return float("nan")
    return float(statistic(real, wm)[0])


def _resampled_labels(cells: dict, samples: int, seed: int):
    """The tasks of every bootstrap sample, drawn with replacement; a task keeps all its policies."""
    rng = np.random.default_rng(seed)
    labels = sorted(cells.keys())
    for _ in range(samples):
        yield list(rng.choice(labels, size=len(labels), replace=True))


def bootstrap_correlations(cells: dict, samples: int = 10000, seed: int = 0) -> dict:
    """Pearson r over the policy-task pairs of one checkpoint, and the bootstrap standard errors of
    Pearson r and Spearman rho: the standard deviations of their values on the same resampled tasks."""
    r = np.empty(samples, dtype=np.float64)
    rho = np.empty(samples, dtype=np.float64)
    for b, labels in enumerate(_resampled_labels(cells, samples, seed)):
        r[b] = _correlation(cells, labels)
        rho[b] = _correlation(cells, labels, spearmanr)
    return {
        "pearson": _correlation(cells, sorted(cells.keys())),
        "pearson_se": float(np.nanstd(r)),
        "spearman_se": float(np.nanstd(rho)),
        "samples": samples,
        "n_labels": len(cells),
    }


def _structural_match(cells_a: dict, cells_b: dict) -> None:
    """Assert two ckpt cell dicts share the same (label, policy, sub_keys, row count)."""
    if sorted(cells_a) != sorted(cells_b):
        raise ValueError("ckpts have different label sets")
    for label in cells_a:
        if sorted(cells_a[label]) != sorted(cells_b[label]):
            raise ValueError(f"label {label!r}: different policies across ckpts")
        for policy in cells_a[label]:
            ca, cb = cells_a[label][policy], cells_b[label][policy]
            if ca["sub_keys"] != cb["sub_keys"]:
                raise ValueError(f"({label!r}, {policy!r}): different sub_keys")
            if len(ca["rows"]) != len(cb["rows"]):
                raise ValueError(f"({label!r}, {policy!r}): different row count")


def bootstrap_delta_r(cells_a: dict, cells_b: dict, samples: int = 10000, seed: int = 0) -> dict:
    """Paired difference r(B) − r(A) for two checkpoints, on the same resampled tasks."""
    _structural_match(cells_a, cells_b)
    labels = sorted(cells_a.keys())
    delta = np.empty(samples, dtype=np.float64)
    for b, resampled in enumerate(_resampled_labels(cells_a, samples, seed)):
        delta[b] = _correlation(cells_b, resampled) - _correlation(cells_a, resampled)
    # Two-sided bootstrap p: 2 × min(P[Δ ≥ 0], P[Δ ≤ 0]), capped at 1 for a difference that is
    # zero in every sample.
    p_two = min(2.0 * min(np.mean(delta >= 0), np.mean(delta <= 0)), 1.0)
    return {
        "delta_r": _correlation(cells_b, labels) - _correlation(cells_a, labels),
        "delta_r_se": float(np.nanstd(delta)),
        "p_two_sided": float(p_two),
        "samples": samples,
    }
