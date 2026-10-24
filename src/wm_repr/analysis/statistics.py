"""Training-curve crossings for the training-speed factors."""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def crossing(updates, curve, target, higher=False):
    """First target crossing, interpolated in log(update)."""
    updates, curve = np.asarray(updates, float), np.asarray(curve, float)
    if (updates.ndim != 1 or updates.shape != curve.shape or not len(updates)
            or not np.isfinite(updates).all() or not np.isfinite(curve).all()
            or not np.isfinite(target) or (updates <= 0).any() or (np.diff(updates) <= 0).any()):
        raise ValueError("Use finite curves and strictly increasing positive update counts")
    ok = curve >= target if higher else curve <= target
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return math.nan, math.nan, "not reached"
    k = idx[0]
    if k == 0:
        return updates[0], updates[0], "lower bound"
    t = (target - curve[k - 1]) / (curve[k] - curve[k - 1])
    lu = math.log(updates[k - 1]) + t * (math.log(updates[k]) - math.log(updates[k - 1]))
    return updates[k], math.exp(lu), "reached"


def training_factors(rows, treated, control, metric, higher=False):
    """Compare seed-mean curves and each matched training seed on one validation grid."""
    rows = rows[rows.metric == metric]
    if rows.duplicated(["pipeline", "seed", "update"]).any():
        raise ValueError("Duplicate validation values for a pipeline, seed and update")
    a = rows[rows.pipeline == treated].pivot(index="update", columns="seed", values="value")
    r = rows[rows.pipeline == control].pivot(index="update", columns="seed", values="value")
    if (a.empty or not a.index.equals(r.index) or not a.columns.equals(r.columns)
            or a.isna().any().any() or r.isna().any().any()):
        raise ValueError("Both pipelines need the same complete validation grid and training seeds")
    curves = {"mean": (a.mean(axis=1), r.mean(axis=1))}
    curves.update({str(s): (a[s], r[s]) for s in a.columns})
    results = []
    for seed, (ca, cr) in curves.items():
        end, target = float(a.index.max()), float(cr.iloc[-1])
        at_val, interp, status = crossing(ca.index.values, ca.values, target, higher)
        results.append(dict(metric=metric, pipeline=treated, reference=control, training_seed=seed,
                            final_update=end, reference_final=target, first_validation=at_val,
                            crossing=interp, factor_validation=end / at_val, factor=end / interp,
                            status=status))
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Validation values, one row per pipeline, seed, update and metric")
    parser.add_argument("--treated", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--higher", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    training_factors(pd.read_csv(args.input), args.treated, args.control, args.metric,
                     args.higher).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
