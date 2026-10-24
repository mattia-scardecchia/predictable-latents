"""Cross-validated unpredictable share of the latents in one autoencoder latent dump.

The input is a directory written by ``wm_repr.evaluation.autoencoder --mode dump``. The predictor
regresses the next latent on the leading principal coordinates of the current latent and on the
action, with an own-lag term per coordinate fitted jointly in the frequency basis. The
unpredictable share is its held-out squared error divided by that of the training-fold mean.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import latent_bases as lb

# Principal coordinates of the current latent among the predictor's features.
N_COORDINATES = 256
# The randomised range finder is run for this many directions, of which the predictor uses the
# leading N_COORDINATES. The leading directions of a randomised SVD depend on how many it computes,
# so this number is part of the protocol.
N_DIRECTIONS = 1024


def load_dump(dump_dir: Path) -> dict:
    """Latents and actions of one autoencoder, as written by `--mode dump`."""
    record = json.loads((dump_dir / "dump_record.json").read_text())
    # A "partial" dump is an interrupted job, whose arrays are zero past the point it reached.
    if record.get("status") != "complete":
        raise ValueError(f"dump {dump_dir} is not usable: status={record.get('status')!r}")
    return {
        "latents": np.load(dump_dir / "latents.npy", mmap_mode="r"),
        "actions": np.load(dump_dir / "actions.npy"),
        "record": record,
    }


def build_features(coordinates: np.ndarray, actions: np.ndarray,
                   standardisation: tuple | None = None):
    """Leading principal coordinates of the current latent and the action.

    The columns are standardised so that one penalty acts comparably on a principal coordinate
    (whose scale is the square root of its eigenvalue) and on an action component. The training
    fold's mean and scale are reused on the held-out fold.
    """
    features = np.concatenate([coordinates[:, :N_COORDINATES], actions], axis=1)
    if standardisation is None:
        standardisation = lb.standardisation(features)
    mean, scale = standardisation
    features -= mean
    features /= scale
    return features, (mean, scale)


def select_penalty(current: np.ndarray, target: np.ndarray, coordinates: np.ndarray,
                   actions: np.ndarray, inner: np.ndarray) -> float:
    """Ridge penalty chosen inside the training folds on held-out squared error.

    `current` and `target` hold the centred latents in the frequency basis. The principal directions
    of the outer fold are reused here. The penalty is therefore chosen on training clips only, and
    the directions never saw the outer held-out clips, which is what the reported error requires.
    """
    validation = ~inner
    features, standard = build_features(coordinates[inner], actions[inner])
    solver = lb.OwnLagRidge(features, current[inner], target[inner])
    held_out, _ = build_features(coordinates[validation], actions[validation], standard)
    held_current, held_target = current[validation], target[validation]
    errors = []
    for alpha in lb.RIDGE_ALPHAS:
        beta, weights = solver.solve(alpha * solver.n_rows)
        difference = held_target - held_current * beta - held_out @ weights
        errors.append(float(np.einsum("ij,ij->", difference, difference, dtype=np.float64)))
    return float(lb.RIDGE_ALPHAS[int(np.argmin(errors))])


def fit_fold(current: np.ndarray, target: np.ndarray, coordinates: np.ndarray,
             actions: np.ndarray, alpha: float, latent_shape: tuple[int, ...]) -> dict:
    """Own-lag coefficients and regression weights on all the training clips.

    The weights are carried back from the frequency basis to the latent's own coordinates, so that
    a prediction costs one transform of the latent and none of the weights.
    """
    features, standard = build_features(coordinates, actions)
    solver = lb.OwnLagRidge(features, current, target)
    beta, weights = solver.solve(alpha * solver.n_rows)
    weights = lb.hartley_flat(np.ascontiguousarray(weights), latent_shape)
    return {"standardisation": standard, "own_lag": beta, "weights": weights}


def squared_error(error: np.ndarray) -> float:
    flat = np.ascontiguousarray(error, dtype=np.float32)
    return float(np.einsum("ij,ij->", flat, flat, dtype=np.float64))


def evaluate_fold(totals: dict, current: np.ndarray, following: np.ndarray,
                  current_frequency: np.ndarray, actions: np.ndarray, mean: dict,
                  directions: np.ndarray, fit: dict, latent_shape: tuple[int, ...],
                  chunk: int = 512) -> None:
    """Add the held-out squared errors of the predictor and of the training-fold mean."""
    for start in range(0, current.shape[0], chunk):
        piece = slice(start, start + chunk)
        centred = current[piece] - mean["spatial"]
        centred_frequency = current_frequency[piece] - mean["frequency"]
        target = following[piece]
        features, _ = build_features(centred @ directions, actions[piece], fit["standardisation"])
        own = lb.hartley_flat((centred_frequency * fit["own_lag"]).astype(np.float32), latent_shape)
        prediction = (mean["spatial"] + own + features @ fit["weights"]).astype(np.float32)
        totals["variance"] += squared_error(target - np.broadcast_to(mean["spatial"], target.shape))
        totals["prediction"] += squared_error(target - prediction)


def analyse_pipeline(dump_dir: Path, out_dir: Path, n_folds: int = 5,
                     max_clips: int | None = None) -> dict:
    started = time.time()
    dump = load_dump(dump_dir)
    latents = dump["latents"]
    clips = int(dump["record"].get("clips_done", latents.shape[0]))
    if max_clips is not None:
        clips = min(clips, max_clips)
    frames_per_clip = latents.shape[1]
    if n_folds < 3 or clips < n_folds or frames_per_clip < 3:
        raise ValueError("Need at least three folds, one clip per fold, and three frames per clip")
    latent_shape = tuple(int(s) for s in latents.shape[2:])
    dim = int(np.prod(latent_shape))

    frames = np.ascontiguousarray(latents[:clips], dtype=np.float32).reshape(-1, dim)
    actions = np.ascontiguousarray(
        dump["actions"][:clips, : frames_per_clip - 1], dtype=np.float32
    ).reshape(clips * (frames_per_clip - 1), -1)
    per_clip = frames.reshape(clips, frames_per_clip, dim)
    current = np.ascontiguousarray(per_clip[:, :-1]).reshape(-1, dim)
    following = np.ascontiguousarray(per_clip[:, 1:]).reshape(-1, dim)
    # The frequency coordinates of every transition, computed once: the own-lag term is diagonal
    # in them.
    current_frequency = lb.hartley_flat(current, latent_shape)
    following_frequency = lb.hartley_flat(following, latent_shape)

    totals = {"variance": 0.0, "prediction": 0.0}
    fold_of_clip = lb.clip_folds(clips, n_folds)
    fold_of_transition = np.repeat(fold_of_clip, frames_per_clip - 1)
    fold_of_frame = np.repeat(fold_of_clip, frames_per_clip)
    penalties = []

    for fold in range(n_folds):
        train_frames = frames[fold_of_frame != fold]
        spatial_mean = train_frames.mean(axis=0, dtype=np.float64).astype(np.float32)
        directions = lb.principal_directions(train_frames, spatial_mean, min(N_DIRECTIONS, dim))
        del train_frames
        mean = {"spatial": spatial_mean,
                "frequency": lb.hartley_flat(spatial_mean[None, :], latent_shape)[0]}

        train = fold_of_transition != fold
        test = ~train
        inner = (fold_of_transition != (fold + 1) % n_folds)[train]

        centred = current_frequency[train] - mean["frequency"]
        target = following_frequency[train] - mean["frequency"]
        train_actions = actions[train]
        coordinates = (current[train] - mean["spatial"]) @ directions
        alpha = select_penalty(centred, target, coordinates, train_actions, inner)
        penalties.append(alpha)
        fit = fit_fold(centred, target, coordinates, train_actions, alpha, latent_shape)
        del centred, target, coordinates, train_actions

        evaluate_fold(totals, current[test], following[test], current_frequency[test], actions[test],
                      mean, directions, fit, latent_shape)

    result = {
        "unpredictable_share": totals["prediction"] / totals["variance"],
        "total_variance_energy": totals["variance"],
        "dump_status": dump["record"].get("status"),
        "n_clips": int(clips),
        "n_transitions": int(current.shape[0]),
        "n_folds": int(n_folds),
        "frames_per_clip": int(frames_per_clip),
        "latent_shape": list(latent_shape),
        "max_clips": max_clips,
        "penalties_per_fold": penalties,
        # A penalty at either end of the grid means the search wanted to go further, so the fit is
        # not the best the predictor could do and the grid should be widened.
        "penalty_at_grid_edge": sorted(
            {v for v in penalties if v in (min(lb.RIDGE_ALPHAS), max(lb.RIDGE_ALPHAS))}
        ),
        "penalty_grid": list(lb.RIDGE_ALPHAS),
        "seconds": round(time.time() - started, 1),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "totals.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dump', type=Path, required=True, help='Autoencoder latent dump directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--max-clips', type=int,
                        help='Use only the first N clips (quick test; not the paper protocol)')
    args = parser.parse_args()
    if args.n_folds < 3:
        parser.error('At least three folds are required for held-out penalty selection')
    result = analyse_pipeline(args.dump, args.output, args.n_folds, args.max_clips)
    print(json.dumps({'unpredictable_share': result['unpredictable_share']}, indent=2))


if __name__ == '__main__':
    main()
