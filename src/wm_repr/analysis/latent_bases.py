"""The Hartley basis, the ridge predictor and the latent-geometry statistics.

Frames have layout (height, width, channels). Clip-level folds keep frames from the same clip out
of both train and test sets.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg.blas
import torch
from sklearn.utils.extmath import randomized_svd


# Ridge penalties, as a multiple of the number of training rows. The features are standardised
# first, so their cross-product is about the number of rows times the identity and one grid covers
# every pipeline: a standardised direction keeps the fraction 1 / (1 + alpha) of its unpenalised
# weight, so the grid runs from a fit that is barely penalised to one whose weights are gone.
RIDGE_ALPHAS = (1e-4, 1e-2, 1.0, 1e2, 1e4)


# ---------------------------------------------------------------------------
# Hartley basis
# ---------------------------------------------------------------------------


def hartley_matrix(n_side: int) -> np.ndarray:
    """The one-dimensional Hartley matrix: orthogonal, symmetric and its own inverse."""
    k = np.arange(n_side)
    angle = 2.0 * np.pi * np.outer(k, k) / n_side
    return (np.cos(angle) + np.sin(angle)) / np.sqrt(n_side)


def hartley(maps: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    """Separable two-dimensional Hartley transform of a stack of real maps, (n, side, side, c).

    A real orthogonal transform whose coefficient (u, v) lies in the span of the Fourier bins
    (u, v) and (u, -v), which have the same radial frequency, so a predictor that is diagonal here
    is diagonal in radial frequency. Being real and involutive, the transform needs no complex
    arithmetic and no separate inverse.
    """
    if matrix is None:
        matrix = hartley_matrix(maps.shape[1])
    matrix = matrix.astype(maps.dtype, copy=False)
    return np.einsum("ur,nrvc,vs->nusc", matrix, maps, matrix, optimize=True)


def hartley_flat(flat: np.ndarray, latent_shape: tuple[int, ...],
                 matrix: np.ndarray | None = None, chunk: int = 2048) -> np.ndarray:
    """`hartley` on rows that are flattened latents, in blocks, keeping the input's dtype."""
    if matrix is None:
        matrix = hartley_matrix(latent_shape[0])
    out = np.empty_like(flat)
    for start in range(0, flat.shape[0], chunk):
        piece = slice(start, start + chunk)
        block = flat[piece].reshape(-1, *latent_shape)
        out[piece] = hartley(block, matrix).reshape(block.shape[0], -1)
    return out


# ---------------------------------------------------------------------------
# Folds, principal directions and the predictors
# ---------------------------------------------------------------------------


def clip_folds(n_clips: int, n_folds: int = 5) -> np.ndarray:
    """Fold of every clip. Folds are over clips, never over frames: the ten frames of a clip share
    most of their variance, so a frame-level split would leak a held-out clip into its own fit.
    The clips of the manifest were drawn in a seeded random order, so taking every ``n_folds``-th
    one needs no second random choice and gives folds of equal size."""
    return np.arange(n_clips) % n_folds


def covariance(X: np.ndarray, mean: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Sample covariance (one degree of freedom removed) of the rows of ``X`` about ``mean``.

    Accumulated in double precision over row blocks: a single-precision sum over 20,000 rows loses
    about five digits, which the small eigenvalues of a 16,384-coordinate latent cannot afford.
    The symmetric update computes one triangle in place, which halves both the arithmetic and the
    memory traffic of a 16,384 by 16,384 accumulation.
    """
    n, d = X.shape
    out = np.zeros((d, d), dtype=np.float64, order="F")
    for start in range(0, n, chunk):
        block = np.asfortranarray(X[start : start + chunk].astype(np.float64) - mean)
        out = scipy.linalg.blas.dsyrk(
            alpha=1.0, a=block, trans=1, c=out, beta=1.0, lower=1, overwrite_c=1
        )
    out += np.tril(out, -1).T
    return out / (n - 1)


def gram_eigenvalues(X: np.ndarray, mean: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Eigenvalues of the sample covariance of the rows of ``X`` when the rows are fewer than the
    coordinates, in descending order and padded with zeros to the number of coordinates.

    The centred n-by-d matrix has the same non-zero squared singular values whether they are taken
    from the d-by-d covariance or from the n-by-n Gram matrix; the Gram matrix of 20,000 frames is
    3.2 GB where the covariance of a 196,608-coordinate latent would be 309 GB. Accumulated in
    double precision over column blocks, as ``covariance`` is over row blocks.
    """
    n, d = X.shape
    gram = np.zeros((n, n), dtype=np.float64, order="F")
    for start in range(0, d, chunk):
        block = np.asfortranarray(X[:, start : start + chunk].astype(np.float64) - mean[start : start + chunk])
        gram = scipy.linalg.blas.dsyrk(
            alpha=1.0, a=block, trans=0, c=gram, beta=1.0, lower=1, overwrite_c=1
        )
    gram += np.tril(gram, -1).T
    values = np.clip(np.linalg.eigvalsh(gram)[::-1] / (n - 1), 0.0, None)
    return np.concatenate([values, np.zeros(d - n)]) if d > n else values


def spectrum_scalars(eigenvalues: np.ndarray) -> dict:
    """Components for 90, 95 and 99% of the variance, and the participation ratio.

    ``eigenvalues`` are covariance eigenvalues in descending order. The count is the first
    component at which their cumulative share reaches the threshold. The participation ratio
    is their squared sum divided by the sum of their squares.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    cumulative = np.cumsum(eigenvalues / eigenvalues.sum()) * 100.0
    out = {}
    for threshold in (90, 95, 99):
        index = int(np.searchsorted(cumulative, threshold))
        saturated = index >= len(cumulative)
        out[f"pca_n{threshold}"] = int(len(cumulative)) if saturated else index + 1
        out[f"pca_n{threshold}_saturated"] = bool(saturated)
    out["pca_participation_ratio"] = float(
        eigenvalues.sum() ** 2 / (np.square(eigenvalues).sum() + 1e-12)
    )
    out["pca_n_components_fit"] = int(len(eigenvalues))
    return out


def principal_directions(
    X: np.ndarray, mean: np.ndarray, n_directions: int, random_state: int = 0, n_iter: int = 4
) -> np.ndarray:
    """Leading ``n_directions`` principal directions of the rows of ``X``, as (d, n_directions).

    A randomised range finder on the centred data matrix, not an eigendecomposition of the
    covariance: at 16,384 coordinates the full decomposition costs minutes per fold and squaring
    the data matrix costs accuracy in the directions that matter least.
    """
    centred = X.astype(np.float32) - mean.astype(np.float32)
    _, _, components = randomized_svd(
        centred,
        n_components=n_directions,
        n_oversamples=32,
        n_iter=n_iter,
        random_state=random_state,
    )
    return np.ascontiguousarray(components.T)


class OwnLagRidge:
    """Predict every coordinate of the next latent from its own previous value and p features.

    The model for output j is ``y_j = beta_j * x_j + F w_j``, with a ridge penalty on ``w`` only.
    The 16,384 x 16,384 normal equations of the outputs are never formed: eliminating ``w`` leaves
    one p x p system shared by every output and a scalar per output, so a fit costs two passes of
    p x n x d and the whole penalty grid comes from one eigendecomposition.

    ``beta`` and ``w`` are fitted **together**. Fitting the own-lag coefficient first and
    regressing on what it leaves over is biased whenever a coordinate mixes a predictable and an
    unpredictable part: the coefficient is then pulled towards the unpredictable part, and a
    regression restricted to p directions cannot undo it outside its own span. How much a
    coordinate mixes the two differs between autoencoders, so that bias would differ between the
    pipelines being compared.
    """

    def __init__(self, features: np.ndarray, current: np.ndarray, target: np.ndarray,
                 chunk: int = 2048):
        F = np.ascontiguousarray(features, dtype=np.float64)
        n, p = F.shape
        d = current.shape[1]
        cross_current = np.zeros((p, d), dtype=np.float64)
        cross_target = np.zeros((p, d), dtype=np.float64)
        self._own = np.zeros(d, dtype=np.float64)
        self._cross = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk):
            block = slice(start, start + chunk)
            x = current[block].astype(np.float64)
            y = target[block].astype(np.float64)
            cross_current += F[block].T @ x
            cross_target += F[block].T @ y
            self._own += np.einsum("ij,ij->j", x, x)
            self._cross += np.einsum("ij,ij->j", x, y)
        values, vectors = np.linalg.eigh(F.T @ F)
        self.n_rows = n
        self._values = np.clip(values, 0.0, None)
        self._vectors = vectors
        self._rotated_current = vectors.T @ cross_current
        self._rotated_target = vectors.T @ cross_target

    def solve(self, penalty: float) -> tuple[np.ndarray, np.ndarray]:
        """Joint least squares: returns the own-lag coefficients (d,) and the weights (p, d)."""
        scaled = 1.0 / (self._values + penalty)
        current = self._rotated_current * scaled[:, None]
        target = self._rotated_target * scaled[:, None]
        explained_own = np.einsum("pd,pd->d", self._rotated_current, current)
        explained_cross = np.einsum("pd,pd->d", self._rotated_current, target)
        # What is left of a coordinate's own previous value after the features have taken their
        # part. The floor only guards a coordinate that the features already explain completely.
        remainder = np.maximum(self._own - explained_own, 1e-8 * np.maximum(self._own, 1e-12))
        beta = (self._cross - explained_cross) / remainder
        weights = self._vectors @ (target - current * beta)
        return beta, weights


def standardisation(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and scale of the ridge features. A constant column keeps scale 1, so it stays zero
    after centring and contributes nothing instead of dividing by zero (three of the eleven action
    components are always zero in Bridge, and a fourth is the null-action flag)."""
    mean = F.mean(axis=0)
    scale = F.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, scale


# ---------------------------------------------------------------------------
# Geometry scalars
# ---------------------------------------------------------------------------


def radial_profile(power: np.ndarray) -> np.ndarray:
    """Mean Fourier amplitude over channels in integer radial bins 1..n/2, as a share of their sum.

    ``power`` is (channels, height, width // 2 + 1), the per-channel amplitude of the half-spectrum
    averaged over frames. The constant term is excluded.
    """
    _, height, width_r = power.shape
    fy = np.fft.fftfreq(height) * height
    fx = np.arange(width_r)
    r = np.rint(np.hypot(fy[:, None], fx[None, :])).astype(int)
    amplitude = power.mean(0)
    profile = np.array([amplitude[r == k].mean() for k in range(1, height // 2 + 1)])
    return profile / profile.sum()


def amplitude_spectrum(maps: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Per-channel half-spectrum amplitude averaged over frames, (channels, height, width // 2 + 1).

    ``maps`` is (n, height, width, channels).
    """
    n, height, width, channels = maps.shape
    total = np.zeros((channels, height, width // 2 + 1), dtype=np.float64)
    for start in range(0, n, chunk):
        block = maps[start : start + chunk].astype(np.float32)
        magnitude = np.abs(np.fft.rfft2(np.moveaxis(block, 3, 1), axes=(2, 3)))
        total += magnitude.sum(axis=0, dtype=np.float64)
    return total / n


def turning_angles_tensor(x: torch.Tensor) -> torch.Tensor:
    """Angles between successive displacements along each trajectory, in radians in [0, pi].

    ``x`` is (n, length, d); the return is (n, length - 2), on the device and in the dtype of ``x``.
    """
    v = x[:, 1:] - x[:, :-1]                          # (n, length - 1, d)
    v_a = v[:, :-1]                                   # (n, length - 2, d)
    v_b = v[:, 1:]                                    # (n, length - 2, d)
    cos = (v_a * v_b).sum(-1) / (
        v_a.norm(dim=-1).clamp_min(1e-12) * v_b.norm(dim=-1).clamp_min(1e-12)
    )
    return torch.arccos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def turning_angles(trajectories: np.ndarray) -> np.ndarray:
    """`turning_angles_tensor` of a NumPy array, computed on the CPU."""
    return turning_angles_tensor(torch.as_tensor(trajectories)).numpy()


def _distance_summary(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    return {
        "median": float(np.median(distances)),
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "count": int(len(distances)),
    }


def within_across_distance(
    embeddings_flat: np.ndarray, video_ids: np.ndarray, max_pairs: int = 5000, seed: int = 42
) -> dict:
    """Euclidean distances between frames of the same video and of different videos.

    Within-video distances include every pair of frames of a video. Across-video distances use up
    to ``max_pairs`` pairs of frames from different videos, drawn with randomness fixed by
    ``seed``. Each set is summarised by its median, mean, standard deviation, extremes and count.
    """
    emb = torch.as_tensor(np.ascontiguousarray(embeddings_flat))  # (N, D)
    vids = np.asarray(video_ids)                                  # (N,)
    rng = np.random.default_rng(seed)

    within_dists: list[torch.Tensor] = []
    for vid in np.unique(vids):
        idx = np.where(vids == vid)[0]
        if len(idx) < 2:
            continue
        ev = emb[idx]                               # (n, D)
        dists = (ev.unsqueeze(0) - ev.unsqueeze(1)).norm(dim=-1)  # (n, n)
        triu = torch.triu_indices(len(idx), len(idx), offset=1)
        within_dists.append(dists[triu[0], triu[1]])
    within_arr = torch.cat(within_dists).numpy() if within_dists else np.array([])

    # Oversample, then keep the first max_pairs pairs that join two different videos.
    n = len(vids)
    oversample = min(max_pairs * 6, n * (n - 1))
    candidates = rng.integers(0, n, size=(oversample, 2))
    valid = (vids[candidates[:, 0]] != vids[candidates[:, 1]]) & (
        candidates[:, 0] != candidates[:, 1]
    )
    candidates = candidates[valid][:max_pairs]
    i1 = torch.tensor(candidates[:, 0], dtype=torch.long)
    i2 = torch.tensor(candidates[:, 1], dtype=torch.long)
    across_arr = (emb[i1] - emb[i2]).norm(dim=-1).numpy()

    within, across = _distance_summary(within_arr), _distance_summary(across_arr)
    out = {f"distance_within_{k}": v for k, v in within.items()}
    out.update({f"distance_across_{k}": v for k, v in across.items()})
    out["distance_within_across_mean_ratio"] = (
        within["mean"] / across["mean"] if across["mean"] else float("nan")
    )
    out["distance_within_across_median_ratio"] = (
        within["median"] / across["median"] if across["median"] else float("nan")
    )
    return out
