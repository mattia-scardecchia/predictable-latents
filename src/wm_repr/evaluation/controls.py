"""Paired action shuffling and reconstruction ceilings."""
import random
import numpy as np
import torch
from .rollout import run_inference


def _capture_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def derangement(pools: np.ndarray, seed: int) -> np.ndarray:
    """donor[i] = j != i with pools[j] == pools[i]; uniform within each pool, fixed by seed."""
    rng = np.random.default_rng(seed)
    donor = np.empty(len(pools), dtype=np.int64)
    for pool in sorted(set(pools.tolist())):
        members = np.flatnonzero(pools == pool)
        if len(members) < 2:
            raise ValueError(f"Shuffle pool {pool!r} has {len(members)} episode(s); need >= 2")
        while True:  # rejection sampling; accepts with probability about 1/e
            order = rng.permutation(len(members))
            if not np.any(order == np.arange(len(members))):
                break
        donor[members] = members[order]
    return donor


def run_controls(model, loader, *, permutation_seed=0, within_task=False, **kwargs):
    # Build fixed metric networks before pairing the two sampling RNG states.
    if kwargs.get("lpips_fn") is None:
        kwargs["lpips_fn"] = model.lpips
    state = _capture_rng()
    truth, _ = run_inference(model, loader, **kwargs)
    groups = truth["tasks"] if within_task else np.zeros(len(truth["mse"]), dtype=int)
    donors = derangement(groups, permutation_seed)
    _restore_rng(state)
    shuffled, _ = run_inference(model, loader, actions_override=truth["actions_transformed"][donors], **kwargs)
    ceiling, _ = run_inference(model, loader, ceiling=True, **kwargs)
    return {"true_actions": truth, "shuffled_actions": shuffled, "ceiling": ceiling}, donors
