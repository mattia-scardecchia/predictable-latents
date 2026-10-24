"""Registry of the autoencoder fine-tuning objectives.

Each concrete class declares ``wrapper_cls``, the AE wrapper class for world-model training, and is
registered under its class name with the @register decorator; a checkpoint records that name.

Class names and the paper's objectives. "Hierarchical" classes use the
past-conditioned decoder, which reconstructs a frame from its latent and the pixels
of an earlier frame (the preceding frame in training, the first context frame in
world-model rollouts). The other classes use the pretrained autoencoder's ordinary
decoder. A temporal weight of 0 leaves reconstruction as the only objective.
    LitTrueJEPA                       latent future prediction
    LitPixelPredictionJEPA            pixel future prediction
    LitContrastiveJEPA                contrastive predictive coding
    LitHierarchicalJEPA               latent future prediction, past-conditioned decoder
    LitHierarchicalInverseDynamics    inverse dynamics, past-conditioned decoder
"""

from __future__ import annotations

_BY_CLASS_NAME: dict[str, type] = {}  # "littruejepa" → class


def register(cls):
    """Class decorator that registers a JEPA class by its class name."""
    name = cls.__name__.lower()
    if name in _BY_CLASS_NAME:
        raise ValueError(
            f"Duplicate class name registration: {name!r} "
            f"(existing: {_BY_CLASS_NAME[name].__name__}, new: {cls.__name__})"
        )
    _BY_CLASS_NAME[name] = cls
    return cls


def get_by_class_name(name: str) -> type:
    """Look up a JEPA class by its class name (case-insensitive)."""
    key = name.lower()
    if key not in _BY_CLASS_NAME:
        available = ", ".join(sorted(_BY_CLASS_NAME.keys()))
        raise KeyError(
            f"Unknown JEPA class name: {name!r}. Available: {available}"
        )
    return _BY_CLASS_NAME[key]
