"""Numerical helpers shared by the memory and classifier."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float32]


def as_float_matrix(values: ArrayLike, *, name: str = "values") -> FloatArray:
    """Return a finite, two-dimensional float32 matrix."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; received shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty; received shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(matrix)


def normalize_rows(values: ArrayLike, *, name: str = "values") -> FloatArray:
    """L2-normalize each row and reject zero-norm vectors."""
    matrix = as_float_matrix(values, name=name)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError(f"{name} contains a zero-norm vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def normalize_vector(value: ArrayLike, *, name: str = "value") -> FloatArray:
    """L2-normalize one vector and return a one-dimensional float32 array."""
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D vector; received {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float32).eps:
        raise ValueError(f"{name} has zero norm")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def softmax(values: ArrayLike, *, temperature: float = 1.0) -> FloatArray:
    """Compute a stable one-dimensional softmax."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("softmax expects a non-empty 1D array")
    shifted = (logits / temperature) - np.max(logits / temperature)
    weights = np.exp(shifted)
    return np.asarray(weights / weights.sum(), dtype=np.float32)


def cosine_to_unit(value: float | np.ndarray) -> float | np.ndarray:
    """Map cosine similarity from [-1, 1] to a bounded [0, 1] score."""
    return np.clip((np.asarray(value) + 1.0) / 2.0, 0.0, 1.0)
