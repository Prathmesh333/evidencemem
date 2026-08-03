"""Deterministic split utilities for image-classification experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.model_selection import train_test_split


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset: str
    seed: int
    sample_count: int
    train_count: int
    validation_count: int
    labels_sha256: str
    indices_sha256: str


def _sha256_array(values: NDArray[np.generic]) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def stratified_train_validation_indices(
    labels: ArrayLike,
    *,
    validation_size: int,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Create stable, sorted train/validation indices with class stratification."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty 1D array")
    if not 0 < validation_size < label_array.size:
        raise ValueError("validation_size must lie strictly between zero and sample count")
    indices = np.arange(label_array.size, dtype=np.int64)
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=label_array,
    )
    return np.sort(train_indices), np.sort(validation_indices)


def save_split(
    directory: str | Path,
    *,
    dataset: str,
    labels: ArrayLike,
    train_indices: ArrayLike,
    validation_indices: ArrayLike,
    seed: int,
) -> SplitManifest:
    """Persist split indices and a hash manifest for later reproducibility checks."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    label_array = np.asarray(labels, dtype=np.int64)
    train_array = np.asarray(train_indices, dtype=np.int64)
    validation_array = np.asarray(validation_indices, dtype=np.int64)
    if np.intersect1d(train_array, validation_array).size:
        raise ValueError("train and validation indices overlap")

    combined = np.concatenate([train_array, validation_array])
    manifest = SplitManifest(
        dataset=dataset,
        seed=seed,
        sample_count=int(label_array.size),
        train_count=int(train_array.size),
        validation_count=int(validation_array.size),
        labels_sha256=_sha256_array(label_array),
        indices_sha256=_sha256_array(combined),
    )
    archive_tmp = destination / "split_indices.tmp.npz"
    archive_path = destination / "split_indices.npz"
    np.savez_compressed(
        archive_tmp,
        train_indices=train_array,
        validation_indices=validation_array,
    )
    archive_tmp.replace(archive_path)

    manifest_tmp = destination / "split_manifest.tmp.json"
    manifest_path = destination / "split_manifest.json"
    manifest_tmp.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest
