"""Crash-safe embedding caches with provenance manifests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import FloatArray, as_float_matrix


@dataclass(frozen=True, slots=True)
class EmbeddingCacheManifest:
    schema_version: int
    dataset: str
    split: str
    model_name: str
    pretrained: str
    preprocess_fingerprint: str
    source_revision: str
    sample_count: int
    dimension: int
    dtype: str
    normalized: bool
    embeddings_sha256: str
    labels_sha256: str
    sample_ids_sha256: str


@dataclass(frozen=True, slots=True)
class EmbeddingCache:
    embeddings: FloatArray
    labels: NDArray[np.int64]
    sample_ids: NDArray[np.str_]
    manifest: EmbeddingCacheManifest


def hash_sample_ids(sample_ids: ArrayLike) -> str:
    values = np.asarray(sample_ids, dtype=np.str_)
    joined = "\0".join(values.tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def hash_array(values: ArrayLike) -> str:
    """Hash an array's dtype, shape, and contiguous byte representation."""
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def save_embedding_cache(
    path: str | Path,
    embeddings: ArrayLike,
    labels: ArrayLike,
    sample_ids: ArrayLike,
    *,
    dataset: str,
    split: str,
    model_name: str,
    pretrained: str,
    preprocess_fingerprint: str = "unspecified",
    source_revision: str = "unknown",
) -> EmbeddingCacheManifest:
    """Atomically write an embedding archive and adjacent JSON manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = as_float_matrix(embeddings, name="embeddings")
    label_array = np.asarray(labels, dtype=np.int64)
    id_array = np.asarray(sample_ids, dtype=np.str_)
    if label_array.shape != (matrix.shape[0],) or id_array.shape != (matrix.shape[0],):
        raise ValueError("labels and sample_ids must match the embedding count")

    norms = np.linalg.norm(matrix, axis=1)
    normalized = bool(np.allclose(norms, 1.0, atol=1e-4))
    manifest = EmbeddingCacheManifest(
        schema_version=2,
        dataset=dataset,
        split=split,
        model_name=model_name,
        pretrained=pretrained,
        preprocess_fingerprint=preprocess_fingerprint,
        source_revision=source_revision,
        sample_count=int(matrix.shape[0]),
        dimension=int(matrix.shape[1]),
        dtype=str(matrix.dtype),
        normalized=normalized,
        embeddings_sha256=hash_array(matrix),
        labels_sha256=hash_array(label_array),
        sample_ids_sha256=hash_sample_ids(id_array),
    )

    archive_tmp = destination.with_name(destination.stem + ".tmp.npz")
    np.savez_compressed(
        archive_tmp,
        embeddings=matrix,
        labels=label_array,
        sample_ids=id_array,
    )
    archive_tmp.replace(destination)

    manifest_path = destination.with_suffix(".json")
    manifest_tmp = manifest_path.with_name(manifest_path.stem + ".tmp.json")
    manifest_tmp.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    manifest_tmp.replace(manifest_path)
    return manifest


def load_embedding_cache(path: str | Path, *, require_normalized: bool = True) -> EmbeddingCache:
    """Load a cache and verify its shape, sample identity, and normalization."""
    source = Path(path)
    manifest_data: dict[str, Any] = json.loads(
        source.with_suffix(".json").read_text(encoding="utf-8")
    )
    if manifest_data.get("schema_version") != 2:
        raise ValueError("unsupported embedding cache manifest; regenerate the cache")
    manifest = EmbeddingCacheManifest(**manifest_data)
    with np.load(source, allow_pickle=False) as archive:
        matrix = as_float_matrix(archive["embeddings"], name="cached embeddings")
        labels = np.asarray(archive["labels"], dtype=np.int64)
        sample_ids = np.asarray(archive["sample_ids"], dtype=np.str_)

    if matrix.shape != (manifest.sample_count, manifest.dimension):
        raise ValueError("embedding cache shape does not match its manifest")
    if labels.shape != (manifest.sample_count,) or sample_ids.shape != (manifest.sample_count,):
        raise ValueError("cache labels or sample_ids do not match its manifest")
    if hash_sample_ids(sample_ids) != manifest.sample_ids_sha256:
        raise ValueError("sample_ids hash does not match the cache manifest")
    if hash_array(matrix) != manifest.embeddings_sha256:
        raise ValueError("embedding hash does not match the cache manifest")
    if hash_array(labels) != manifest.labels_sha256:
        raise ValueError("label hash does not match the cache manifest")
    if str(matrix.dtype) != manifest.dtype:
        raise ValueError("embedding dtype does not match the cache manifest")
    if require_normalized and not manifest.normalized:
        raise ValueError("embedding cache is not L2-normalized")
    return EmbeddingCache(matrix, labels, sample_ids, manifest)
