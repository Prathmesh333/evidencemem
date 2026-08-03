"""Replaceable exact inner-product indices for normalized embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .utils import FloatArray, normalize_rows

IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class IndexSearch:
    similarities: FloatArray
    indices: IntArray


class VectorIndex(Protocol):
    @property
    def size(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def build(self, vectors: ArrayLike) -> None: ...

    def search(self, queries: ArrayLike, k: int) -> IndexSearch: ...


class ExactNumpyIndex:
    """Deterministic cosine search using normalized NumPy matrices."""

    def __init__(self) -> None:
        self._vectors: FloatArray | None = None

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def dimension(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[1])

    def build(self, vectors: ArrayLike) -> None:
        self._vectors = normalize_rows(vectors, name="index vectors")

    def search(self, queries: ArrayLike, k: int) -> IndexSearch:
        if self._vectors is None:
            raise RuntimeError("index has not been built")
        if k < 1:
            raise ValueError("k must be at least one")
        query_matrix = normalize_rows(queries, name="queries")
        if query_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {query_matrix.shape[1]} does not match index {self.dimension}"
            )
        k_eff = min(k, self.size)
        scores = query_matrix @ self._vectors.T
        candidates = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)

        ordered_indices = np.empty_like(candidates, dtype=np.int64)
        ordered_scores = np.empty_like(candidate_scores, dtype=np.float32)
        for row in range(query_matrix.shape[0]):
            order = np.lexsort((candidates[row], -candidate_scores[row]))
            ordered_indices[row] = candidates[row, order]
            ordered_scores[row] = candidate_scores[row, order]
        return IndexSearch(ordered_scores, ordered_indices)


class FaissFlatIPIndex:
    """FAISS exact inner-product search, imported only when available."""

    def __init__(self) -> None:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("FAISS is not installed; use backend='numpy'") from exc
        self._faiss = faiss
        self._index = None
        self._dimension = 0

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return self._dimension

    def build(self, vectors: ArrayLike) -> None:
        matrix = normalize_rows(vectors, name="index vectors")
        self._dimension = int(matrix.shape[1])
        self._index = self._faiss.IndexFlatIP(self._dimension)
        self._index.add(matrix)

    def search(self, queries: ArrayLike, k: int) -> IndexSearch:
        if self._index is None:
            raise RuntimeError("index has not been built")
        if k < 1:
            raise ValueError("k must be at least one")
        query_matrix = normalize_rows(queries, name="queries")
        if query_matrix.shape[1] != self.dimension:
            raise ValueError(
                f"query dimension {query_matrix.shape[1]} does not match index {self.dimension}"
            )
        similarities, indices = self._index.search(query_matrix, min(k, self.size))
        return IndexSearch(
            np.asarray(similarities, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
        )


def make_index(backend: str = "auto") -> VectorIndex:
    """Construct an exact index; prefer FAISS when requested and available."""
    normalized = backend.lower()
    if normalized == "numpy":
        return ExactNumpyIndex()
    if normalized == "faiss":
        return FaissFlatIPIndex()
    if normalized == "auto":
        try:
            return FaissFlatIPIndex()
        except ImportError:
            return ExactNumpyIndex()
    raise ValueError("backend must be one of: auto, faiss, numpy")
