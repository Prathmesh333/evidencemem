"""Shared experiment primitives used by scripts and the Colab notebook.

The reusable package owns prototype construction and scoring. Keeping these
operations here prevents the notebook from drifting into a second, subtly
different implementation of the method.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .memory import EvidenceMemory, ReliabilityWeights, SelectionConfig
from .utils import FloatArray, cosine_to_unit, normalize_rows

IntArray = NDArray[np.int64]
PrototypeMethod = Literal[
    "random",
    "centroid",
    "kmeans_medoids",
    "facility_no_reliability",
    "evidencemem",
]


@dataclass(frozen=True, slots=True)
class PrototypeArrays:
    """Portable array representation of one fitted prototype memory."""

    prototypes: FloatArray
    labels: IntArray
    source_indices: IntArray
    reliabilities: FloatArray
    compactness: FloatArray
    purity: FloatArray
    text_alignment: FloatArray
    method: str
    budget_per_class: int
    seed: int

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            prototypes=self.prototypes,
            labels=self.labels,
            source_indices=self.source_indices,
            reliabilities=self.reliabilities,
            compactness=self.compactness,
            purity=self.purity,
            text_alignment=self.text_alignment,
            method=np.array(self.method, dtype=np.str_),
            budget_per_class=np.array(self.budget_per_class, dtype=np.int64),
            seed=np.array(self.seed, dtype=np.int64),
        )
        temporary.replace(destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PrototypeArrays:
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(
                prototypes=np.asarray(archive["prototypes"], dtype=np.float32),
                labels=np.asarray(archive["labels"], dtype=np.int64),
                source_indices=np.asarray(archive["source_indices"], dtype=np.int64),
                reliabilities=np.asarray(archive["reliabilities"], dtype=np.float32),
                compactness=np.asarray(archive["compactness"], dtype=np.float32),
                purity=np.asarray(archive["purity"], dtype=np.float32),
                text_alignment=np.asarray(archive["text_alignment"], dtype=np.float32),
                method=str(archive["method"]),
                budget_per_class=int(archive["budget_per_class"]),
                seed=int(archive["seed"]),
            )


def reweight_prototype_reliability(
    memory: PrototypeArrays,
    weights: ReliabilityWeights,
    *,
    minimum: float = 0.05,
) -> PrototypeArrays:
    """Recompute reliability on compatible unit scales.

    Compactness and text alignment are cosine similarities in ``[-1, 1]``;
    purity already lies in ``[0, 1]``. Missing text alignment is excluded from
    the active weight for that prototype, matching :class:`EvidenceMemory`.
    """
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("minimum reliability must lie in [0, 1]")

    compactness = np.asarray(cosine_to_unit(memory.compactness), dtype=np.float32)
    purity = np.clip(np.asarray(memory.purity, dtype=np.float32), 0.0, 1.0)
    alignment_raw = np.asarray(memory.text_alignment, dtype=np.float32)
    alignment_present = np.isfinite(alignment_raw)
    alignment = np.asarray(
        cosine_to_unit(np.nan_to_num(alignment_raw, nan=0.0)), dtype=np.float32
    )

    numerator = weights.compactness * compactness + weights.purity * purity
    denominator = np.full(
        len(memory.prototypes),
        weights.compactness + weights.purity,
        dtype=np.float32,
    )
    numerator += weights.text_alignment * alignment * alignment_present
    denominator += weights.text_alignment * alignment_present
    reliability = numerator / np.clip(denominator, 1e-12, None)
    reliability = np.clip(reliability, minimum, 1.0).astype(np.float32)
    return replace(memory, reliabilities=reliability)


def _mapping_from_matrix(values: ArrayLike | None) -> dict[int, FloatArray]:
    if values is None:
        return {}
    matrix = normalize_rows(values, name="text prototypes")
    return {index: matrix[index] for index in range(matrix.shape[0])}


def _arrays_from_memory(
    memory: EvidenceMemory,
    *,
    method: str,
    budget_per_class: int,
    seed: int,
    use_reliability: bool,
) -> PrototypeArrays:
    alignments = np.array(
        [
            np.nan if prototype.text_alignment is None else prototype.text_alignment
            for prototype in memory.prototypes
        ],
        dtype=np.float32,
    )
    return PrototypeArrays(
        prototypes=np.stack([prototype.vector for prototype in memory.prototypes]),
        labels=np.array([prototype.class_id for prototype in memory.prototypes], dtype=np.int64),
        source_indices=np.array(
            [int(prototype.sample_id) for prototype in memory.prototypes], dtype=np.int64
        ),
        reliabilities=np.array(
            [prototype.reliability if use_reliability else 1.0 for prototype in memory.prototypes],
            dtype=np.float32,
        ),
        compactness=np.array(
            [prototype.compactness for prototype in memory.prototypes], dtype=np.float32
        ),
        purity=np.array([prototype.purity for prototype in memory.prototypes], dtype=np.float32),
        text_alignment=alignments,
        method=method,
        budget_per_class=budget_per_class,
        seed=seed,
    )


def fit_prototype_memory(
    embeddings: ArrayLike,
    labels: ArrayLike,
    *,
    budget_per_class: int,
    method: PrototypeMethod,
    seed: int,
    text_prototypes: ArrayLike | None = None,
    reliability_weights: ReliabilityWeights | None = None,
    selection_config: SelectionConfig | None = None,
    purity_k: int = 32,
) -> PrototypeArrays:
    """Fit a matched-memory baseline or EvidenceMem using one implementation."""
    matrix = normalize_rows(embeddings, name="embeddings")
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.shape != (matrix.shape[0],):
        raise ValueError("labels must match embeddings")
    if budget_per_class < 1:
        raise ValueError("budget_per_class must be positive")

    class_ids = sorted(int(value) for value in np.unique(label_array))
    class_names = {class_id: str(class_id) for class_id in class_ids}
    text_mapping = _mapping_from_matrix(text_prototypes)

    if method in {"random", "centroid"}:
        selected: list[int] = []
        for class_id in class_ids:
            local = np.flatnonzero(label_array == class_id)
            count = min(budget_per_class, len(local))
            if method == "random":
                rng = np.random.default_rng(seed + class_id * 1009)
                chosen = rng.choice(local, size=count, replace=False)
            else:
                center = normalize_rows(matrix[local].mean(axis=0, keepdims=True))[0]
                chosen = np.array([local[int(np.argmax(matrix[local] @ center))]])
            selected.extend(int(index) for index in chosen)
        indices = np.array(selected, dtype=np.int64)
        count = len(indices)
        return PrototypeArrays(
            prototypes=matrix[indices],
            labels=label_array[indices],
            source_indices=indices,
            reliabilities=np.ones(count, dtype=np.float32),
            compactness=np.ones(count, dtype=np.float32),
            purity=np.ones(count, dtype=np.float32),
            text_alignment=np.full(count, np.nan, dtype=np.float32),
            method=method,
            budget_per_class=1 if method == "centroid" else budget_per_class,
            seed=seed,
        )

    if method == "kmeans_medoids":
        resolved_selection = SelectionConfig(
            strategy="kmeans_medoids",
            candidate_multiplier=1.0,
            coverage_weight=1.0,
            reliability_weight=0.0,
        )
        use_reliability = False
    elif method == "facility_no_reliability":
        resolved_selection = SelectionConfig(
            strategy="reliability_facility",
            candidate_multiplier=(selection_config or SelectionConfig()).candidate_multiplier,
            coverage_weight=1.0,
            reliability_weight=0.0,
        )
        use_reliability = False
    else:
        resolved_selection = selection_config or SelectionConfig()
        use_reliability = True

    memory = EvidenceMemory(
        index_backend="auto",
        reliability_weights=reliability_weights,
        selection_config=resolved_selection,
    ).build(
        matrix,
        label_array,
        class_names,
        prototypes_per_class=budget_per_class,
        sample_ids=[str(index) for index in range(len(label_array))],
        text_prototypes=text_mapping,
        purity_k=purity_k,
        duplicate_threshold=None,
        random_state=seed,
    )
    return _arrays_from_memory(
        memory,
        method=method,
        budget_per_class=budget_per_class,
        seed=seed,
        use_reliability=use_reliability,
    )


def exact_search(
    vectors: ArrayLike, queries: ArrayLike, k: int
) -> tuple[FloatArray, IntArray]:
    """Search normalized vectors with FAISS when present and NumPy otherwise."""
    matrix = normalize_rows(vectors, name="vectors")
    query_matrix = normalize_rows(queries, name="queries")
    if matrix.shape[1] != query_matrix.shape[1]:
        raise ValueError("query and vector dimensions do not match")
    if k < 1:
        raise ValueError("k must be positive")
    k_eff = min(k, matrix.shape[0])
    try:
        import faiss  # type: ignore[import-not-found]

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        similarities, indices = index.search(query_matrix, k_eff)
        return (
            np.asarray(similarities, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
        )
    except ImportError:
        scores = query_matrix @ matrix.T
        candidates = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        candidate_scores = np.take_along_axis(scores, candidates, axis=1)
        order = np.argsort(-candidate_scores, axis=1, kind="stable")
        return (
            np.take_along_axis(candidate_scores, order, axis=1).astype(np.float32),
            np.take_along_axis(candidates, order, axis=1).astype(np.int64),
        )


def visual_class_scores(
    memory: PrototypeArrays,
    queries: ArrayLike,
    *,
    k: int,
    n_classes: int,
    temperature: float = 0.07,
) -> FloatArray:
    """Return normalized reliability-weighted retrieval votes."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    similarities, indices = exact_search(memory.prototypes, queries, k)
    retrieved_labels = memory.labels[indices]
    retrieved_reliability = memory.reliabilities[indices]
    weights = np.exp((similarities - similarities.max(axis=1, keepdims=True)) / temperature)
    weights *= np.maximum(retrieved_reliability, 1e-8)
    scores = np.zeros((len(indices), n_classes), dtype=np.float32)
    rows = np.repeat(np.arange(len(indices)), indices.shape[1])
    np.add.at(scores, (rows, retrieved_labels.ravel()), weights.ravel())
    return scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)


def class_conditional_visual_scores(
    memory: PrototypeArrays,
    queries: ArrayLike,
    *,
    k_per_class: int,
    n_classes: int,
    temperature: float = 0.07,
    reliability_power: float = 1.0,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Score each class from its own best prototypes using log-mean-exp.

    Global top-k voting can ignore a class entirely when a salient foreground
    object dominates the neighbours. This scorer retrieves an equal top-k set
    inside every class, aggregates similarity and reliability, and returns the
    selected prototype indices for evidence auditing.
    """
    if k_per_class < 1:
        raise ValueError("k_per_class must be positive")
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if reliability_power < 0:
        raise ValueError("reliability_power must be non-negative")
    if np.any(memory.labels < 0) or np.any(memory.labels >= n_classes):
        raise ValueError("memory labels must lie in [0, n_classes)")

    query_matrix = normalize_rows(queries, name="queries")
    class_logits = np.full((len(query_matrix), n_classes), -np.inf, dtype=np.float32)
    selected = np.full(
        (len(query_matrix), n_classes, k_per_class), -1, dtype=np.int64
    )
    for class_id in range(n_classes):
        class_indices = np.flatnonzero(memory.labels == class_id)
        if not len(class_indices):
            continue
        similarities, local_indices = exact_search(
            memory.prototypes[class_indices], query_matrix, k_per_class
        )
        global_indices = class_indices[local_indices]
        selected[:, class_id, : global_indices.shape[1]] = global_indices
        reliability = np.clip(memory.reliabilities[global_indices], 1e-8, 1.0)
        log_weights = similarities / temperature
        if reliability_power:
            log_weights += reliability_power * np.log(reliability)
        row_max = log_weights.max(axis=1, keepdims=True)
        class_logits[:, class_id] = (
            row_max[:, 0]
            + np.log(np.exp(log_weights - row_max).mean(axis=1))
        )

    if np.any(~np.isfinite(class_logits).all(axis=1)):
        missing = np.flatnonzero(~np.isfinite(class_logits).any(axis=0)).tolist()
        raise ValueError(f"memory has no prototypes for classes: {missing}")
    class_logits -= class_logits.max(axis=1, keepdims=True)
    probabilities = np.exp(class_logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(probabilities, dtype=np.float32), selected


def text_class_scores(
    queries: ArrayLike, text_prototypes: ArrayLike, *, temperature: float = 0.07
) -> FloatArray:
    """Return temperature-scaled CLIP class probabilities."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query_matrix = normalize_rows(queries, name="queries")
    text_matrix = normalize_rows(text_prototypes, name="text prototypes")
    logits = query_matrix @ text_matrix.T / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return np.asarray(
        probabilities / probabilities.sum(axis=1, keepdims=True), dtype=np.float32
    )


def fused_class_scores(
    memory: PrototypeArrays,
    queries: ArrayLike,
    text_prototypes: ArrayLike,
    *,
    text_weight: float,
    k: int,
    visual_temperature: float = 0.07,
    text_temperature: float = 0.07,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Fuse visual and textual probabilities using an unambiguous text weight."""
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must lie in [0, 1]")
    text = text_class_scores(queries, text_prototypes, temperature=text_temperature)
    visual = visual_class_scores(
        memory,
        queries,
        k=k,
        n_classes=text.shape[1],
        temperature=visual_temperature,
    )
    fused = (1.0 - text_weight) * visual + text_weight * text
    return np.asarray(fused, dtype=np.float32), visual, text


def weighted_knn_scores(
    train_embeddings: ArrayLike,
    train_labels: ArrayLike,
    queries: ArrayLike,
    *,
    k: int,
    n_classes: int,
    temperature: float = 0.07,
) -> FloatArray:
    """Conventional similarity-weighted kNN probability scores."""
    labels = np.asarray(train_labels, dtype=np.int64)
    similarities, indices = exact_search(train_embeddings, queries, k)
    weights = np.exp((similarities - similarities.max(axis=1, keepdims=True)) / temperature)
    scores = np.zeros((len(indices), n_classes), dtype=np.float32)
    rows = np.repeat(np.arange(len(indices)), indices.shape[1])
    np.add.at(scores, (rows, labels[indices].ravel()), weights.ravel())
    return scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)


def tip_adapter_scores(
    cache: PrototypeArrays,
    queries: ArrayLike,
    text_prototypes: ArrayLike,
    *,
    beta: float,
    cache_weight: float,
    clip_logit_scale: float = 100.0,
) -> FloatArray:
    """Training-free Tip-Adapter logits at a matched cache budget."""
    if beta <= 0 or cache_weight < 0 or clip_logit_scale <= 0:
        raise ValueError(
            "beta and clip_logit_scale must be positive; cache_weight cannot be negative"
        )
    query_matrix = normalize_rows(queries, name="queries")
    text_matrix = normalize_rows(text_prototypes, name="text prototypes")
    affinity = query_matrix @ cache.prototypes.T
    cache_affinity = np.exp(-beta * (1.0 - affinity))
    one_hot = np.eye(text_matrix.shape[0], dtype=np.float32)[cache.labels]
    cache_logits = cache_affinity @ one_hot
    text_logits = clip_logit_scale * (query_matrix @ text_matrix.T)
    return np.asarray(text_logits + cache_weight * cache_logits, dtype=np.float32)
