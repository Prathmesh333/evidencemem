"""Construction and online maintenance of displayable visual prototypes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from math import ceil
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import MiniBatchKMeans

from .index import VectorIndex, make_index
from .schema import Prototype, SearchResult
from .utils import FloatArray, cosine_to_unit, normalize_rows, normalize_vector


@dataclass(frozen=True, slots=True)
class ReliabilityWeights:
    compactness: float = 0.45
    text_alignment: float = 0.20
    purity: float = 0.35

    def __post_init__(self) -> None:
        if min(self.compactness, self.text_alignment, self.purity) < 0:
            raise ValueError("reliability weights must be non-negative")
        if self.compactness + self.text_alignment + self.purity <= 0:
            raise ValueError("at least one reliability weight must be positive")


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Exact-budget prototype selection settings.

    ``reliability_facility`` first over-clusters each class to obtain real-image
    medoid candidates, then greedily maximizes a monotone submodular objective:
    class coverage (facility location) plus a modular reliability reward.
    ``kmeans_medoids`` retains one medoid from each of exactly ``budget``
    clusters and is the matched plain-medoid baseline.
    """

    strategy: Literal["reliability_facility", "kmeans_medoids"] = (
        "reliability_facility"
    )
    candidate_multiplier: float = 2.0
    coverage_weight: float = 0.75
    reliability_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.strategy not in {"reliability_facility", "kmeans_medoids"}:
            raise ValueError("unknown prototype selection strategy")
        if self.candidate_multiplier < 1.0:
            raise ValueError("candidate_multiplier must be at least one")
        if self.coverage_weight < 0 or self.reliability_weight < 0:
            raise ValueError("selection weights must be non-negative")
        if self.coverage_weight + self.reliability_weight <= 0:
            raise ValueError("at least one selection weight must be positive")


class EvidenceMemory:
    """A bounded collection of class-conditional, real-image medoids."""

    def __init__(
        self,
        prototypes: Sequence[Prototype] | None = None,
        *,
        index_backend: str = "auto",
        reliability_weights: ReliabilityWeights | None = None,
        selection_config: SelectionConfig | None = None,
    ) -> None:
        self.prototypes: list[Prototype] = list(prototypes or [])
        self.class_names: dict[int, str] = {
            prototype.class_id: prototype.class_name for prototype in self.prototypes
        }
        self.index_backend = index_backend
        self.reliability_weights = reliability_weights or ReliabilityWeights()
        self.selection_config = selection_config or SelectionConfig()
        self._index: VectorIndex = make_index(index_backend)
        if self.prototypes:
            self._rebuild_index()

    def __len__(self) -> int:
        return len(self.prototypes)

    @property
    def dimension(self) -> int:
        return 0 if not self.prototypes else int(self.prototypes[0].vector.shape[0])

    def _rebuild_index(self) -> None:
        if not self.prototypes:
            self._index = make_index(self.index_backend)
            return
        vectors = np.stack([prototype.vector for prototype in self.prototypes])
        self._index = make_index(self.index_backend)
        self._index.build(vectors)

    def _combine_reliability(
        self,
        compactness: float,
        purity: float,
        text_alignment: float | None,
    ) -> float:
        weighted = [
            (self.reliability_weights.compactness, float(cosine_to_unit(compactness))),
            (self.reliability_weights.purity, float(np.clip(purity, 0.0, 1.0))),
        ]
        if text_alignment is not None:
            weighted.append(
                (
                    self.reliability_weights.text_alignment,
                    float(cosine_to_unit(text_alignment)),
                )
            )
        active_weight = sum(weight for weight, _ in weighted)
        if active_weight <= 0:
            return 1.0
        return float(sum(weight * value for weight, value in weighted) / active_weight)

    def _select_exact_budget(
        self,
        candidates: Sequence[Prototype],
        class_vectors: FloatArray,
        budget: int,
    ) -> list[Prototype]:
        """Greedily maximize coverage plus reliability at an exact budget."""
        if budget >= len(candidates):
            return list(candidates)
        ordered = sorted(candidates, key=lambda item: item.sample_id)
        candidate_vectors = np.stack([item.vector for item in ordered])
        coverage = np.asarray(
            cosine_to_unit(class_vectors @ candidate_vectors.T), dtype=np.float32
        )
        current = np.zeros(class_vectors.shape[0], dtype=np.float32)
        selected: list[int] = []
        config = self.selection_config
        initial_coverage = coverage.mean(axis=0)
        queue = [
            (
                -(
                    config.coverage_weight * float(initial_coverage[index])
                    + config.reliability_weight * ordered[index].reliability / budget
                ),
                index,
                0,
            )
            for index in range(len(ordered))
        ]
        heapify(queue)

        for iteration in range(budget):
            while queue:
                _, candidate_index, evaluated_at = heappop(queue)
                if evaluated_at == iteration:
                    selected.append(candidate_index)
                    current = np.maximum(current, coverage[:, candidate_index])
                    break
                proposed = np.maximum(current, coverage[:, candidate_index])
                coverage_gain = float(np.mean(proposed - current))
                reliability_gain = ordered[candidate_index].reliability / budget
                gain = (
                    config.coverage_weight * coverage_gain
                    + config.reliability_weight * reliability_gain
                )
                heappush(queue, (-gain, candidate_index, iteration))
            else:
                raise RuntimeError("candidate queue was exhausted before reaching the budget")
        return [ordered[index] for index in selected]

    @staticmethod
    def _purities_at_medoids(
        index: VectorIndex,
        labels: NDArray[np.int64],
        medoid_indices: Sequence[int],
        class_id: int,
        k: int,
        vectors: FloatArray,
    ) -> list[float]:
        if index.size == 1:
            return [1.0] * len(medoid_indices)
        k_eff = min(max(k, 1), index.size - 1)
        medoid_array = np.asarray(medoid_indices, dtype=np.int64)
        result = index.search(vectors[medoid_array], min(index.size, k_eff + 1))
        purities: list[float] = []
        for medoid_index, neighbor_row in zip(medoid_array, result.indices, strict=True):
            neighbors = neighbor_row[neighbor_row != medoid_index][:k_eff]
            purities.append(
                float(np.mean(labels[neighbors] == class_id)) if len(neighbors) else 1.0
            )
        return purities

    def build(
        self,
        embeddings: ArrayLike,
        labels: ArrayLike,
        class_names: Mapping[int, str],
        *,
        prototypes_per_class: int,
        sample_ids: Sequence[str] | None = None,
        image_paths: Sequence[str | None] | None = None,
        text_prototypes: Mapping[int, ArrayLike] | None = None,
        purity_k: int = 10,
        duplicate_threshold: float | None = None,
        random_state: int = 0,
    ) -> EvidenceMemory:
        """Build class-wise KMeans medoids and compute their reliability."""
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be at least one")
        if duplicate_threshold is not None and not -1.0 <= duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must lie in [-1, 1]")

        matrix = normalize_rows(embeddings, name="embeddings")
        label_array = np.asarray(labels, dtype=np.int64)
        if label_array.ndim != 1 or label_array.shape[0] != matrix.shape[0]:
            raise ValueError("labels must be a 1D array matching the embedding count")
        count = matrix.shape[0]
        ids = list(sample_ids) if sample_ids is not None else [str(i) for i in range(count)]
        paths = list(image_paths) if image_paths is not None else [None] * count
        if len(ids) != count or len(paths) != count:
            raise ValueError("sample_ids and image_paths must match the embedding count")

        text_vectors = {
            int(class_id): normalize_vector(vector, name=f"text prototype {class_id}")
            for class_id, vector in (text_prototypes or {}).items()
        }
        purity_index = make_index(self.index_backend)
        purity_index.build(matrix)
        built: list[Prototype] = []
        for class_id in sorted(int(value) for value in np.unique(label_array)):
            if class_id not in class_names:
                raise ValueError(f"missing class name for class id {class_id}")
            global_indices = np.flatnonzero(label_array == class_id)
            class_vectors = matrix[global_indices]
            requested = min(prototypes_per_class, class_vectors.shape[0])
            if self.selection_config.strategy == "reliability_facility":
                expanded_budget = int(
                    ceil(requested * self.selection_config.candidate_multiplier)
                )
                cluster_count = min(
                    class_vectors.shape[0],
                    max(requested, expanded_budget),
                )
            else:
                cluster_count = requested

            if cluster_count == 1:
                assignments = np.zeros(class_vectors.shape[0], dtype=np.int64)
            else:
                clusterer = MiniBatchKMeans(
                    n_clusters=cluster_count,
                    random_state=random_state,
                    n_init=10,
                    batch_size=min(1024, max(32, class_vectors.shape[0])),
                    reassignment_ratio=0.0,
                )
                assignments = np.asarray(clusterer.fit_predict(class_vectors), dtype=np.int64)

            candidates: list[Prototype] = []
            candidate_indices: list[int] = []
            for cluster_id in range(cluster_count):
                local_members = np.flatnonzero(assignments == cluster_id)
                if local_members.size == 0:
                    continue
                member_vectors = class_vectors[local_members]
                centroid = normalize_vector(member_vectors.mean(axis=0), name="cluster centroid")
                similarities = member_vectors @ centroid
                medoid_local = int(local_members[int(np.argmax(similarities))])
                medoid_global = int(global_indices[medoid_local])
                medoid = matrix[medoid_global]
                compactness = float(np.mean(member_vectors @ medoid))
                text_alignment = (
                    float(medoid @ text_vectors[class_id]) if class_id in text_vectors else None
                )
                candidates.append(
                    Prototype(
                        vector=medoid,
                        centroid=centroid,
                        class_id=class_id,
                        class_name=class_names[class_id],
                        sample_id=str(ids[medoid_global]),
                        image_path=paths[medoid_global],
                        cluster_size=int(local_members.size),
                        reliability=1.0,
                        compactness=compactness,
                        purity=1.0,
                        text_alignment=text_alignment,
                    )
                )
                candidate_indices.append(medoid_global)

            purities = self._purities_at_medoids(
                purity_index,
                label_array,
                candidate_indices,
                class_id,
                purity_k,
                matrix,
            )
            for candidate, purity in zip(candidates, purities, strict=True):
                candidate.purity = purity
                candidate.reliability = self._combine_reliability(
                    candidate.compactness,
                    candidate.purity,
                    candidate.text_alignment,
                )

            if duplicate_threshold is not None:
                candidates.sort(key=lambda item: (-item.reliability, item.sample_id))
                deduplicated: list[Prototype] = []
                for candidate in candidates:
                    similarities = [candidate.vector @ item.vector for item in deduplicated]
                    if not similarities or max(similarities) <= duplicate_threshold:
                        deduplicated.append(candidate)
                candidates = deduplicated

            if self.selection_config.strategy == "reliability_facility":
                selected = self._select_exact_budget(candidates, class_vectors, requested)
            else:
                selected = list(candidates[:requested])
            if len(selected) != requested:
                raise RuntimeError(
                    "prototype selection returned "
                    f"{len(selected)} items for a budget of {requested}; "
                    "disable duplicate filtering or raise its threshold"
                )
            built.extend(selected)

        if not built:
            raise ValueError("memory construction produced no prototypes")
        self.prototypes = built
        self.class_names = {int(key): str(value) for key, value in class_names.items()}
        self._rebuild_index()
        return self

    def search(self, query: ArrayLike, k: int = 10) -> tuple[SearchResult, ...]:
        """Retrieve top-k prototypes and update their usage counters."""
        if not self.prototypes:
            raise RuntimeError("cannot search an empty memory")
        vector = normalize_vector(query, name="query")
        result = self._index.search(vector[None, :], k)
        evidence: list[SearchResult] = []
        for index, similarity in zip(result.indices[0], result.similarities[0], strict=True):
            prototype = self.prototypes[int(index)]
            prototype.usage_count += 1
            evidence.append(
                SearchResult(
                    prototype_index=int(index),
                    class_id=prototype.class_id,
                    class_name=prototype.class_name,
                    sample_id=prototype.sample_id,
                    image_path=prototype.image_path,
                    similarity=float(similarity),
                    reliability=prototype.reliability,
                )
            )
        return tuple(evidence)

    def add_class(
        self,
        class_id: int,
        class_name: str,
        support_embeddings: ArrayLike,
        *,
        prototypes_per_class: int,
        sample_ids: Sequence[str] | None = None,
        image_paths: Sequence[str | None] | None = None,
        text_prototype: ArrayLike | None = None,
        purity_k: int = 10,
        duplicate_threshold: float | None = None,
        random_state: int = 0,
    ) -> int:
        """Insert a previously unseen class without changing existing prototypes."""
        if class_id in self.class_names:
            raise ValueError(f"class id {class_id} already exists")
        matrix = normalize_rows(support_embeddings, name="support_embeddings")
        labels = np.full(matrix.shape[0], class_id, dtype=np.int64)
        temporary = EvidenceMemory(
            index_backend="numpy",
            reliability_weights=self.reliability_weights,
            selection_config=self.selection_config,
        )
        temporary.build(
            matrix,
            labels,
            {class_id: class_name},
            prototypes_per_class=prototypes_per_class,
            sample_ids=sample_ids,
            image_paths=image_paths,
            text_prototypes=({class_id: text_prototype} if text_prototype is not None else None),
            purity_k=purity_k,
            duplicate_threshold=duplicate_threshold,
            random_state=random_state,
        )
        self.prototypes.extend(temporary.prototypes)
        self.class_names[class_id] = class_name
        self._rebuild_index()
        return len(temporary.prototypes)

    def update(
        self,
        embedding: ArrayLike,
        class_id: int,
        *,
        class_name: str | None = None,
        sample_id: str,
        image_path: str | None = None,
        merge_threshold: float = 0.90,
        text_prototype: ArrayLike | None = None,
    ) -> str:
        """Merge a labeled sample into its nearest class prototype or insert it."""
        if not -1.0 <= merge_threshold <= 1.0:
            raise ValueError("merge_threshold must lie in [-1, 1]")
        vector = normalize_vector(embedding, name="embedding")
        text_vector = (
            normalize_vector(text_prototype, name="text_prototype")
            if text_prototype is not None
            else None
        )
        matching = [
            index
            for index, prototype in enumerate(self.prototypes)
            if prototype.class_id == class_id
        ]
        if matching:
            similarities = np.array(
                [self.prototypes[index].vector @ vector for index in matching], dtype=np.float32
            )
            target_index = matching[int(np.argmax(similarities))]
            similarity = float(np.max(similarities))
            if similarity >= merge_threshold:
                prototype = self.prototypes[target_index]
                new_count = prototype.cluster_size + 1
                centroid = normalize_vector(
                    prototype.centroid * prototype.cluster_size + vector,
                    name="updated centroid",
                )
                if float(vector @ centroid) > float(prototype.vector @ centroid):
                    prototype.vector = vector
                    prototype.sample_id = sample_id
                    prototype.image_path = image_path
                prototype.centroid = centroid
                prototype.cluster_size = new_count
                prototype.compactness = (
                    prototype.compactness * (new_count - 1) + similarity
                ) / new_count
                if text_vector is not None:
                    prototype.text_alignment = float(prototype.vector @ text_vector)
                prototype.reliability = self._combine_reliability(
                    prototype.compactness,
                    prototype.purity,
                    prototype.text_alignment,
                )
                self._rebuild_index()
                return "merged"

        resolved_name = class_name or self.class_names.get(class_id)
        if resolved_name is None:
            raise ValueError("class_name is required when inserting a new class id")
        alignment = float(vector @ text_vector) if text_vector is not None else None
        reliability = self._combine_reliability(1.0, 1.0, alignment)
        self.prototypes.append(
            Prototype(
                vector=vector,
                centroid=vector,
                class_id=class_id,
                class_name=resolved_name,
                sample_id=sample_id,
                image_path=image_path,
                reliability=reliability,
                compactness=1.0,
                purity=1.0,
                text_alignment=alignment,
            )
        )
        self.class_names[class_id] = resolved_name
        self._rebuild_index()
        return "inserted"

    def prune(self, max_size: int, *, min_per_class: int = 1) -> list[Prototype]:
        """Remove least reliable, least used prototypes while preserving class floors."""
        if max_size < 1 or min_per_class < 1:
            raise ValueError("max_size and min_per_class must be positive")
        minimum_required = min_per_class * len(self.class_names)
        if max_size < minimum_required:
            raise ValueError(
                f"max_size={max_size} cannot preserve {min_per_class} prototype(s) "
                f"for each of {len(self.class_names)} classes"
            )
        removed: list[Prototype] = []
        while len(self.prototypes) > max_size:
            class_counts = Counter(item.class_id for item in self.prototypes)
            eligible = [
                (index, item)
                for index, item in enumerate(self.prototypes)
                if class_counts[item.class_id] > min_per_class
            ]
            if not eligible:
                break
            remove_index, _ = min(
                eligible,
                key=lambda pair: (
                    pair[1].reliability,
                    pair[1].usage_count,
                    pair[1].cluster_size,
                    pair[1].sample_id,
                ),
            )
            removed.append(self.prototypes.pop(remove_index))
        self._rebuild_index()
        return removed

    def save(self, path: str | Path) -> Path:
        """Persist prototypes in a portable compressed NumPy archive."""
        if not self.prototypes:
            raise RuntimeError("cannot save an empty memory")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            vectors=np.stack([item.vector for item in self.prototypes]),
            centroids=np.stack([item.centroid for item in self.prototypes]),
            class_ids=np.array([item.class_id for item in self.prototypes], dtype=np.int64),
            class_names=np.array([item.class_name for item in self.prototypes], dtype=np.str_),
            sample_ids=np.array([item.sample_id for item in self.prototypes], dtype=np.str_),
            image_paths=np.array(
                [item.image_path or "" for item in self.prototypes], dtype=np.str_
            ),
            cluster_sizes=np.array([item.cluster_size for item in self.prototypes], dtype=np.int64),
            reliabilities=np.array(
                [item.reliability for item in self.prototypes], dtype=np.float32
            ),
            compactness=np.array([item.compactness for item in self.prototypes], dtype=np.float32),
            purities=np.array([item.purity for item in self.prototypes], dtype=np.float32),
            text_alignments=np.array(
                [
                    np.nan if item.text_alignment is None else item.text_alignment
                    for item in self.prototypes
                ],
                dtype=np.float32,
            ),
            usage_counts=np.array([item.usage_count for item in self.prototypes], dtype=np.int64),
            reliability_weights=np.array(
                [
                    self.reliability_weights.compactness,
                    self.reliability_weights.text_alignment,
                    self.reliability_weights.purity,
                ],
                dtype=np.float64,
            ),
            selection_strategy=np.array(self.selection_config.strategy, dtype=np.str_),
            selection_values=np.array(
                [
                    self.selection_config.candidate_multiplier,
                    self.selection_config.coverage_weight,
                    self.selection_config.reliability_weight,
                ],
                dtype=np.float64,
            ),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, *, index_backend: str = "auto") -> EvidenceMemory:
        """Restore an archive produced by :meth:`save` and rebuild its index."""
        with np.load(Path(path), allow_pickle=False) as archive:
            if "reliability_weights" in archive.files:
                stored_reliability = archive["reliability_weights"]
                reliability_weights = ReliabilityWeights(
                    compactness=float(stored_reliability[0]),
                    text_alignment=float(stored_reliability[1]),
                    purity=float(stored_reliability[2]),
                )
            else:
                reliability_weights = ReliabilityWeights()
            if "selection_values" in archive.files:
                stored_selection = archive["selection_values"]
                selection_config = SelectionConfig(
                    strategy=str(archive["selection_strategy"]),
                    candidate_multiplier=float(stored_selection[0]),
                    coverage_weight=float(stored_selection[1]),
                    reliability_weight=float(stored_selection[2]),
                )
            else:
                selection_config = SelectionConfig()
            prototypes = []
            for index in range(int(archive["vectors"].shape[0])):
                alignment = float(archive["text_alignments"][index])
                prototypes.append(
                    Prototype(
                        vector=archive["vectors"][index],
                        centroid=archive["centroids"][index],
                        class_id=int(archive["class_ids"][index]),
                        class_name=str(archive["class_names"][index]),
                        sample_id=str(archive["sample_ids"][index]),
                        image_path=str(archive["image_paths"][index]) or None,
                        cluster_size=int(archive["cluster_sizes"][index]),
                        reliability=float(archive["reliabilities"][index]),
                        compactness=float(archive["compactness"][index]),
                        purity=float(archive["purities"][index]),
                        text_alignment=None if np.isnan(alignment) else alignment,
                        usage_count=int(archive["usage_counts"][index]),
                    )
                )
        return cls(
            prototypes,
            index_backend=index_backend,
            reliability_weights=reliability_weights,
            selection_config=selection_config,
        )
