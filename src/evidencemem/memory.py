"""Construction and online maintenance of displayable visual prototypes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import MiniBatchKMeans

from .index import VectorIndex, make_index
from .schema import Prototype, SearchResult
from .utils import FloatArray, cosine_to_unit, normalize_rows, normalize_vector


@dataclass(frozen=True, slots=True)
class ReliabilityWeights:
    compactness: float = 0.40
    text_alignment: float = 0.20
    purity: float = 0.40

    def __post_init__(self) -> None:
        if min(self.compactness, self.text_alignment, self.purity) < 0:
            raise ValueError("reliability weights must be non-negative")
        if self.compactness + self.text_alignment + self.purity <= 0:
            raise ValueError("at least one reliability weight must be positive")


class EvidenceMemory:
    """A bounded collection of class-conditional, real-image medoids."""

    def __init__(
        self,
        prototypes: Sequence[Prototype] | None = None,
        *,
        index_backend: str = "auto",
        reliability_weights: ReliabilityWeights | None = None,
    ) -> None:
        self.prototypes: list[Prototype] = list(prototypes or [])
        self.class_names: dict[int, str] = {
            prototype.class_id: prototype.class_name for prototype in self.prototypes
        }
        self.index_backend = index_backend
        self.reliability_weights = reliability_weights or ReliabilityWeights()
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

    @staticmethod
    def _purity_at_medoid(
        embeddings: FloatArray,
        labels: NDArray[np.int64],
        medoid_index: int,
        class_id: int,
        k: int,
    ) -> float:
        if embeddings.shape[0] == 1:
            return 1.0
        k_eff = min(max(k, 1), embeddings.shape[0] - 1)
        similarities = embeddings @ embeddings[medoid_index]
        similarities[medoid_index] = -np.inf
        neighbor_indices = np.argpartition(-similarities, kth=k_eff - 1)[:k_eff]
        return float(np.mean(labels[neighbor_indices] == class_id))

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
        duplicate_threshold: float | None = 0.98,
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
        built: list[Prototype] = []
        for class_id in sorted(int(value) for value in np.unique(label_array)):
            if class_id not in class_names:
                raise ValueError(f"missing class name for class id {class_id}")
            global_indices = np.flatnonzero(label_array == class_id)
            class_vectors = matrix[global_indices]
            cluster_count = min(prototypes_per_class, class_vectors.shape[0])

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
                purity = self._purity_at_medoid(
                    matrix, label_array, medoid_global, class_id, purity_k
                )
                text_alignment = (
                    float(medoid @ text_vectors[class_id]) if class_id in text_vectors else None
                )
                reliability = self._combine_reliability(compactness, purity, text_alignment)
                candidates.append(
                    Prototype(
                        vector=medoid,
                        centroid=centroid,
                        class_id=class_id,
                        class_name=class_names[class_id],
                        sample_id=str(ids[medoid_global]),
                        image_path=paths[medoid_global],
                        cluster_size=int(local_members.size),
                        reliability=reliability,
                        compactness=compactness,
                        purity=purity,
                        text_alignment=text_alignment,
                    )
                )

            candidates.sort(key=lambda item: (-item.reliability, item.sample_id))
            accepted: list[Prototype] = []
            for candidate in candidates:
                is_duplicate = False
                if duplicate_threshold is not None and accepted:
                    similarities = [candidate.vector @ item.vector for item in accepted]
                    is_duplicate = max(similarities) > duplicate_threshold
                if not is_duplicate:
                    accepted.append(candidate)
            built.extend(accepted)

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
        duplicate_threshold: float | None = 0.98,
        random_state: int = 0,
    ) -> int:
        """Insert a previously unseen class without changing existing prototypes."""
        if class_id in self.class_names:
            raise ValueError(f"class id {class_id} already exists")
        matrix = normalize_rows(support_embeddings, name="support_embeddings")
        labels = np.full(matrix.shape[0], class_id, dtype=np.int64)
        temporary = EvidenceMemory(
            index_backend="numpy", reliability_weights=self.reliability_weights
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
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, *, index_backend: str = "auto") -> EvidenceMemory:
        """Restore an archive produced by :meth:`save` and rebuild its index."""
        with np.load(Path(path), allow_pickle=False) as archive:
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
        return cls(prototypes, index_backend=index_backend)
