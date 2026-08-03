"""Typed records returned by EvidenceMem."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .utils import normalize_vector


@dataclass(slots=True)
class Prototype:
    """A real, displayable sample representing one local class mode."""

    vector: NDArray[np.float32]
    centroid: NDArray[np.float32]
    class_id: int
    class_name: str
    sample_id: str
    image_path: str | None = None
    cluster_size: int = 1
    reliability: float = 1.0
    compactness: float = 1.0
    purity: float = 1.0
    text_alignment: float | None = None
    usage_count: int = 0

    def __post_init__(self) -> None:
        self.vector = normalize_vector(self.vector, name="prototype vector")
        self.centroid = normalize_vector(self.centroid, name="prototype centroid")
        if self.cluster_size < 1:
            raise ValueError("cluster_size must be at least one")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must lie in [0, 1]")
        if not 0.0 <= self.purity <= 1.0:
            raise ValueError("purity must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One retrieved prototype and its cosine similarity to a query."""

    prototype_index: int
    class_id: int
    class_name: str
    sample_id: str
    image_path: str | None
    similarity: float
    reliability: float


@dataclass(frozen=True, slots=True)
class Prediction:
    """A classification decision with confidence and inspectable evidence."""

    class_id: int
    class_name: str
    confidence: float
    is_unknown: bool
    final_scores: dict[int, float]
    visual_scores: dict[int, float]
    text_scores: dict[int, float]
    confidence_components: dict[str, float]
    evidence: tuple[SearchResult, ...] = field(default_factory=tuple)
