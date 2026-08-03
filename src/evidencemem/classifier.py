"""Reliability-weighted retrieval, text fusion, and unknown rejection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from .memory import EvidenceMemory
from .schema import Prediction, SearchResult
from .utils import cosine_to_unit, normalize_vector, softmax

DEFAULT_CONFIDENCE_WEIGHTS = {
    "score": 0.20,
    "margin": 0.20,
    "agreement": 0.20,
    "similarity": 0.20,
    "reliability": 0.20,
}


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """Weights and validation-selected rejection thresholds."""

    weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_CONFIDENCE_WEIGHTS))
    confidence_threshold: float | None = None
    similarity_threshold: float | None = None
    agreement_threshold: float | None = None

    def __post_init__(self) -> None:
        expected = set(DEFAULT_CONFIDENCE_WEIGHTS)
        if set(self.weights) != expected:
            raise ValueError(f"confidence weights must contain exactly {sorted(expected)}")
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("confidence weights must be non-negative with a positive sum")


class EvidenceMemClassifier:
    """Classify normalized VLM embeddings with visual and textual evidence."""

    def __init__(
        self,
        memory: EvidenceMemory,
        text_prototypes: Mapping[int, ArrayLike] | None = None,
        *,
        fusion_weight: float = 0.50,
        visual_temperature: float = 0.07,
        text_temperature: float = 0.07,
        confidence: ConfidenceConfig | None = None,
    ) -> None:
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must lie in [0, 1]")
        if visual_temperature <= 0 or text_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.memory = memory
        self.text_prototypes = {
            int(class_id): normalize_vector(vector, name=f"text prototype {class_id}")
            for class_id, vector in (text_prototypes or {}).items()
        }
        if self.text_prototypes and memory.dimension:
            mismatched = [
                class_id
                for class_id, vector in self.text_prototypes.items()
                if vector.shape[0] != memory.dimension
            ]
            if mismatched:
                raise ValueError(f"text prototype dimensions mismatch for classes {mismatched}")
        self.fusion_weight = fusion_weight
        self.visual_temperature = visual_temperature
        self.text_temperature = text_temperature
        self.confidence_config = confidence or ConfidenceConfig()

    @staticmethod
    def _visual_scores(
        evidence: tuple[SearchResult, ...],
        class_ids: list[int],
        temperature: float,
    ) -> dict[int, float]:
        scores = {class_id: 0.0 for class_id in class_ids}
        similarities = np.array([item.similarity for item in evidence], dtype=np.float64)
        shifted = (similarities - similarities.max()) / temperature
        weights = np.exp(shifted) * np.array(
            [max(item.reliability, 1e-8) for item in evidence], dtype=np.float64
        )
        denominator = float(weights.sum())
        for item, weight in zip(evidence, weights, strict=True):
            scores[item.class_id] += float(weight / denominator)
        return scores

    def _text_scores(self, query: np.ndarray, class_ids: list[int]) -> dict[int, float]:
        scores = {class_id: 0.0 for class_id in class_ids}
        available = [class_id for class_id in class_ids if class_id in self.text_prototypes]
        if not available:
            return scores
        logits = np.array(
            [query @ self.text_prototypes[class_id] for class_id in available],
            dtype=np.float32,
        )
        probabilities = softmax(logits, temperature=self.text_temperature)
        for class_id, probability in zip(available, probabilities, strict=True):
            scores[class_id] = float(probability)
        return scores

    def predict_embedding(self, query: ArrayLike, *, k: int = 10) -> Prediction:
        """Predict one embedding and return all evidence used by the rule."""
        vector = normalize_vector(query, name="query")
        evidence = self.memory.search(vector, k)
        class_ids = sorted(set(self.memory.class_names) | set(self.text_prototypes))
        visual_scores = self._visual_scores(evidence, class_ids, self.visual_temperature)
        text_scores = self._text_scores(vector, class_ids)
        effective_fusion = self.fusion_weight if self.text_prototypes else 1.0
        final_scores = {
            class_id: effective_fusion * visual_scores[class_id]
            + (1.0 - effective_fusion) * text_scores[class_id]
            for class_id in class_ids
        }

        predicted_id = max(class_ids, key=lambda class_id: (final_scores[class_id], -class_id))
        ordered_scores = sorted(final_scores.values(), reverse=True)
        top_score = float(ordered_scores[0])
        margin = top_score - float(ordered_scores[1]) if len(ordered_scores) > 1 else top_score
        predicted_evidence = [item for item in evidence if item.class_id == predicted_id]
        agreement = len(predicted_evidence) / len(evidence)
        max_similarity = max(item.similarity for item in evidence)

        if predicted_evidence:
            rel_logits = np.array(
                [item.similarity for item in predicted_evidence], dtype=np.float32
            )
            rel_weights = softmax(rel_logits, temperature=self.visual_temperature)
            evidence_reliability = float(
                sum(
                    item.reliability * weight
                    for item, weight in zip(predicted_evidence, rel_weights, strict=True)
                )
            )
        else:
            evidence_reliability = 0.0

        components = {
            "score": top_score,
            "margin": margin,
            "agreement": float(agreement),
            "similarity": float(max_similarity),
            "reliability": evidence_reliability,
        }
        bounded_components = {
            **components,
            "similarity": float(cosine_to_unit(max_similarity)),
        }
        weight_sum = float(sum(self.confidence_config.weights.values()))
        combined_confidence = (
            sum(
                self.confidence_config.weights[name] * bounded_components[name]
                for name in DEFAULT_CONFIDENCE_WEIGHTS
            )
            / weight_sum
        )

        config = self.confidence_config
        is_unknown = bool(
            (
                config.confidence_threshold is not None
                and combined_confidence < config.confidence_threshold
            )
            or (
                config.similarity_threshold is not None
                and max_similarity < config.similarity_threshold
            )
            or (config.agreement_threshold is not None and agreement < config.agreement_threshold)
        )
        class_name = self.memory.class_names.get(predicted_id, str(predicted_id))
        return Prediction(
            class_id=predicted_id,
            class_name=class_name,
            confidence=float(combined_confidence),
            is_unknown=is_unknown,
            final_scores={key: float(value) for key, value in final_scores.items()},
            visual_scores={key: float(value) for key, value in visual_scores.items()},
            text_scores={key: float(value) for key, value in text_scores.items()},
            confidence_components=components,
            evidence=evidence,
        )
