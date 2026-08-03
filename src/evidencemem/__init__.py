"""EvidenceMem: reliability-aware prototype memory for VLM embeddings."""

from .classifier import ConfidenceConfig, EvidenceMemClassifier
from .encoder import canonical_open_clip_model_name
from .memory import EvidenceMemory, ReliabilityWeights, SelectionConfig
from .schema import Prediction, Prototype, SearchResult

__all__ = [
    "ConfidenceConfig",
    "EvidenceMemClassifier",
    "EvidenceMemory",
    "Prediction",
    "Prototype",
    "ReliabilityWeights",
    "SelectionConfig",
    "SearchResult",
    "canonical_open_clip_model_name",
]

__version__ = "0.2.0"
