"""EvidenceMem: reliability-aware prototype memory for VLM embeddings."""

from .classifier import ConfidenceConfig, EvidenceMemClassifier
from .memory import EvidenceMemory, ReliabilityWeights
from .schema import Prediction, Prototype, SearchResult

__all__ = [
    "ConfidenceConfig",
    "EvidenceMemClassifier",
    "EvidenceMemory",
    "Prediction",
    "Prototype",
    "ReliabilityWeights",
    "SearchResult",
]

__version__ = "0.1.0"
