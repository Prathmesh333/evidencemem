import numpy as np

from evidencemem import ConfidenceConfig, EvidenceMemClassifier, EvidenceMemory


def make_memory() -> tuple[EvidenceMemory, dict[int, np.ndarray]]:
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.05, 0.95, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    texts = {
        0: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    memory = EvidenceMemory(index_backend="numpy").build(
        embeddings,
        labels,
        {0: "cat", 1: "dog"},
        prototypes_per_class=2,
        text_prototypes=texts,
        duplicate_threshold=None,
    )
    return memory, texts


def test_classifier_returns_prediction_scores_and_evidence() -> None:
    memory, texts = make_memory()
    classifier = EvidenceMemClassifier(memory, texts, text_weight=0.5)

    prediction = classifier.predict_embedding(np.array([1.0, 0.0, 0.0]), k=3)

    assert prediction.class_id == 0
    assert prediction.class_name == "cat"
    assert not prediction.is_unknown
    assert len(prediction.evidence) == 3
    assert np.isclose(sum(prediction.final_scores.values()), 1.0)
    assert prediction.confidence_components["agreement"] >= 2 / 3


def test_classifier_can_reject_low_agreement() -> None:
    memory, texts = make_memory()
    classifier = EvidenceMemClassifier(
        memory,
        texts,
        confidence=ConfidenceConfig(agreement_threshold=0.8),
    )

    prediction = classifier.predict_embedding(np.array([0.7, 0.7, 0.0]), k=4)

    assert prediction.is_unknown


def test_text_weight_has_unambiguous_endpoints() -> None:
    memory, texts = make_memory()
    query = np.array([0.8, 0.2, 0.0], dtype=np.float32)

    visual_only = EvidenceMemClassifier(memory, texts, text_weight=0.0).predict_embedding(
        query, k=3
    )
    text_only = EvidenceMemClassifier(memory, texts, text_weight=1.0).predict_embedding(
        query, k=3
    )

    assert visual_only.final_scores == visual_only.visual_scores
    assert text_only.final_scores == text_only.text_scores
