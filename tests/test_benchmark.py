import numpy as np

from evidencemem.benchmark import (
    PrototypeArrays,
    class_conditional_visual_scores,
    fit_prototype_memory,
    fused_class_scores,
    reweight_prototype_reliability,
    tip_adapter_scores,
    weighted_knn_scores,
)
from evidencemem.memory import ReliabilityWeights


def toy_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    embeddings = np.array(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    text = np.eye(2, dtype=np.float32)
    return embeddings, labels, text


def test_shared_scores_are_normalized_and_matched() -> None:
    embeddings, labels, text = toy_data()
    memory = fit_prototype_memory(
        embeddings,
        labels,
        budget_per_class=1,
        method="facility_no_reliability",
        seed=7,
        text_prototypes=text,
    )

    fused, visual, text_scores = fused_class_scores(
        memory,
        embeddings,
        text,
        text_weight=0.25,
        k=2,
    )
    knn = weighted_knn_scores(
        embeddings,
        labels,
        embeddings,
        k=2,
        n_classes=2,
    )
    tip = tip_adapter_scores(memory, embeddings, text, beta=1.0, cache_weight=1.0)

    assert memory.prototypes.shape[0] == 2
    assert np.allclose(fused, 0.75 * visual + 0.25 * text_scores)
    assert np.allclose(fused.sum(axis=1), 1.0)
    assert np.allclose(knn.sum(axis=1), 1.0)
    assert tip.shape == (4, 2)


def test_reweight_prototype_reliability_uses_compatible_scales() -> None:
    memory = PrototypeArrays(
        prototypes=np.eye(2, dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        source_indices=np.array([0, 1], dtype=np.int64),
        reliabilities=np.ones(2, dtype=np.float32),
        compactness=np.array([-1.0, 1.0], dtype=np.float32),
        purity=np.array([1.0, 0.0], dtype=np.float32),
        text_alignment=np.array([np.nan, 1.0], dtype=np.float32),
        method="test",
        budget_per_class=1,
        seed=7,
    )

    compactness_only = reweight_prototype_reliability(
        memory,
        ReliabilityWeights(compactness=1.0, purity=0.0, text_alignment=0.0),
    )
    equal_weights = reweight_prototype_reliability(
        memory,
        ReliabilityWeights(compactness=1.0, purity=1.0, text_alignment=1.0),
    )

    assert np.allclose(compactness_only.reliabilities, [0.05, 1.0])
    assert np.allclose(equal_weights.reliabilities, [0.5, 2.0 / 3.0])


def test_class_conditional_scores_compare_equal_evidence_per_class() -> None:
    memory = PrototypeArrays(
        prototypes=np.array(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            dtype=np.float32,
        ),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        source_indices=np.arange(4, dtype=np.int64),
        reliabilities=np.array([1.0, 0.8, 1.0, 0.8], dtype=np.float32),
        compactness=np.ones(4, dtype=np.float32),
        purity=np.ones(4, dtype=np.float32),
        text_alignment=np.ones(4, dtype=np.float32),
        method="test",
        budget_per_class=2,
        seed=7,
    )
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores, selected = class_conditional_visual_scores(
        memory,
        queries,
        k_per_class=2,
        n_classes=2,
    )

    assert np.allclose(scores.sum(axis=1), 1.0)
    assert np.array_equal(scores.argmax(axis=1), [0, 1])
    assert selected.shape == (2, 2, 2)
    assert set(selected[0, 0]) == {0, 1}
    assert set(selected[0, 1]) == {2, 3}
