import numpy as np

from evidencemem.benchmark import (
    fit_prototype_memory,
    fused_class_scores,
    tip_adapter_scores,
    weighted_knn_scores,
)


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
