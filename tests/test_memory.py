from collections import Counter

import numpy as np

from evidencemem import EvidenceMemory


def clustered_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    rng = np.random.default_rng(seed)
    centers = {
        0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    samples = []
    labels = []
    for class_id, center in centers.items():
        samples.append(center + 0.04 * rng.normal(size=(20, 4)))
        labels.extend([class_id] * 20)
    return (
        np.concatenate(samples).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        centers,
    )


def test_build_selects_reliable_real_sample_medoids() -> None:
    embeddings, labels, centers = clustered_data()
    sample_ids = [f"sample-{index}" for index in range(len(labels))]
    memory = EvidenceMemory(index_backend="numpy").build(
        embeddings,
        labels,
        {0: "zero", 1: "one"},
        prototypes_per_class=3,
        sample_ids=sample_ids,
        text_prototypes=centers,
        duplicate_threshold=None,
        random_state=7,
    )

    assert len(memory) == 6
    assert all(item.sample_id in sample_ids for item in memory.prototypes)
    assert all(0.0 <= item.reliability <= 1.0 for item in memory.prototypes)
    assert all(np.isclose(np.linalg.norm(item.vector), 1.0) for item in memory.prototypes)
    assert memory.search(centers[1], k=3)[0].class_id == 1


def test_update_merge_insert_and_prune_preserve_classes() -> None:
    embeddings, labels, centers = clustered_data()
    memory = EvidenceMemory(index_backend="numpy").build(
        embeddings,
        labels,
        {0: "zero", 1: "one"},
        prototypes_per_class=2,
        duplicate_threshold=None,
    )

    original_size = len(memory)
    action = memory.update(centers[0], 0, sample_id="near-zero", merge_threshold=0.8)
    assert action == "merged"
    assert len(memory) == original_size

    action = memory.update(
        np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        0,
        sample_id="new-mode",
        merge_threshold=0.95,
    )
    assert action == "inserted"
    assert len(memory) == original_size + 1

    removed = memory.prune(original_size, min_per_class=1)
    counts = Counter(item.class_id for item in memory.prototypes)
    assert len(removed) == 1
    assert counts[0] >= 1 and counts[1] >= 1


def test_save_and_load_round_trip(tmp_path) -> None:
    embeddings, labels, centers = clustered_data()
    memory = EvidenceMemory(index_backend="numpy").build(
        embeddings,
        labels,
        {0: "zero", 1: "one"},
        prototypes_per_class=2,
        text_prototypes=centers,
        duplicate_threshold=None,
    )
    archive = memory.save(tmp_path / "memory.npz")

    restored = EvidenceMemory.load(archive, index_backend="numpy")

    assert len(restored) == len(memory)
    assert restored.class_names == memory.class_names
    assert restored.search(centers[0], k=1)[0].class_id == 0
