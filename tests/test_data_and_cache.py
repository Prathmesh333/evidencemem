import json

import numpy as np

from evidencemem.cache import load_embedding_cache, save_embedding_cache
from evidencemem.data import save_split, stratified_train_validation_indices


def test_stratified_split_is_deterministic_and_balanced(tmp_path) -> None:
    labels = np.repeat(np.arange(4), 10)
    train_a, validation_a = stratified_train_validation_indices(labels, validation_size=8, seed=42)
    train_b, validation_b = stratified_train_validation_indices(labels, validation_size=8, seed=42)

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert np.bincount(labels[validation_a]).tolist() == [2, 2, 2, 2]
    assert np.intersect1d(train_a, validation_a).size == 0

    manifest = save_split(
        tmp_path,
        dataset="toy",
        labels=labels,
        train_indices=train_a,
        validation_indices=validation_a,
        seed=42,
    )
    assert manifest.train_count == 32
    assert json.loads((tmp_path / "split_manifest.json").read_text())["seed"] == 42


def test_embedding_cache_round_trip(tmp_path) -> None:
    embeddings = np.eye(3, dtype=np.float32)
    labels = np.array([0, 1, 2], dtype=np.int64)
    sample_ids = np.array(["a", "b", "c"])
    path = tmp_path / "train.npz"

    manifest = save_embedding_cache(
        path,
        embeddings,
        labels,
        sample_ids,
        dataset="toy",
        split="train",
        model_name="mock",
        pretrained="none",
    )
    restored = load_embedding_cache(path)

    assert manifest.normalized
    assert restored.manifest == manifest
    assert np.array_equal(restored.embeddings, embeddings)
    assert np.array_equal(restored.labels, labels)
    assert restored.sample_ids.tolist() == ["a", "b", "c"]
