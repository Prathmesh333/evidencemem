"""Fast end-to-end check of the core with synthetic normalized embeddings."""

from __future__ import annotations

import numpy as np

from evidencemem import EvidenceMemClassifier, EvidenceMemory


def main() -> None:
    rng = np.random.default_rng(42)
    dimensions = 16
    class_count = 3
    samples_per_class = 40
    centers = rng.normal(size=(class_count, dimensions)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    embeddings = []
    labels = []
    for class_id, center in enumerate(centers):
        samples = center + 0.08 * rng.normal(size=(samples_per_class, dimensions))
        embeddings.append(samples.astype(np.float32))
        labels.extend([class_id] * samples_per_class)

    matrix = np.concatenate(embeddings, axis=0)
    label_array = np.asarray(labels, dtype=np.int64)
    names = {0: "red", 1: "green", 2: "blue"}

    memory = EvidenceMemory(index_backend="numpy").build(
        matrix,
        label_array,
        names,
        prototypes_per_class=4,
        text_prototypes={index: center for index, center in enumerate(centers)},
        random_state=42,
    )
    classifier = EvidenceMemClassifier(
        memory,
        {index: center for index, center in enumerate(centers)},
        fusion_weight=0.5,
    )
    prediction = classifier.predict_embedding(centers[1], k=5)
    if prediction.class_id != 1:
        raise SystemExit(f"smoke test failed: expected class 1, got {prediction.class_id}")

    print(f"prototypes={len(memory)} dimension={memory.dimension}")
    print(
        f"prediction={prediction.class_name} confidence={prediction.confidence:.3f} "
        f"unknown={prediction.is_unknown}"
    )
    print("evidence=" + ", ".join(item.class_name for item in prediction.evidence))


if __name__ == "__main__":
    main()
