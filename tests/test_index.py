import numpy as np

from evidencemem.index import ExactNumpyIndex


def test_exact_index_orders_by_cosine_similarity() -> None:
    index = ExactNumpyIndex()
    index.build(np.eye(3, dtype=np.float32))

    result = index.search(np.array([[0.9, 0.1, 0.0]], dtype=np.float32), k=2)

    assert result.indices.tolist() == [[0, 1]]
    assert result.similarities[0, 0] > result.similarities[0, 1]


def test_exact_index_rejects_dimension_mismatch() -> None:
    index = ExactNumpyIndex()
    index.build(np.eye(3, dtype=np.float32))

    try:
        index.search(np.ones((1, 2), dtype=np.float32), k=1)
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("dimension mismatch should raise ValueError")
