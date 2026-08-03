import json

import pytest

from evidencemem.artifacts import (
    atomic_write_json,
    finalize_run_manifest,
    start_run_manifest,
)


def test_run_manifest_requires_and_hashes_outputs(tmp_path) -> None:
    manifest = start_run_manifest(
        config={"mode": "validation", "seed": 7},
        repository=tmp_path,
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
    )
    (tmp_path / "metrics.csv").write_text("accuracy\n0.9\n", encoding="utf-8")
    completed = finalize_run_manifest(
        manifest,
        run_directory=tmp_path,
        required_artifacts=["metrics.csv"],
    )
    destination = atomic_write_json(tmp_path / "run_manifest.json", completed)

    restored = json.loads(destination.read_text(encoding="utf-8"))
    assert restored["status"] == "complete"
    assert restored["encoder"]["model_name"] == "ViT-B-32-quickgelu"
    assert len(restored["artifacts"]["metrics.csv"]["sha256"]) == 64

    with pytest.raises(FileNotFoundError, match="missing artifacts"):
        finalize_run_manifest(
            manifest,
            run_directory=tmp_path,
            required_artifacts=["predictions.csv"],
        )
