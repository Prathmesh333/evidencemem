import ast
import hashlib
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evidencemem.cache import load_embedding_cache, save_embedding_cache

REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK = REPOSITORY / "notebooks" / "EvidenceMem_UIE22K_V4_T4.ipynb"
CONFIRMATORY_NOTEBOOK = (
    REPOSITORY / "notebooks" / "EvidenceMem_UIE22K_V4_Confirmatory_T4.ipynb"
)


def tagged_source(tag: str, notebook_path: Path = NOTEBOOK) -> str:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    matching = [
        cell["source"]
        for cell in notebook["cells"]
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(matching) == 1
    return matching[0]


def load_pooled_feature_helper():
    tree = ast.parse(tagged_source("embeddings"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "pooled_feature_tensor"
    )
    namespace = {
        "torch": SimpleNamespace(is_tensor=lambda value: isinstance(value, np.ndarray))
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<helper>", "exec"), namespace)
    return namespace["pooled_feature_tensor"]


def load_compatible_cache_helper(cache_directory: Path):
    tree = ast.parse(tagged_source("embeddings"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "compatible_embedding_cache"
    )
    namespace = {
        "CACHE_DIR": cache_directory,
        "PREPROCESS_CONTRACTS": {
            "open_clip": "open_clip_native_eval_v1",
            "transformers": "hf_slow_processor_native_eval_v2",
        },
        "PROTOCOL_ID": "uie22k_evidencemem_v4",
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "load_embedding_cache": load_embedding_cache,
        "np": np,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<cache>", "exec"), namespace)
    return namespace["compatible_embedding_cache"]


def test_siglip_structured_feature_outputs_are_unwrapped() -> None:
    pooled_feature_tensor = load_pooled_feature_helper()
    features = np.ones((2, 4), dtype=np.float32)

    assert pooled_feature_tensor(features, source="direct") is features
    assert (
        pooled_feature_tensor(
            SimpleNamespace(pooler_output=features), source="structured"
        )
        is features
    )
    assert pooled_feature_tensor((np.ones((2, 3, 4)), features), source="tuple") is features
    with pytest.raises(TypeError, match="without a pooled tensor"):
        pooled_feature_tensor(SimpleNamespace(pooler_output=None), source="missing")


def test_notebook_can_reuse_verified_failed_run_caches() -> None:
    embeddings = tagged_source("embeddings")

    assert 'use_fast=False' in embeddings
    assert "compatible_embedding_cache(" in embeddings
    assert 'manifest.dataset == "UIE-22K"' in embeddings
    assert "cached.sample_ids.astype(str), expected_ids" in embeddings
    assert "cached.labels, expected_labels" in embeddings
    assert 'spec["backend"] == "open_clip"' in embeddings


def test_failed_run_openclip_cache_is_recovered_by_manifest(tmp_path: Path) -> None:
    spec = {
        "key": "clip_b32_224",
        "backend": "open_clip",
        "model": "ViT-B-32-quickgelu",
        "weights": "openai",
        "resolution": 224,
        "batch_size": 128,
    }
    embeddings = np.eye(2, dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int64)
    sample_ids = np.array(["sample-a", "sample-b"])
    legacy_fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    old_path = tmp_path / "uie22k_train_clip_b32_224_old-source.npz"
    save_embedding_cache(
        old_path,
        embeddings,
        labels,
        sample_ids,
        dataset="UIE-22K",
        split="train",
        model_name=spec["model"],
        pretrained=spec["weights"],
        preprocess_fingerprint=legacy_fingerprint,
        source_revision="old-source",
    )

    compatible_embedding_cache = load_compatible_cache_helper(tmp_path)
    selected_path, cached = compatible_embedding_cache(
        "train",
        spec,
        sample_ids,
        labels,
        tmp_path / "uie22k_train_clip_b32_224_new-source.npz",
    )

    assert selected_path == old_path
    assert np.array_equal(cached.embeddings, embeddings)


def load_confirmatory_stat_helpers():
    tree = ast.parse(tagged_source("confirmatory-statistics", CONFIRMATORY_NOTEBOOK))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {"hierarchical_paired_accuracy", "evaluate_frozen_hypothesis"}
    ]
    namespace = {"np": np, "product": product}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<statistics>", "exec"),
        namespace,
    )
    return namespace


def test_confirmatory_notebook_is_hard_frozen() -> None:
    notebook = json.loads(CONFIRMATORY_NOTEBOOK.read_text(encoding="utf-8"))
    parameters = tagged_source("parameters", CONFIRMATORY_NOTEBOOK)
    bootstrap = tagged_source("bootstrap", CONFIRMATORY_NOTEBOOK)
    guard = tagged_source("frozen-manifest-guard", CONFIRMATORY_NOTEBOOK)
    statistics = tagged_source("confirmatory-statistics", CONFIRMATORY_NOTEBOOK)
    export = tagged_source("export", CONFIRMATORY_NOTEBOOK)

    assert 'EVALUATION_STAGE = "confirmatory"' in parameters
    assert 'CONFIRMATORY_ENCODER_KEY = "siglip2_b16_384"' in parameters
    assert "EVIDENCEMEM_EVALUATION_STAGE" not in parameters
    assert "EVIDENCEMEM_CONFIRMATORY_ENCODER" not in parameters
    assert 'REPOSITORY_REF = "uie22k-v4-confirmatory-2026-09-02"' in bootstrap
    assert "EVIDENCEMEM_REF" not in bootstrap
    assert "f9eece5f3f489fd2b986ca89b797c2843e53e2618cd93361b730f0c77bff2c09" in parameters
    assert "9fec2b475fe962bd92fd8d6e496ec7bcc0b1835e" in parameters
    assert "manifest_matches_development" in guard
    assert "package_tree_matches_development" in guard
    assert "hierarchical_paired_accuracy" in statistics
    assert "H1_full_knn_noninferiority_1pp" in parameters
    assert "H2_facility_superiority" in parameters
    assert 'RUN_DIR / "confirmatory_hypotheses.csv"' in statistics
    assert '"confirmatory_hypotheses_complete"' in export
    assert '"paired_tests_complete"' in export
    assert '"frozen_protocol.json"' in export

    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert len(cell_ids) == len(set(cell_ids))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        assert cell.get("execution_count") is None
        assert cell.get("outputs") == []
        ast.parse("".join(cell.get("source", [])), filename=f"confirmatory-{index}")


def test_confirmatory_prediction_cell_matches_development() -> None:
    assert tagged_source("main-experiment", CONFIRMATORY_NOTEBOOK) == tagged_source(
        "main-experiment", NOTEBOOK
    )


def test_hierarchical_confirmatory_statistics() -> None:
    helpers = load_confirmatory_stat_helpers()
    hierarchical = helpers["hierarchical_paired_accuracy"]
    decide = helpers["evaluate_frozen_hypothesis"]
    treatment = np.ones((3, 8), dtype=bool)
    baseline = np.zeros((3, 8), dtype=bool)

    result = hierarchical(treatment, baseline, seed=7, draws=200, batch_size=25)

    assert result["delta"] == pytest.approx(1.0)
    assert result["ci_low"] == pytest.approx(1.0)
    assert result["ci_high"] == pytest.approx(1.0)
    assert result["n_seeds"] == 3
    assert result["n_examples"] == 8
    decision = decide(
        {"alternative": "noninferiority", "margin": 0.01},
        result,
    )
    assert decision["supported"] is True
    assert decision["decision_boundary"] == pytest.approx(-0.01)

    with pytest.raises(ValueError, match="matching"):
        hierarchical(treatment, baseline[:, :-1], seed=7, draws=20)
