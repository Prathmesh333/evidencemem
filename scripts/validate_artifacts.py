"""Fail when the source notebook or package metadata can reproduce known artifact bugs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPOSITORY / "notebooks" / "EvidenceMem_Colab_T4.ipynb"
UIE_NOTEBOOK = REPOSITORY / "notebooks" / "EvidenceMem_UIE22K_V4_T4.ipynb"
UIE_CONFIRMATORY_NOTEBOOK = (
    REPOSITORY / "notebooks" / "EvidenceMem_UIE22K_V4_Confirmatory_T4.ipynb"
)
LEGACY_NOTEBOOK = REPOSITORY / "results" / "evidencemem.ipynb"


def code_sources(notebook: dict[str, object]) -> list[str]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return [
        "".join(cell.get("source", []))
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    ]


def main() -> None:
    source = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    source_code = code_sources(source)
    joined = "\n".join(source_code)
    requirements = {
        "activation-compatible model": 'OPENCLIP_MODEL = "ViT-B-32-quickgelu"',
        "single-process Colab loading": "num_workers=0",
        "shared package method": "fit_prototype_memory(",
        "matched cache baseline": "tip_adapter_scores(",
        "tamper-evident embedding cache": "save_embedding_cache(",
        "run finalization": "finalize_run_manifest(",
        "private repository authentication": "get_github_token()",
        "Kaggle runtime path": 'Path("/kaggle/working")',
        "same-kernel source activation": "sys.path.insert(0, source_path)",
        "bootstrap package self-test": "Bootstrap verified: EvidenceMem",
        "stale checkout refresh": '"fetch", "--depth", "1", "origin"',
        "stale module eviction": 'module_name.startswith("evidencemem.")',
        "source-isolated memory cache": "|{SOURCE_ID}|",
    }
    missing = [name for name, marker in requirements.items() if marker not in joined]
    if missing:
        raise SystemExit(f"source notebook is missing safeguards: {missing}")
    forbidden = {
        "float16 embedding cache": "embeddings=x.astype(np.float16)",
        "duplicated KMeans method": "MiniBatchKMeans(",
        "multiprocess loader": "num_workers=2",
    }
    present = [name for name, marker in forbidden.items() if marker in joined]
    if present:
        raise SystemExit(f"source notebook still contains known failure modes: {present}")

    for index, cell in enumerate(source["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise SystemExit(f"source notebook cell {index} contains retained outputs")
        filtered = "\n".join(
            line
            for line in "".join(cell.get("source", [])).splitlines()
            if not line.lstrip().startswith(("%", "!"))
        )
        ast.parse(filtered, filename=f"{SOURCE_NOTEBOOK.name}:cell-{index}")

    uie = json.loads(UIE_NOTEBOOK.read_text(encoding="utf-8"))
    uie_code = code_sources(uie)
    uie_joined = "\n".join(uie_code)
    uie_requirements = {
        "development-confirmatory split": 'EVALUATION_STAGE = os.environ.get(',
        "frozen confirmatory encoder": "CONFIRMATORY_ENCODER_KEY",
        "native encoder sweep": '"siglip2_b16_384"',
        "structured SigLIP output handling": "pooled_feature_tensor(",
        "stable SigLIP processor": "use_fast=False",
        "verified attached-cache recovery": "compatible_embedding_cache(",
        "scale-corrected reliability": "reweight_prototype_reliability(",
        "class-conditional scorer": "class_conditional_visual_scores(",
        "continuous fusion": '"EvidenceMem v4 continuous fusion"',
        "matched 20-40-80 budgets": "budgets=(20, 40, 80)",
        "calibration artifact": 'RUN_DIR / "calibration_summary.csv"',
        "paired tests": 'RUN_DIR / "paired_tests.json"',
        "final-claim gate": '"ready_for_final_claims"',
    }
    uie_missing = [
        name for name, marker in uie_requirements.items() if marker not in uie_joined
    ]
    if uie_missing:
        raise SystemExit(f"UIE v4 notebook is missing safeguards: {uie_missing}")
    uie_forbidden = {
        "obsolete resolution loop": "CFG.resolutions",
        "obsolete shared-resolution cache": "RESOLUTION_DATA",
        "obsolete v3 scorer": "tune_v3_memory(",
        "obsolete test field": "CFG.test_per_class",
    }
    uie_present = [
        name for name, marker in uie_forbidden.items() if marker in uie_joined
    ]
    if uie_present:
        raise SystemExit(f"UIE v4 notebook contains obsolete code: {uie_present}")
    cell_ids = [cell.get("id") for cell in uie["cells"]]
    if len(cell_ids) != len(set(cell_ids)):
        raise SystemExit("UIE v4 notebook cell IDs are not unique")
    for index, cell in enumerate(uie["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise SystemExit(f"UIE v4 source cell {index} contains retained outputs")
        ast.parse("".join(cell.get("source", [])), filename=f"{UIE_NOTEBOOK.name}:cell-{index}")

    confirmatory = json.loads(UIE_CONFIRMATORY_NOTEBOOK.read_text(encoding="utf-8"))
    confirmatory_code = code_sources(confirmatory)
    confirmatory_joined = "\n".join(confirmatory_code)
    confirmatory_requirements = {
        "hard-frozen stage": 'EVALUATION_STAGE = "confirmatory"',
        "hard-frozen encoder": 'CONFIRMATORY_ENCODER_KEY = "siglip2_b16_384"',
        "immutable repository tag": (
            'REPOSITORY_REF = "uie22k-v4-confirmatory-2026-09-02"'
        ),
        "frozen manifest": (
            "f9eece5f3f489fd2b986ca89b797c2843e53e2618cd93361b730f0c77bff2c09"
        ),
        "frozen package tree": "9fec2b475fe962bd92fd8d6e496ec7bcc0b1835e",
        "manifest guard": '"manifest_matches_development"',
        "package guard": '"package_tree_matches_development"',
        "hierarchical paired inference": "hierarchical_paired_accuracy(",
        "primary noninferiority hypothesis": "H1_full_knn_noninferiority_1pp",
        "secondary superiority hypothesis": "H2_facility_superiority",
        "confirmatory statistics artifact": (
            'RUN_DIR / "confirmatory_hypotheses.csv"'
        ),
        "complete paired-test gate": '"paired_tests_complete"',
    }
    confirmatory_missing = [
        name
        for name, marker in confirmatory_requirements.items()
        if marker not in confirmatory_joined
    ]
    if confirmatory_missing:
        raise SystemExit(
            "UIE confirmatory notebook is missing safeguards: "
            f"{confirmatory_missing}"
        )
    confirmatory_forbidden = {
        "environment-overridable stage": "EVIDENCEMEM_EVALUATION_STAGE",
        "environment-overridable encoder": "EVIDENCEMEM_CONFIRMATORY_ENCODER",
        "environment-overridable source": "EVIDENCEMEM_REF",
    }
    confirmatory_present = [
        name
        for name, marker in confirmatory_forbidden.items()
        if marker in confirmatory_joined
    ]
    if confirmatory_present:
        raise SystemExit(
            "UIE confirmatory notebook contains mutable controls: "
            f"{confirmatory_present}"
        )
    development_main = next(
        cell["source"]
        for cell in uie["cells"]
        if "main-experiment" in cell.get("metadata", {}).get("tags", [])
    )
    confirmatory_main = next(
        cell["source"]
        for cell in confirmatory["cells"]
        if "main-experiment" in cell.get("metadata", {}).get("tags", [])
    )
    if development_main != confirmatory_main:
        raise SystemExit("confirmatory notebook changed the frozen prediction cell")
    confirmatory_cell_ids = [cell.get("id") for cell in confirmatory["cells"]]
    if len(confirmatory_cell_ids) != len(set(confirmatory_cell_ids)):
        raise SystemExit("UIE confirmatory notebook cell IDs are not unique")
    for index, cell in enumerate(confirmatory["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise SystemExit(
                f"UIE confirmatory source cell {index} contains retained outputs"
            )
        ast.parse(
            "".join(cell.get("source", [])),
            filename=f"{UIE_CONFIRMATORY_NOTEBOOK.name}:cell-{index}",
        )

    legacy = json.loads(LEGACY_NOTEBOOK.read_text(encoding="utf-8"))
    executed = sum(
        cell.get("execution_count") is not None
        for cell in legacy["cells"]
        if cell.get("cell_type") == "code"
    )
    if executed == 0:
        raise SystemExit("legacy notebook unexpectedly contains no executed cells")

    pyproject = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    package = (REPOSITORY / "src" / "evidencemem" / "__init__.py").read_text(
        encoding="utf-8"
    )
    project_version = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    package_version = re.search(r'^__version__ = "([^"]+)"', package, re.MULTILINE)
    if not project_version or not package_version or project_version[1] != package_version[1]:
        raise SystemExit("package version and project metadata do not match")

    readme = (REPOSITORY / "README.md").read_text(encoding="utf-8").lower()
    if "neurips" in readme:
        raise SystemExit("root README must remain venue-neutral")
    print(
        f"Artifact audit passed: {len(source_code)} clean source cells, "
        f"{len(uie_code)} UIE v4 code cells, "
        f"{len(confirmatory_code)} frozen confirmatory code cells, "
        f"{executed} executed legacy cells, "
        f"package {project_version[1]}."
    )


if __name__ == "__main__":
    main()
