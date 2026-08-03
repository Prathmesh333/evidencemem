"""Fail when the source notebook or package metadata can reproduce known artifact bugs."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = REPOSITORY / "notebooks" / "EvidenceMem_Colab_T4.ipynb"
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
        f"{executed} executed legacy cells, package {project_version[1]}."
    )


if __name__ == "__main__":
    main()
