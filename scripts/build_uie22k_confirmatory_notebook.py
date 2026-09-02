"""Build the hard-frozen UIE-22K confirmatory Kaggle notebook."""

from __future__ import annotations

import ast
import copy
import json
import pprint
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_NOTEBOOK = ROOT / "notebooks" / "EvidenceMem_UIE22K_V4_T4.ipynb"
OUTPUT_NOTEBOOK = (
    ROOT / "notebooks" / "EvidenceMem_UIE22K_V4_Confirmatory_T4.ipynb"
)
STATISTICS_CELL = (
    ROOT
    / "scripts"
    / "notebook_cells"
    / "uie22k_v4_confirmatory_stats.py"
)
FROZEN_REPOSITORY_TAG = "uie22k-v4-confirmatory-2026-09-02"
FROZEN_DEVELOPMENT_RECORD = {
    "protocol_id": "uie22k_evidencemem_v4",
    "protocol_revision": "2.0.1",
    "manifest_id": "f9eece5f3f489fd2b986ca89b797c2843e53e2618cd93361b730f0c77bff2c09",
    "development_commit": "0b766243eb6352db067b7de2815e4472cca0c6d2",
    "development_package_tree": "9fec2b475fe962bd92fd8d6e496ec7bcc0b1835e",
    "development_main_cell_blob": "d1adfdc3c15aae64e87b904ad8613786c2af3e5b",
    "development_notebook_sha256": (
        "c394871ec35dd7a29576722b26d19a8b7433d3c284260a58e604ec68c8fbee77"
    ),
    "encoder_key": "siglip2_b16_384",
    "method": "EvidenceMem v4 continuous fusion",
    "default_budget_per_class": 40,
    "sample_seed": 2026,
    "seeds": (7, 17, 29, 43, 61),
    "development_accuracy": 0.9505454545454546,
    "development_macro_f1": 0.9502520045278262,
    "development_ece_15": 0.016757387634576574,
    "confirmatory_hypotheses": (
        {
            "id": "H1_full_knn_noninferiority_1pp",
            "role": "primary",
            "baseline": "Full kNN",
            "alternative": "noninferiority",
            "margin": 0.01,
        },
        {
            "id": "H2_facility_superiority",
            "role": "secondary",
            "baseline": "Facility selection (no reliability) fused",
            "alternative": "superiority",
            "margin": 0.0,
        },
    ),
}


def normalized_source(cell: dict[str, object]) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else str(value)


def set_source(cell: dict[str, object], value: str) -> None:
    cell["source"] = textwrap.dedent(value).strip() + "\n"


def tagged_index(cells: list[dict[str, object]], tag: str) -> int:
    matching = [
        index
        for index, cell in enumerate(cells)
        if tag in cell.get("metadata", {}).get("tags", [])
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one cell tagged {tag!r}, found {len(matching)}")
    return matching[0]


def replace_once(value: str, old: str, new: str, *, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} marker, found {count}")
    return value.replace(old, new, 1)


def code_cell(value: str, tag: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": [],
        "source": textwrap.dedent(value).strip() + "\n",
    }


def freeze_bootstrap(cells: list[dict[str, object]]) -> None:
    index = tagged_index(cells, "bootstrap")
    value = normalized_source(cells[index])
    value = replace_once(
        value,
        'REPOSITORY_REF = os.environ.get("EVIDENCEMEM_REF", "codex/artifact-hardening")',
        (
            "# Immutable tag created for the confirmatory protocol. Do not override it.\n"
            f'REPOSITORY_REF = "{FROZEN_REPOSITORY_TAG}"'
        ),
        label="repository ref",
    )
    set_source(cells[index], value)


def freeze_parameters(cells: list[dict[str, object]]) -> None:
    index = tagged_index(cells, "parameters")
    value = normalized_source(cells[index])
    value = replace_once(
        value,
        '''EVALUATION_STAGE = os.environ.get(
    "EVIDENCEMEM_EVALUATION_STAGE", "development"
).strip().lower()
CONFIRMATORY_ENCODER_KEY = os.environ.get(
    "EVIDENCEMEM_CONFIRMATORY_ENCODER", ""
).strip()''',
        '''# Hard frozen after the completed development run. These values are not
# environment-overridable, which prevents accidental test-set reselection.
EVALUATION_STAGE = "confirmatory"
CONFIRMATORY_ENCODER_KEY = "siglip2_b16_384"''',
        label="evaluation stage",
    )
    frozen_literal = pprint.pformat(
        FROZEN_DEVELOPMENT_RECORD,
        sort_dicts=False,
        width=88,
    )
    value = replace_once(
        value,
        'PROTOCOL_REVISION = "2.0.1"\n',
        (
            'PROTOCOL_REVISION = "2.0.1"\n'
            f"FROZEN_DEVELOPMENT_RECORD = {frozen_literal}\n"
        ),
        label="protocol revision",
    )
    value = replace_once(
        value,
        '''if EVALUATION_STAGE not in {"development", "confirmatory"}:
    raise ValueError(
        "EVALUATION_STAGE must be 'development' or 'confirmatory'."
    )
if EVALUATION_STAGE == "confirmatory" and not CONFIRMATORY_ENCODER_KEY:
    raise ValueError(
        "Set EVIDENCEMEM_CONFIRMATORY_ENCODER to the frozen development "
        "winner before a confirmatory run."
    )''',
        '''if EVALUATION_STAGE != "confirmatory":
    raise ValueError("This dedicated notebook is confirmatory-only.")
if CONFIRMATORY_ENCODER_KEY != FROZEN_DEVELOPMENT_RECORD["encoder_key"]:
    raise ValueError("The confirmatory encoder differs from the frozen winner.")''',
        label="stage validation",
    )
    value = replace_once(
        value,
        'print("Evaluation stage:", EVALUATION_STAGE)\n',
        (
            'print("Evaluation stage:", EVALUATION_STAGE)\n'
            'print("Frozen development record:", FROZEN_DEVELOPMENT_RECORD)\n'
        ),
        label="configuration output",
    )
    set_source(cells[index], value)


def frozen_manifest_guard_cell() -> dict[str, object]:
    return code_cell(
        r'''
        # Confirm that this run uses the exact data split and package implementation
        # that were frozen before the confirmatory labels were evaluated.
        package_tree_id = subprocess.check_output(
            ["git", "rev-parse", "HEAD:src/evidencemem"],
            cwd=REPO_DIR,
            text=True,
        ).strip()
        FROZEN_PROTOCOL_CHECKS = {
            "confirmatory_stage": EVALUATION_STAGE == "confirmatory",
            "paper_mode": CFG.mode == "paper",
            "protocol_id_matches_development": (
                PROTOCOL_ID == FROZEN_DEVELOPMENT_RECORD["protocol_id"]
            ),
            "protocol_revision_matches_development": (
                PROTOCOL_REVISION
                == FROZEN_DEVELOPMENT_RECORD["protocol_revision"]
            ),
            "manifest_matches_development": (
                MANIFEST_ID == FROZEN_DEVELOPMENT_RECORD["manifest_id"]
            ),
            "package_tree_matches_development": (
                package_tree_id
                == FROZEN_DEVELOPMENT_RECORD["development_package_tree"]
            ),
            "encoder_matches_development_selection": (
                CONFIRMATORY_ENCODER_KEY
                == FROZEN_DEVELOPMENT_RECORD["encoder_key"]
                and set(ENCODER_SPECS) == {CONFIRMATORY_ENCODER_KEY}
            ),
            "method_matches_development_selection": (
                FROZEN_DEVELOPMENT_RECORD["method"]
                == "EvidenceMem v4 continuous fusion"
            ),
            "budget_matches_development_selection": (
                CFG.default_budget
                == FROZEN_DEVELOPMENT_RECORD["default_budget_per_class"]
            ),
            "sample_seed_matches_development": (
                CFG.sample_seed == FROZEN_DEVELOPMENT_RECORD["sample_seed"]
            ),
            "seeds_match_development": (
                tuple(CFG.seeds) == tuple(FROZEN_DEVELOPMENT_RECORD["seeds"])
            ),
        }
        failed_frozen_checks = [
            name for name, passed in FROZEN_PROTOCOL_CHECKS.items() if not passed
        ]
        if failed_frozen_checks:
            raise RuntimeError(
                "Frozen confirmatory protocol mismatch: "
                + ", ".join(failed_frozen_checks)
            )
        frozen_protocol_record = {
            "repository_tag": REPOSITORY_REF,
            "confirmatory_source_id": SOURCE_ID,
            "confirmatory_package_tree": package_tree_id,
            "development": FROZEN_DEVELOPMENT_RECORD,
            "checks": FROZEN_PROTOCOL_CHECKS,
        }
        atomic_json(RUN_DIR / "frozen_protocol.json", frozen_protocol_record)
        display(pd.DataFrame([FROZEN_PROTOCOL_CHECKS]))
        ''',
        "frozen-manifest-guard",
    )


def patch_export(cells: list[dict[str, object]]) -> None:
    index = tagged_index(cells, "export")
    value = normalized_source(cells[index])
    value = replace_once(
        value,
        "integrity_checks = {\n",
        '''expected_method_paired_tests = len(ENCODER_DATA) * len(CFG.seeds) * 6
expected_encoder_paired_tests = (
    (len(ENCODER_DATA) - 1) * len(CFG.seeds)
    if "clip_b32_224" in ENCODER_DATA
    else 0
)
expected_paired_tests = expected_method_paired_tests + expected_encoder_paired_tests
expected_hypothesis_ids = {
    item["id"] for item in FROZEN_DEVELOPMENT_RECORD["confirmatory_hypotheses"]
}
hypothesis_numeric_columns = [
    "accuracy_delta",
    "ci_low",
    "ci_high",
    "bootstrap_p_two_sided",
    "seed_sign_flip_p_two_sided",
]

integrity_checks = {
    **FROZEN_PROTOCOL_CHECKS,
    "confirmatory_hypotheses_complete": (
        set(CONFIRMATORY_HYPOTHESES["hypothesis_id"])
        == expected_hypothesis_ids
    ),
    "confirmatory_statistics_finite": bool(
        np.isfinite(
            CONFIRMATORY_HYPOTHESES[hypothesis_numeric_columns].to_numpy(
                dtype=float
            )
        ).all()
    ),
    "paired_tests_complete": len(paired_results) == expected_paired_tests,
''',
        label="integrity checks",
    )
    value = replace_once(
        value,
        '    "result_summary": result_summary,\n',
        '''    "result_summary": result_summary,
    "frozen_development_record": FROZEN_DEVELOPMENT_RECORD,
    "confirmatory_hypotheses": CONFIRMATORY_HYPOTHESES.to_dict(
        orient="records"
    ),
    "hypothesis_support": {
        str(row["hypothesis_id"]): bool(row["supported"])
        for row in CONFIRMATORY_HYPOTHESES.to_dict(orient="records")
    },
''',
        label="claim gate results",
    )
    set_source(cells[index], value)


def build_cells() -> list[dict[str, object]]:
    notebook = json.loads(DEVELOPMENT_NOTEBOOK.read_text(encoding="utf-8"))
    cells = copy.deepcopy(notebook["cells"])
    set_source(
        cells[0],
        r'''
        # EvidenceMem v4 UIE-22K: frozen confirmatory evaluation

        This notebook is the one-shot confirmatory run created after the development
        experiment selected `siglip2_b16_384`. It evaluates only that encoder on the
        untouched 3,300-image confirmatory split. The data manifest, package tree,
        method, 40-image-per-class budget, seeds, and hypotheses are hard frozen.

        **Primary hypothesis.** EvidenceMem at 40 stored images per class is
        non-inferior to full kNN at a predeclared one-percentage-point margin while
        using 30 times fewer stored images.

        **Secondary hypothesis.** Reliability-aware continuous fusion is more
        accurate than the equal-count facility-selection baseline. A failed
        hypothesis remains a valid confirmatory result and must not trigger tuning.
        ''',
    )
    set_source(
        cells[1],
        r'''
        ## Before running on Kaggle

        1. Create a new Kaggle notebook from this file. Do not edit its configuration.
        2. Select a **GPU T4** accelerator and turn **Internet on**.
        3. Add the dataset
           `rhtsingh/130k-images-512x512-universal-image-embeddings`.
        4. Optionally add the successful development notebook as an input so verified
           SigLIP2 train/validation caches can be reused. Confirmatory embeddings are
           always computed from the untouched confirmatory split.
        5. Use **Run All once**. Do not tune any setting after reading these results.
        6. Download the final ZIP. It contains the frozen protocol, raw predictions,
           paired tests, hierarchical confidence intervals, and file hashes.

        The notebook refuses to run if the manifest, package implementation, encoder,
        method, budget, sample seed, or five stochastic seeds differ from the frozen
        development record.
        ''',
    )
    freeze_bootstrap(cells)
    freeze_parameters(cells)

    sampling_index = tagged_index(cells, "uie-sampling")
    cells.insert(sampling_index + 1, frozen_manifest_guard_cell())

    main_index = tagged_index(cells, "main-experiment")
    statistics_source = STATISTICS_CELL.read_text(encoding="utf-8")
    cells.insert(
        main_index + 1,
        code_cell(statistics_source, "confirmatory-statistics"),
    )
    patch_export(cells)
    for index, cell in enumerate(cells):
        cell["id"] = f"uie22k-confirmatory-{index:02d}"
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None or cell.get("outputs"):
                raise RuntimeError(f"Generated code cell {index} retained outputs")
            ast.parse(
                normalized_source(cell),
                filename=f"{OUTPUT_NOTEBOOK.name}:cell-{index}",
            )
    return cells


def main() -> None:
    development = json.loads(DEVELOPMENT_NOTEBOOK.read_text(encoding="utf-8"))
    notebook = {
        "cells": build_cells(),
        "metadata": development.get("metadata", {}),
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
