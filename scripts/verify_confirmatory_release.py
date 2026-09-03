"""Audit and summarize the committed UIE-22K confirmatory release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tempfile
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

DEFAULT_RELEASE = Path("results/confirmatory/uie22k-confirmatory")

# These reader-facing figures are regenerated from the verified CSV tables by
# scripts/build_publication_figures.py. The ZIP preserves the original figures, while
# the loose copies intentionally use descriptive publication labels.
PUBLICATION_DERIVED_ARTIFACTS = {
    "calibration_ece.pdf",
    "main_accuracy.pdf",
    "memory_budget_accuracy.pdf",
}

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    temporary.replace(path)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lower) & (confidence <= upper)
        if np.any(mask):
            value += mask.mean() * abs(
                (prediction[mask] == labels[mask]).mean() - confidence[mask].mean()
            )
    return float(value)


def selective_aurc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    prediction = probabilities.argmax(axis=1)
    errors = (prediction != labels).astype(np.float64)
    order = np.argsort(-probabilities.max(axis=1), kind="stable")
    risks = np.cumsum(errors[order]) / np.arange(1, len(errors) + 1)
    coverage = np.arange(1, len(errors) + 1, dtype=np.float64) / len(errors)
    if len(errors) == 1:
        return float(risks[0])
    return float(np.sum((risks[:-1] + risks[1:]) * 0.5 * np.diff(coverage)))


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(scores, np.float64) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip(
        "_"
    )


def load_csv_from_zip(archive: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    text = archive.read(name).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def audit_notebook(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    execution_counts = [cell.get("execution_count") for cell in code_cells]
    error_outputs = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    return {
        "sha256": file_sha256(path),
        "cells": len(notebook["cells"]),
        "code_cells": len(code_cells),
        "executed_code_cells": sum(value is not None for value in execution_counts),
        "execution_counts": execution_counts,
        "error_outputs": len(error_outputs),
    }


def verify_original_manifest(
    release: Path, archive: zipfile.ZipFile
) -> tuple[dict[str, Any], int]:
    manifest = json.loads(archive.read("run_manifest.json"))
    if manifest.get("status") != "complete":
        raise RuntimeError("The source run manifest is not complete")
    verified = 0
    for name, expected in manifest["artifacts"].items():
        payload = archive.read(name)
        if len(payload) != int(expected["bytes"]):
            raise RuntimeError(f"Archive size mismatch: {name}")
        if bytes_sha256(payload) != expected["sha256"]:
            raise RuntimeError(f"Archive hash mismatch: {name}")
        loose_path = release / name
        if name not in PUBLICATION_DERIVED_ARTIFACTS:
            if not loose_path.is_file() or file_sha256(loose_path) != expected["sha256"]:
                raise RuntimeError(f"Loose result does not match source manifest: {name}")
        elif not loose_path.is_file():
            raise RuntimeError(f"Missing publication figure: {name}")
        verified += 1
    return manifest, verified


def recompute_prediction_metrics(
    archive: zipfile.ZipFile,
) -> tuple[dict[str, Any], dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]]:
    rows = load_csv_from_zip(archive, "classification_results.csv")
    packets: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    maximum_error = 0.0
    verified = 0
    metric_names = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "nll",
        "brier",
        "ece_15",
        "ece_15_uncalibrated",
        "nll_uncalibrated",
        "aurc",
        "selective_coverage",
        "selective_accuracy",
    )
    for row in rows:
        seed = int(row["seed"])
        method = row["method"]
        name = (
            f"predictions_{row['encoder_key']}_s{seed}_{safe_filename(method)}.npz"
        )
        with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as packet:
            labels = np.asarray(packet["labels"], dtype=np.int64)
            predictions = np.asarray(packet["predictions"], dtype=np.int64)
            probabilities = np.asarray(packet["probabilities"], dtype=np.float64)
            scores = np.asarray(packet["scores"], dtype=np.float64)
        if labels.shape != (3300,) or probabilities.shape != (3300, 11):
            raise RuntimeError(f"Unexpected prediction shape: {name}")
        if not np.isfinite(probabilities).all() or not np.isfinite(scores).all():
            raise RuntimeError(f"Non-finite prediction values: {name}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-10):
            raise RuntimeError(f"Probability rows do not sum to one: {name}")
        if not np.array_equal(predictions, probabilities.argmax(axis=1)):
            raise RuntimeError(f"Stored predictions differ from probability argmax: {name}")

        one_hot = np.eye(probabilities.shape[1])[labels]
        uncalibrated = softmax(scores)
        threshold = float(row["confidence_threshold_90pct_validation_coverage"])
        retained = probabilities.max(axis=1) >= threshold
        computed = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, average="macro")),
            "nll": float(log_loss(labels, probabilities, labels=np.arange(11))),
            "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
            "ece_15": expected_calibration_error(probabilities, labels),
            "ece_15_uncalibrated": expected_calibration_error(uncalibrated, labels),
            "nll_uncalibrated": float(
                log_loss(labels, uncalibrated, labels=np.arange(11))
            ),
            "aurc": selective_aurc(probabilities, labels),
            "selective_coverage": float(retained.mean()),
            "selective_accuracy": float(accuracy_score(labels[retained], predictions[retained])),
        }
        for metric in metric_names:
            maximum_error = max(maximum_error, abs(computed[metric] - float(row[metric])))
        packets[(seed, method)] = (labels, predictions)
        verified += 1
    if maximum_error > 1e-9:
        raise RuntimeError(f"Recomputed metric mismatch: {maximum_error:.3g}")
    return {
        "prediction_packets": verified,
        "examples_per_packet": 3300,
        "classes": 11,
        "maximum_recomputed_metric_error": maximum_error,
    }, packets


def class_names(release: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with (release / "uie22k_manifest.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            mapping[int(row["label_id"])] = row["label"]
    return mapping


def derive_error_analysis(
    release: Path,
    packets: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    *,
    write_tables: bool,
) -> dict[str, Any]:
    names = class_names(release)
    seeds = (7, 17, 29, 43, 61)
    methods = {
        "evidence": "EvidenceMem v4 continuous fusion",
        "full_knn": "Full kNN",
        "facility": "Facility selection (no reliability) fused",
    }
    per_class_rows: list[dict[str, Any]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    for class_id, class_name in sorted(names.items()):
        scores: dict[str, list[float]] = {key: [] for key in methods}
        for seed in seeds:
            for key, method in methods.items():
                labels, predictions = packets[(seed, method)]
                mask = labels == class_id
                scores[key].append(float(np.mean(predictions[mask] == labels[mask])))
            labels, evidence_predictions = packets[(seed, methods["evidence"])]
            mask = (labels == class_id) & (evidence_predictions != labels)
            confusion.update((class_id, int(value)) for value in evidence_predictions[mask])
        evidence = float(np.mean(scores["evidence"]))
        full_knn = float(np.mean(scores["full_knn"]))
        facility = float(np.mean(scores["facility"]))
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "examples_per_seed": 300,
                "evidencemem_accuracy": evidence,
                "full_knn_accuracy": full_knn,
                "facility_accuracy": facility,
                "evidencemem_minus_full_knn_pp": 100.0 * (evidence - full_knn),
                "evidencemem_minus_facility_pp": 100.0 * (evidence - facility),
            }
        )
    confusion_rows = [
        {
            "true_class": names[true_id],
            "predicted_class": names[predicted_id],
            "count_across_five_seeds": count,
        }
        for (true_id, predicted_id), count in confusion.most_common()
    ]
    if write_tables:
        atomic_csv(
            release / "per_class_accuracy.csv",
            list(per_class_rows[0]),
            per_class_rows,
        )
        atomic_csv(
            release / "confusion_summary.csv",
            list(confusion_rows[0]),
            confusion_rows,
        )
    return {
        "largest_evidencemem_shortfalls_vs_full_knn": sorted(
            per_class_rows, key=lambda row: row["evidencemem_minus_full_knn_pp"]
        )[:3],
        "most_common_evidencemem_confusions": confusion_rows[:3],
    }


def build_release_manifest(release: Path, audit: dict[str, Any]) -> dict[str, Any]:
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in sorted(release.iterdir())
        if path.is_file() and path.name != "release_manifest.json"
    }
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(UTC).isoformat(),
        "release_id": release.name,
        "source_run": "paper_confirmatory_uie22k_evidencemem_7ce2d2de",
        "files": files,
        "audit": audit,
    }


def verify_release_manifest(release: Path) -> int:
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    expected_names = {
        path.name
        for path in release.iterdir()
        if path.is_file() and path.name != "release_manifest.json"
    }
    if set(manifest["files"]) != expected_names:
        raise RuntimeError("The release manifest does not cover every published file")
    for name, expected in manifest["files"].items():
        path = release / name
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Published file size mismatch: {name}")
        if file_sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Published file hash mismatch: {name}")
    return len(expected_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate derived error-analysis tables and the release manifest.",
    )
    arguments = parser.parse_args()
    release = arguments.release.resolve()
    notebook_path = release / "EvidenceMem_UIE22K_Confirmatory_T4_executed.ipynb"
    archive_path = release / "full_run_with_predictions.zip"
    notebook_audit = audit_notebook(notebook_path)
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt_member}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("The result archive contains duplicate member names")
        run_manifest, manifest_artifacts = verify_original_manifest(release, archive)
        prediction_audit, packets = recompute_prediction_metrics(archive)
        error_analysis = derive_error_analysis(
            release, packets, write_tables=arguments.refresh
        )
        archive_audit = {
            "sha256": file_sha256(archive_path),
            "members": len(names),
            "source_manifest_artifacts_verified": manifest_artifacts,
            "source_manifest_status": run_manifest["status"],
            "duplicate_member_names": False,
            "crc_clean": True,
        }
    audit = {
        "notebook": notebook_audit,
        "archive": archive_audit,
        "predictions": prediction_audit,
        "error_analysis": error_analysis,
    }
    if arguments.refresh:
        atomic_json(release / "paper_analysis.json", audit)
        atomic_json(release / "release_manifest.json", build_release_manifest(release, audit))
    published_files = verify_release_manifest(release)
    if notebook_audit["error_outputs"] or notebook_audit["executed_code_cells"] != 14:
        raise RuntimeError("The published notebook is not a clean 14-cell executed run")
    if not math.isclose(prediction_audit["maximum_recomputed_metric_error"], 0.0, abs_tol=1e-9):
        raise RuntimeError("Prediction metrics do not reproduce the published table")
    print(
        "CONFIRMATORY RELEASE VERIFIED: "
        f"{published_files} files, {prediction_audit['prediction_packets']} prediction packets"
    )


if __name__ == "__main__":
    main()
