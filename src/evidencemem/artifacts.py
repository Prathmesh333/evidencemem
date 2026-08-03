"""Provenance helpers for auditable experiment outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def git_revision(repository: str | Path = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def package_versions(names: Sequence[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(
            [
                "evidencemem",
                "numpy",
                "scikit-learn",
                "scipy",
                "torch",
                "torchvision",
                "open-clip-torch",
                "faiss-cpu",
                "pandas",
            ]
        ),
    }
    try:
        import torch

        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        snapshot["cuda_available"] = False
        snapshot["cuda_version"] = None
        snapshot["gpu"] = None
    return snapshot


def start_run_manifest(
    *,
    config: Mapping[str, Any],
    repository: str | Path,
    model_name: str,
    pretrained: str,
) -> dict[str, Any]:
    resolved_config = {
        "experiment": dict(config),
        "encoder": {"model_name": model_name, "pretrained": pretrained},
    }
    config_hash = canonical_json_sha256(resolved_config)
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "completed_utc": None,
        "run_id": f"{config.get('mode', 'run')}-{config_hash[:12]}",
        "source_revision": git_revision(repository),
        "config_sha256": config_hash,
        "config": resolved_config,
        "encoder": {"model_name": model_name, "pretrained": pretrained},
        "environment": environment_snapshot(),
        "artifacts": {},
    }


def finalize_run_manifest(
    manifest: Mapping[str, Any],
    *,
    run_directory: str | Path,
    required_artifacts: Sequence[str],
) -> dict[str, Any]:
    directory = Path(run_directory)
    missing = [name for name in required_artifacts if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cannot finalize run; missing artifacts: {missing}")
    completed = dict(manifest)
    completed["status"] = "complete"
    completed["completed_utc"] = datetime.now(UTC).isoformat()
    completed["artifacts"] = {
        name: {"sha256": file_sha256(directory / name), "bytes": (directory / name).stat().st_size}
        for name in sorted(required_artifacts)
    }
    return completed
