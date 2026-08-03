"""Check whether a corrected full run supports the paper's primary empirical claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/corrected"))
    arguments = parser.parse_args()

    manifests = sorted(arguments.results.glob("*/run_manifest.json"))
    if not manifests:
        raise SystemExit(
            "NOT READY: no corrected full-run manifest exists under results/corrected/."
        )
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("config", {}).get("experiment", {})
    failures: list[str] = []
    if manifest.get("status") != "complete":
        failures.append("run manifest is not complete")
    if config.get("mode") != "paper":
        failures.append("latest corrected run is not paper mode")
    if len(config.get("seeds", [])) < 3:
        failures.append("fewer than three seeds")
    if int(config.get("test_size", 0)) != 10_000:
        failures.append("test set is not the full 10,000 examples")

    run_directory = manifest_path.parent
    gate_path = run_directory / "claim_validation.json"
    if not gate_path.is_file():
        failures.append("claim_validation.json is missing")
    else:
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        primary = gate.get("signals", {}).get(
            "primary_reliability_gain_on_at_least_half_of_budgets", False
        )
        if not primary:
            failures.append("primary matched-ablation signal did not pass")
        if not str(gate.get("go_no_go", "")).startswith("GO:"):
            failures.append("claim gate is NO-GO")

    if failures:
        raise SystemExit("NOT READY: " + "; ".join(failures))
    print(f"READY FOR PAPER DRAFTING: verified {manifest_path}")


if __name__ == "__main__":
    main()
