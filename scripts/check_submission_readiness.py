"""Check artifact readiness and report the remaining archival-submission gaps."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_RELEASE = Path("results/confirmatory/uie22k-confirmatory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--target",
        choices=("paper-draft", "archival-submission"),
        default="paper-draft",
    )
    arguments = parser.parse_args()

    verifier = Path(__file__).with_name("verify_confirmatory_release.py")
    subprocess.run(
        [sys.executable, str(verifier), "--release", str(arguments.release)],
        check=True,
    )
    required_paper_files = (
        Path("paper/main.tex"),
        Path("paper/references.bib"),
        Path("paper/main.pdf"),
    )
    missing = [str(path) for path in required_paper_files if not path.is_file()]
    if missing:
        raise SystemExit("NOT READY: missing paper files: " + ", ".join(missing))

    if arguments.target == "paper-draft":
        print("READY: verified confirmatory artifacts and compiled paper draft are present")
        return

    gaps = (
        "a registered external-dataset confirmatory result",
        "a second encoder family",
        "serialized-byte storage measurements",
        "encoder-inclusive latency measurements",
        "a direct evidence-fidelity or human-audit measure",
    )
    raise SystemExit(
        "NOT READY FOR A STRONG ARCHIVAL SUBMISSION: " + "; ".join(gaps)
    )


if __name__ == "__main__":
    main()
