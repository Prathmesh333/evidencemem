"""Build publication-facing figures from the verified confirmatory tables.

The raw result files keep their immutable run identifiers. This script maps those
identifiers to descriptive labels before it creates figures for readers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "confirmatory" / "uie22k-confirmatory"
DEFAULT_PAPER_FIGURES = ROOT / "paper" / "figures"

COLORS = {
    "evidence": "#0072B2",
    "baseline": "#7A7A7A",
    "control": "#D55E00",
    "random": "#CC79A7",
}


def _finish(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches="tight", metadata={"Creator": "EvidenceMem"})
    plt.close()


def build_main_accuracy(results: Path, destinations: list[Path]) -> None:
    frame = pd.read_csv(results / "publication_classification_summary.csv")
    order = [
        "Zero-shot text classifier",
        "Tip-Adapter matched cache",
        "EvidenceMem global retrieval",
        "Facility selection without reliability",
        "EvidenceMem fixed fusion",
        "EvidenceMem reliability-weighted fusion",
        "Full kNN",
        "Linear probe",
    ]
    labels = {
        "Zero-shot text classifier": "Zero-shot text",
        "Tip-Adapter matched cache": "Tip-Adapter\n(matched cache)",
        "EvidenceMem global retrieval": "EvidenceMem\nglobal retrieval",
        "Facility selection without reliability": "Facility selection\n(no reliability)",
        "EvidenceMem fixed fusion": "EvidenceMem\nfixed fusion",
        "EvidenceMem reliability-weighted fusion": "EvidenceMem\nreliability fusion",
        "Full kNN": "Full kNN",
        "Linear probe": "Linear probe",
    }
    plot = frame.set_index("method").loc[order].reset_index()
    colors = [
        COLORS["evidence"] if name.startswith("EvidenceMem") else
        COLORS["control"] if name.startswith("Facility") else
        COLORS["baseline"]
        for name in plot["method"]
    ]

    plt.figure(figsize=(10.2, 3.3))
    bars = plt.bar(
        range(len(plot)),
        plot["accuracy_percent"],
        yerr=plot["accuracy_std_percent"],
        capsize=3,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
    )
    plt.xticks(range(len(plot)), [labels[name] for name in plot["method"]], fontsize=8)
    plt.ylabel("Top-1 accuracy (%)")
    plt.ylim(86, 98)
    plt.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    plt.gca().set_axisbelow(True)
    for bar, value in zip(bars, plot["accuracy_percent"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.32, f"{value:.2f}",
                 ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    _finish(destinations)


def build_budget(results: Path, destinations: list[Path]) -> None:
    frame = pd.read_csv(results / "memory_budget_summary.csv")
    label_map = {
        "EvidenceMem v4 continuous fusion": "EvidenceMem reliability fusion",
        "Facility selection (no reliability)": "Facility selection",
        "Random memory": "Random memory",
    }
    frame = frame[frame["method"].isin(label_map)].copy()
    frame["label"] = frame["method"].map(label_map)
    styles = {
        "EvidenceMem reliability fusion": (COLORS["evidence"], "o"),
        "Facility selection": (COLORS["control"], "s"),
        "Random memory": (COLORS["random"], "^"),
    }

    plt.figure(figsize=(4.5, 3.25))
    for label, group in frame.groupby("label", sort=False):
        group = group.sort_values("budget_per_class")
        color, marker = styles[label]
        plt.errorbar(
            group["budget_per_class"],
            100 * group["accuracy_mean"],
            yerr=100 * group["accuracy_std"],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            capsize=3,
        )
    plt.xlabel("Stored images per class")
    plt.ylabel("Top-1 accuracy (%)")
    plt.xticks(sorted(frame["budget_per_class"].unique()))
    plt.grid(color="#D9D9D9", linewidth=0.6, alpha=0.8)
    plt.legend(frameon=False, fontsize=7.5, loc="lower right")
    plt.tight_layout()
    _finish(destinations)


def build_calibration(results: Path, destinations: list[Path]) -> None:
    frame = pd.read_csv(results / "publication_classification_summary.csv")
    order = [
        "Full kNN",
        "EvidenceMem reliability-weighted fusion",
        "Facility selection without reliability",
        "K-means medoids",
        "Random memory",
        "Tip-Adapter matched cache",
        "Zero-shot text classifier",
    ]
    labels = {
        "Full kNN": "Full kNN",
        "EvidenceMem reliability-weighted fusion": "EvidenceMem reliability fusion",
        "Facility selection without reliability": "Facility selection",
        "K-means medoids": "K-means medoids",
        "Random memory": "Random memory",
        "Tip-Adapter matched cache": "Tip-Adapter",
        "Zero-shot text classifier": "Zero-shot text",
    }
    plot = frame.set_index("method").loc[order].reset_index()
    colors = [COLORS["evidence"] if name.startswith("EvidenceMem") else COLORS["baseline"]
              for name in plot["method"]]
    plt.figure(figsize=(6.6, 3.5))
    plt.barh(range(len(plot)), plot["ece_percent"], color=colors)
    plt.yticks(range(len(plot)), [labels[name] for name in plot["method"]], fontsize=8)
    plt.xlabel("Expected calibration error (%)")
    plt.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    _finish(destinations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--paper-figures", type=Path, default=DEFAULT_PAPER_FIGURES)
    args = parser.parse_args()

    results = args.results.resolve()
    paper = args.paper_figures.resolve()
    build_main_accuracy(
        results,
        [results / "main_accuracy.pdf", paper / "main_accuracy.pdf"],
    )
    build_budget(
        results,
        [results / "memory_budget_accuracy.pdf", paper / "memory_budget_accuracy.pdf"],
    )
    build_calibration(
        results,
        [results / "calibration_ece.pdf", paper / "calibration_ece.pdf"],
    )
    print(f"Wrote publication figures to {results} and {paper}")


if __name__ == "__main__":
    main()
