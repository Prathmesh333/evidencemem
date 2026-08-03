# EvidenceMem

EvidenceMem is a research implementation of a compact, updateable visual
memory over frozen vision-language embeddings. It selects real training images
as class-conditional medoids, scores their reliability, retrieves them as
inspectable decision evidence, fuses visual and class-text scores, and rejects
inputs that lack sufficient support.

The target is the NeurIPS 2026 VLM4RWD workshop. The current repository is a
research work in progress: no accuracy, OOD, or continual-learning claims are
made before the corresponding experiments are complete.

## Research question

> Can a bounded, dynamically updateable prototype memory retain useful
> classification accuracy while exposing the retrieved evidence used by its
> decision rule, supporting class insertion, and improving unknown-input
> detection without updating the vision-language backbone?

## What is being tested

- Class-wise clustering followed by displayable medoid selection.
- Reliability from cluster compactness, neighborhood purity, and class-text
  alignment.
- Reliability-weighted visual retrieval fused with CLIP text-label scores.
- Confidence from score, margin, neighbor agreement, similarity, and evidence
  reliability.
- New-class insertion and bounded-memory pruning without encoder retraining.

Tip-Adapter already established retrieval-based cache adaptation for CLIP, and
deep nearest neighbors are an established OOD baseline. EvidenceMem therefore
does **not** claim novelty for cache-based classification or kNN rejection in
isolation. The empirical question is whether a single reliability-aware,
bounded medoid memory provides a useful joint trade-off across compression,
evidence quality, class insertion, and OOD detection.

## Quick start: lightweight core

The memory and classifier core can be tested without downloading CLIP or image
datasets:

```powershell
python -m pip install -e .[dev]
pytest
python scripts/smoke_core.py
```

For the full vision pipeline, use Python 3.11 and install the vision extra:

```powershell
conda env create -f environment.yml
conda activate evidencemem
python -m pip install -e .[vision,dev]
```

The current machine has a 4 GB GTX 1650. Experiments therefore extract CLIP
embeddings once, cache them, and reuse them across seeds and ablations.

## Project layout

```text
configs/                 Reproducible experiment settings
docs/                    Research plan, literature map, and compute budget
paper/                   Workshop-paper outline and later LaTeX sources
scripts/                 Reproduction and smoke-test entry points
src/evidencemem/         Memory, index, classifier, and confidence code
tests/                   Fast synthetic unit tests
outputs/                 Generated artifacts (large files are gitignored)
```

## Status

- [x] Private GitHub repository and local project initialized
- [x] Claims-to-experiments matrix and initial novelty audit
- [x] Exact cosine/FAISS index abstraction
- [x] Prototype schema and adaptive memory core
- [x] Visual-text classifier and confidence interface
- [x] CLIP extraction entry point, deterministic splits, and cache manifests
- [ ] Download OpenCLIP weights and run the measured 512-image pre-flight
- [ ] Baselines and memory-budget study
- [ ] OOD and continual-class experiments
- [ ] Three-seed ablations and workshop paper

See [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md) for the scientific plan.
