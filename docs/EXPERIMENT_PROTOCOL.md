# Frozen experiment protocol

## Primary question

Does reliability-aware facility selection improve a bounded real-image memory over an
otherwise matched coverage-only selector?

The comparison fixes the encoder, normalized embeddings, class split, candidate
multiplier, final prototype count, prompt ensemble, validation set, retrieval rule, and
hyperparameter search space. The treatment adds reliability to selection and voting.

## Protocol levels

### Level 0: software validation

- Synthetic unit tests only.
- Confirms exact budgets, serialization, score endpoints, tamper rejection, and manifest
  finalization.
- Supports no dataset-performance claim.

### Level 1: Colab validation mode

- One seed and bounded CIFAR subsets.
- Runs every experiment and creates every required artifact.
- Detects memory, runtime, and cross-cell failures.
- Supports no final numerical claim.

### Level 2: corrected full run

- Seeds: 7, 17, and 29.
- CIFAR-10: 45,000 train, 5,000 validation, 10,000 official test examples.
- Secondary CIFAR-100 classification and CIFAR-100/SVHN OOD evaluation.
- Full predeclared memory-budget, top-k, and text-weight grids.
- This is the minimum evidence needed to decide whether the idea merits a paper draft.

### Level 3: strong archival submission

The CIFAR study alone is not broad enough for a highly selective archival venue. After a
positive Level 2 gate, add at least:

- ImageNet-1K or a predeclared large-scale subset with public split indices;
- natural distribution shifts such as ImageNet-V2, ImageNet-R, ImageNet-A, and
  ImageNet-Sketch where licences and compute allow;
- at least one additional encoder family or scale;
- Tip-Adapter, CLIP-Adapter, and a current prototype/cache baseline implemented from
  official code or checked equations; and
- memory-build time, peak RAM/VRAM, warm query latency, and stored bytes.

Level 3 must be a separate registered configuration. It must not be added selectively
after seeing which datasets favour the method.

## Data and split rules

1. Persist stratified train/validation indices before tuning.
2. Record source dataset versions and sample IDs.
3. Use one embedding cache per exact model, checkpoint, preprocessing pipeline, split,
   sample order, dtype, and source revision.
4. Reject a cache whose array hashes or manifest fields differ.
5. Never tune on official test labels.
6. Use the same cached embeddings for every frozen-encoder method.

## Methods

### Primary method

- Overcluster to twice the final class budget.
- Score compactness, local purity, and text alignment.
- Greedily select an exact budget with coverage weight 0.75 and reliability weight 0.25.
- Use reliability-weighted visual voting.
- Tune retrieval depth and text weight on validation data.

### Required matched controls

- Coverage-only facility selection with uniform visual voting.
- Reliability-selected prototypes with uniform voting.
- Coverage-only selected prototypes with reliability voting, if reliability values can be
  assigned without changing selection.
- Plain K-means medoids.
- Random equal-count support memory.
- Tip-Adapter with the same random support memory and count.

### Context baselines

- Frozen CLIP zero-shot.
- Conventional similarity-weighted full kNN.
- One centroid-nearest real example per class.
- Fixed linear probe.
- Supervised image model, reported separately because it changes the training regime.

## Tuning

- Tune each memory method separately; do not reuse EvidenceMem’s selected values for its
  baselines.
- Break exact validation ties deterministically toward smaller (k), then smaller text
  weight.
- Select calibration temperature on validation negative log likelihood.
- Tune Tip-Adapter beta and cache weight only on validation data.
- Save the complete validation grid, not only its winner.

## Metrics and uncertainty

Classification:

- top-1 accuracy, macro F1, NLL, and 15-bin ECE;
- mean and standard deviation across stochastic seeds;
- per-seed paired bootstrap intervals for accuracy differences; and
- exact McNemar tests for predeclared primary comparisons.

Efficiency:

- stored image count and compression ratio;
- serialized bytes and float32 vector bytes;
- prototype construction time;
- warm batch-one and batch-1,000 query latency; and
- peak host and accelerator memory.

OOD analysis is secondary. Report AUROC, AUPR-OOD, and FPR95 for near and far shifts.
Compare with text MSP, maximum prototype similarity, fused MSP, predictive entropy,
probability margin, and an energy-style score. A losing score remains a negative result.

Continual insertion is also secondary. Report old-class accuracy before and after,
new-class accuracy, average accuracy, forgetting, insertion time, encoder updates, and a
hash confirming that old prototypes were unchanged.

## Run completion

A run is complete only if all required CSV, JSON, prediction, figure, environment, and
journal files exist and are hashed by `run_manifest.json`. Interrupted directories remain
`status: running` and may not be cited.

After copying a full archive into `results/corrected/<run-id>/`, run:

```bash
python scripts/check_submission_readiness.py
```
