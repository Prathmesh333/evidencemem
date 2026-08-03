# EvidenceMem research plan

## One-sentence contribution

EvidenceMem studies whether reliability-aware, displayable medoids can serve as
a single bounded external memory for compressed CLIP classification,
inspectable retrieval evidence, online class insertion, and unknown-input
rejection without updating the encoder.

This is a hypothesis, not an established result.

## What, why, and so what

- **What:** A class-conditional medoid memory in which compactness, local label
  purity, and class-text alignment produce a reliability score used during
  prototype retention, retrieval voting, confidence estimation, and pruning.
- **Why:** A controlled suite of memory-budget, fusion, OOD, evidence-quality,
  and incremental-class experiments will test each role independently.
- **So what:** If the hypotheses hold, one frozen representation could support
  changing label sets and inspectable decisions with less stored data and no
  classifier-head retraining.

## Novelty boundary

The following are prior art and are not novelty claims:

- CLIP provides the joint image-text embedding space and prompt-based zero-shot
  classification.
- Tip-Adapter constructs a key-value cache from labeled visual features and
  combines retrieval with CLIP knowledge for few-shot adaptation.
- Deep kNN work establishes nearest-neighbor distance as an OOD signal.
- Existing continual-CLIP methods study class-incremental adaptation.

EvidenceMem must therefore earn its contribution through the **joint design and
evaluation** of bounded real-image medoids, reliability-aware selection and
decision making, inspectable evidence, and class insertion. If the full method
does not beat fair component baselines, the paper should be reframed as an
analysis or negative result.

## Claims-to-experiments matrix

| ID | Provisional claim | Primary experiment | Required comparison | Success signal |
|---|---|---|---|---|
| H1 | Compact memory retains useful classification accuracy | CIFAR-10/100 classification | full kNN, centroid, linear probe | small gap to full kNN at 5-10% memory |
| H2 | Reliability-aware medoids use memory better | prototype-budget curve | random exemplars, plain KMeans medoids | higher mean accuracy at equal count |
| H3 | Visual-text fusion helps | lambda sweep | text-only and visual-only endpoints | validation-selected fusion improves accuracy or OOD |
| H4 | Classes can be inserted without encoder updates | staged CIFAR-10 protocol | zero-shot, fixed/retrained linear probe | competitive new-class accuracy with measured insertion time |
| H5 | Combined evidence confidence helps reject unknowns | CIFAR-10 vs SVHN/CIFAR-100 | max text score, kNN similarity, entropy, margin | higher AUROC and/or lower FPR95 |
| H6 | Retrieved prototypes are useful decision evidence | Precision@1/3/5 and failure taxonomy | random and plain-medoid memory | high label precision with honest failure cases |

## Experimental rules

1. Create stratified train/validation splits once and persist their indices.
2. Tune thresholds, temperatures, fusion weights, and `k` on validation data
   only; keep final test sets untouched.
3. Cache encoder embeddings and reuse the exact cache across methods.
4. Use the same prototype count for all memory-selection comparisons.
5. Report mean and standard deviation over three seeds where randomness exists.
6. Measure warm query latency separately from one-time embedding extraction.
7. Preserve per-example predictions so paired tests and error analysis remain
   possible.
8. Describe retrieved images as evidence used by the decision rule, never as a
   causal explanation.

## Implementation phases

### Phase A: scientific core

- Exact/FAISS inner-product index abstraction.
- Class-wise clustering, medoid selection, reliability, deduplication.
- Visual score aggregation, text scores, fusion, confidence components.
- Online update, new-class insertion, bounded pruning, serialization.

### Phase B: reproducible data pipeline

- Deterministic CIFAR-10, CIFAR-100, and SVHN downloads and splits.
- OpenCLIP ViT-B/32 image and text encoding.
- Crash-safe embedding cache with manifest and checksums.

### Phase C: baselines and experiments

- Zero-shot CLIP, full kNN, centroid, random memory, plain medoids, linear probe.
- Memory-budget, fusion, top-k, OOD, continual, noise, and ablation runs.
- Per-example outputs, summary CSVs, bootstrapped confidence intervals, and
  paired McNemar tests for key classification comparisons.

### Phase D: paper and review

- Results-first figures and experiment log.
- Eight-page anonymized NeurIPS 2026 workshop paper.
- Claim-to-result verification and critical simulated reviews.

## Go/no-go checkpoint

After the memory-budget and OOD pilots, continue with the full story only if at
least two of the following hold:

1. Reliability-aware medoids beat plain medoids or random exemplars.
2. Fusion improves accuracy or OOD without degrading the other materially.
3. Combined confidence improves over maximum similarity.
4. Evidence Precision@k remains strong at a compact memory budget.

Otherwise, simplify the contribution or report a rigorous negative finding.

