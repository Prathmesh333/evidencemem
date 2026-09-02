# EvidenceMem research plan

## Completed study

The UIE-22K v4 confirmatory study is complete. A 440-image memory reached 94.9273%
accuracy. Full kNN reached 95.9091% with 13,200 stored images. The primary
non-inferiority rule did not pass because the 95% interval crossed the predeclared
one-point boundary.

Reliability-aware scoring exceeded matched facility selection by 0.1212 accuracy points.
The 95% interval included zero. The secondary superiority rule did not pass.

These results close the current protocol. Do not tune the confirmatory split again.

## Next research question

On a new dataset, does class-conditional prototype scoring improve a coverage-selected
memory? If it does, does reliability weighting add a measurable gain after support set,
retrieval depth, and text fusion are matched?

This sequence isolates two mechanisms that v4 combined. It also responds to the current
result: continuous fusion differed from fixed fusion by only 0.0061 accuracy points, and
four of five validation runs selected a zero gate slope.

## Contribution boundary

EvidenceMem does not claim novelty for a frozen vision--language encoder, vector search,
K-means, medoids, kNN, prompt ensembling, cache fusion, or retrieved examples. The next
study can claim a scoring contribution only if a registered, matched ablation supports
that mechanism on untouched data.

The facility objective supplies the exact-budget support set. Its submodularity is an
established mathematical fact, not a new theorem or an accuracy guarantee.

## Required work

### 1. Register a new protocol

- Choose an external dataset before examining method results.
- Save content hashes and fixed train, validation, and confirmatory indices.
- Declare one primary metric, one effect boundary, and one uncertainty procedure.
- Freeze the encoder, memory budget, seeds, and model-selection rule.

### 2. Isolate the mechanism

Use a small factorial comparison at an equal stored-image budget:

- global retrieval versus class-conditional retrieval;
- uniform prototype weights versus reliability weights; and
- visual-only scores versus fixed text fusion.

Do not add a continuous gate unless development data shows a stable nonzero slope across
seeds. Keep the confirmatory hypothesis limited to one mechanism.

### 3. Improve external validity

- Add a second encoder family.
- Include natural distribution shifts when the dataset supports them.
- Compare with Tip-Adapter and a current cache-compression or prototype baseline.
- Report class-level results for visually and semantically overlapping categories.

### 4. Measure system cost

- Count serialized bytes for images, embeddings, labels, and learned parameters.
- Measure encoder-inclusive latency and memory-only scoring latency separately.
- Report memory-build time and peak host and accelerator memory.
- Keep hardware, batch size, warm-up, and repetition count fixed.

### 5. Test evidence quality

- Define evidence precision before evaluation.
- Measure whether returned prototypes agree with the predicted and true labels.
- Use a fixed rule to select qualitative successes and failures.
- Add a human audit only after the task, sampling rule, and evaluation form are fixed.

## Decision rule

Proceed with a positive mechanism paper only if the registered matched ablation passes.
If the mechanism fails again, present it as a negative result or remove the mechanism.
Do not search for a favorable class, seed, budget, or metric after the confirmatory run.

The active protocol is in `docs/EXPERIMENT_PROTOCOL.md`. Claim status is in
`docs/CLAIM_LEDGER.md`. The verified result archive is in
`results/confirmatory/uie22k-v4-7ce2d2de/`.
