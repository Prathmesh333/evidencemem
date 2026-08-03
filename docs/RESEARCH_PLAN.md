# EvidenceMem research plan

## Research question

At a fixed memory budget, can reliability-aware facility selection choose better
displayable prototypes than an otherwise matched coverage-only selector?

This is an empirical hypothesis. The repository currently verifies the implementation
and the set-objective structure, not the accuracy claim.

## Contribution boundary

EvidenceMem does not claim novelty for CLIP, vector search, K-means, medoids, kNN,
prompt ensembling, cache fusion, or retrieving examples. The proposed contribution is the
specific exact-budget selection objective over real-image medoid candidates and a
controlled evaluation of its reliability term.

The paper must compare against Tip-Adapter and other close cache/prototype methods. If the
matched coverage-only selector performs equally well, the proposed mechanism is not
supported even if the combined system beats zero-shot CLIP.

## Work packages

### A. Correctness and provenance

- Keep one method implementation shared by package, tests, and notebook.
- Resolve encoder/checkpoint activation compatibility.
- Hash ordered samples and cached arrays.
- Record the git revision, environment, configuration, and every required result file.
- Keep historical artifacts separate from corrected evidence.

### B. Primary experiment

- Use equal candidate pools and final budgets.
- Compare reliability-aware and coverage-only selection.
- Tune each method on validation data.
- Report the full budget curve over three seeds.
- Preserve per-example outputs for paired tests.

### C. Mechanism ablations

- reliability in selection versus no reliability in selection;
- reliability in voting versus uniform voting;
- compactness, purity, and alignment individually;
- visual-only, text-only, and validation-selected fusion; and
- candidate multiplier and objective-weight sensitivity.

### D. Strong baselines and scale

- matched Tip-Adapter;
- CLIP-Adapter or another trainable lightweight adapter;
- current prototype/cache selection baseline;
- large-scale dataset and natural shifts; and
- more than one encoder family or scale.

### E. Secondary behaviour

- evidence precision and fixed-rule qualitative failures;
- class insertion with old-prototype hashes;
- OOD detection against simple established scores; and
- build/query cost and memory use.

## Decision points

1. Run validation mode. Any missing artifact or cross-cell error stops the full run.
2. Run the corrected three-seed protocol.
3. Proceed to a method paper only if the primary matched-ablation signal passes on at
   least half of the predeclared budgets and has meaningful paired uncertainty.
4. Add the large-scale extension before targeting a highly selective archival venue.
5. If the primary signal fails, narrow, analyze the failure, or stop; do not move the
   goalposts.

The detailed rules are frozen in `docs/EXPERIMENT_PROTOCOL.md`, and claim status is
tracked in `docs/CLAIM_LEDGER.md`.
