# Claim ledger

This ledger separates implementation facts, mathematical facts, empirical hypotheses,
and statements that must not appear in a paper. A numerical claim may move to
“supported” only when its source run passes `scripts/check_submission_readiness.py`.

| ID | Claim | Type | Required evidence | Status |
|---|---|---|---|---|
| C0 | The implementation returns exactly the requested prototype budget per class when duplicate filtering is disabled. | Software invariant | Unit test on multiple classes and budgets | Verified by tests |
| C1 | The coverage-plus-reliability set objective is non-negative, monotone, and submodular; cardinality-constrained greedy therefore has the standard approximation guarantee. | Mathematical property | Definition with non-negative coefficients and proof sketch | Ready for formal write-up |
| C2 | Reliability-aware selection improves classification over coverage-only selection at an equal stored-image budget. | Primary empirical hypothesis | Corrected three-seed budget curve, matched tuning, paired uncertainty | Pending corrected GPU run |
| C3 | The gain, if any, is not explained only by text fusion or reliability-weighted voting. | Mechanism hypothesis | Selection/voting/fusion factorial ablation | Pending corrected GPU run |
| C4 | EvidenceMem is competitive with a matched Tip-Adapter cache. | Competitive hypothesis | Same encoder, support pool, memory count, split, and validation tuning | Pending corrected GPU run |
| C5 | Retrieved source images have high label precision and help humans inspect errors. | Supporting descriptive claim | Precision@1/3/5, qualitative failures, optional human utility study | Precision pending; utility study not implemented |
| C6 | Classes can be inserted without changing the frozen encoder or old prototype vectors. | Software and empirical claim | Hash old parameters/prototypes; report old/new accuracy and insertion time | Software path present; corrected experiment pending |
| C7 | A combined confidence score improves OOD detection. | Rejected historical hypothesis | Near/far OOD against entropy, MSP, energy, kNN distance | Historical run did not support it; do not claim |

## Prohibited shortcuts

- Do not use the historical notebook as the source of final numerical claims.
- Do not choose the strongest comparator after looking at test accuracy.
- Do not call retrieved examples causal explanations.
- Do not turn a validation-mode subset run into a paper table.
- Do not describe the approximation guarantee as an accuracy guarantee.
- Do not claim superiority when a confidence interval includes zero or when the matched
  baseline wins.

## Claim decision rule

The main paper proceeds as a method paper only if C2 is supported on at least half of the
predeclared budgets and the effect is stable enough to survive paired uncertainty
analysis. Otherwise the correct response is to simplify the method, study when
reliability fails, or write an analysis/negative-result paper.
