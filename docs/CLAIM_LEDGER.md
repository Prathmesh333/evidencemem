# Claim ledger

This ledger separates implementation facts, mathematical facts, empirical hypotheses,
and statements that must not appear in a paper. A numerical claim may move to
“supported” only when its source run passes `scripts/check_submission_readiness.py`.

| ID | Claim | Type | Required evidence | Status |
|---|---|---|---|---|
| C0 | The implementation returns exactly the requested prototype budget per class when duplicate filtering is disabled. | Software invariant | Unit test on multiple classes and budgets | Verified by tests |
| C1 | The coverage-plus-reliability set objective is non-negative, monotone, and submodular; cardinality-constrained greedy therefore has the standard approximation guarantee. | Mathematical property | Definition with non-negative coefficients and proof sketch | Ready for formal write-up |
| C2 | Reliability-aware selection improves classification over coverage-only selection at an equal stored-image budget. | Mechanism hypothesis | Frozen UIE-22K confirmatory split, matched 40-per-class budget, hierarchical paired interval | Development gain was 0.0303 points on the selected encoder; confirmatory test pending |
| C3 | The gain, if any, is not explained only by text fusion or reliability-weighted voting. | Mechanism hypothesis | Selection/voting/fusion factorial ablation | Pending corrected GPU run |
| C4 | EvidenceMem is competitive with a matched Tip-Adapter cache. | Competitive hypothesis | Same encoder, support pool, memory count, split, and validation tuning | Pending corrected GPU run |
| C5 | Retrieved source images have high label precision and help humans inspect errors. | Supporting descriptive claim | Precision@1/3/5, qualitative failures, optional human utility study | Precision pending; utility study not implemented |
| C6 | Classes can be inserted without changing the frozen encoder or old prototype vectors. | Software and empirical claim | Hash old parameters/prototypes; report old/new accuracy and insertion time | Software path present; corrected experiment pending |
| C7 | A combined confidence score improves OOD detection. | Rejected historical hypothesis | Near/far OOD against entropy, MSP, energy, kNN distance | Historical run did not support it; do not claim |
| C8 | A 40-per-class EvidenceMem is within one accuracy point of full kNN while storing 30 times fewer images. | Primary compression hypothesis | Frozen UIE-22K confirmatory split and hierarchical paired non-inferiority interval with a predeclared 1-point margin | Development difference was -0.8849 points; confirmatory test pending |

## Prohibited shortcuts

- Do not use the historical notebook as the source of final numerical claims.
- Do not choose the strongest comparator after looking at test accuracy.
- Do not call retrieved examples causal explanations.
- Do not turn a validation-mode subset run into a paper table.
- Do not describe the approximation guarantee as an accuracy guarantee.
- Do not claim superiority when a confidence interval includes zero or when the matched
  baseline wins.

## Claim decision rule

An accuracy-improvement paper requires C2 to survive the frozen paired uncertainty
analysis. A memory-compression paper requires C8 to pass its predeclared non-inferiority
margin on the untouched confirmatory split. If C8 passes but C2 fails, the paper must be
framed around compression rather than reliability superiority. If both fail, the correct
response is to study the negative result or register a genuinely new experiment; the
confirmatory split may not be tuned again.
