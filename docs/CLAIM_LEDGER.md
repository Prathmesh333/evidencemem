# Claim ledger

This ledger separates verified facts, unsupported hypotheses, exploratory observations,
and prohibited claims. The only final numerical source is
`results/confirmatory/uie22k-v4-7ce2d2de/`.

| ID | Claim | Evidence | Decision |
|---|---|---|---|
| C0 | The implementation returns the requested number of prototypes per class when duplicate filtering is disabled. | Unit tests across classes and budgets | Verified software invariant |
| C1 | The coverage-only facility objective is nonnegative, monotone, and submodular. Cardinality-constrained greedy selection has the standard approximation guarantee for that objective. | Objective definition, implementation, and cited result | Verified mathematical property; not an accuracy guarantee |
| C2 | Reliability-aware v4 scoring is more accurate than the equal-count facility baseline. | Mean difference +0.1212 points; hierarchical 95% interval [-0.1030, +0.3515] | Not supported |
| C3 | The continuous query-reliability gate provides a material benefit over fixed fusion. | Continuous 94.9273%; fixed 94.9212%; four of five selected gate slopes are zero | Not supported |
| C4 | EvidenceMem outperforms the matched Tip-Adapter implementation on this split. | 94.9273% vs. 91.5758% at 440 stored images | Verified descriptive result for this implementation and split; not a general superiority claim |
| C5 | Retrieved neighbors have high label agreement. | Mean global neighbor-label precision 0.8646 at 40 images per class | Verified descriptive metric; no human-utility claim |
| C6 | A 40-per-class EvidenceMem is non-inferior to full kNN at a one-point margin. | Mean difference -0.9818 points; hierarchical 95% interval [-1.2850, -0.6788] | Not supported because the lower bound is below -1.0 |
| C7 | The 440-image memory uses 30 times fewer stored source images than full kNN. | 13,200 divided by 440 | Verified image-count comparison |
| C8 | EvidenceMem reaches 94.9273% mean accuracy and 94.8777% mean macro F1 on the frozen confirmatory split. | Five memory seeds; 3,300 confirmatory examples per seed | Verified final result |
| C9 | The published metrics can be reproduced from the raw predictions. | 65 prediction packets; maximum metric difference 2.22e-16 | Verified release audit |
| C10 | An 80-per-class memory reaches 95.4364% accuracy. | Five-seed budget curve | Verified exploratory result; not a new confirmatory claim |
| C11 | A combined confidence score improves out-of-distribution detection. | Historical experiment did not support the hypothesis | Rejected historical hypothesis; do not claim |

## Interpretation rules

- Do not call C6 non-inferior because its point estimate is within one point. The frozen
  rule depends on the lower confidence bound.
- Do not describe a failed non-inferiority test as proof that the method is inferior by
  more than one point. The interval crosses the decision boundary.
- Do not convert C2 into a reliability benefit by selecting a favorable seed, class,
  budget, or metric after the confirmatory run.
- Do not describe the v4 support set as reliability-selected. V4 reweights and scores the
  prototypes selected by the coverage-only facility method.
- Do not call retrieved images causal explanations. They are inspectable associations.
- Do not describe 512-pixel source images as native 512-pixel encoder inputs. The frozen
  SigLIP 2 checkpoint uses 384-pixel inputs.
- Do not generalize the result beyond the tested dataset, split, encoder, and notebook
  implementation.

## Final decision

The confirmatory run is complete and valid. Both predeclared hypotheses are unsupported.
The paper can present the run as an audited compression study with a negative reliability
result. A new reliability claim requires a separately registered experiment on untouched
data. The current confirmatory split must not be tuned again.
