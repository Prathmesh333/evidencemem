# Paper workspace

The paper is not ready for numerical drafting. The corrected full run must pass
`scripts/check_submission_readiness.py` first. Until then, this directory contains the
argument and table plan rather than invented prose or copied historical numbers.

## Working title

**EvidenceMem: Reliability-Aware Facility Selection for Bounded Visual Memory**

## One-sentence contribution

EvidenceMem selects an exact budget of real-image medoids by greedily optimizing class
coverage plus prototype reliability, and evaluates whether this selection improves a
frozen vision-language memory over matched coverage-only and cache-adapter baselines.

## Paper argument

1. Frozen vision-language embeddings make external memory attractive, but storing every
   example is costly and plain clustering ignores whether a candidate is trustworthy.
2. Candidate medoids are displayable real images. A coverage-plus-reliability objective
   gives a precise selection rule under a fixed budget.
3. The mathematical contribution is modest but clean: the objective is monotone
   submodular, so exact-budget greedy selection has a standard approximation guarantee.
4. The empirical question is whether the reliability term helps when memory count,
   encoder, support pool, tuning, and scoring are matched.
5. Evidence retrieval, fusion, class insertion, and OOD behaviour describe the method’s
   operating characteristics. They are not separate novelty claims.

## Planned sections

1. Abstract: problem, exact method, strongest supported result, limitation.
2. Introduction: bounded external memory and why selection reliability matters.
3. Related work: cache adapters, prototype/coreset selection, retrieval evidence,
   continual frozen representations, and OOD scoring.
4. Method: candidates, reliability, set objective, greedy algorithm, voting, and cost.
5. Experiments: primary matched comparison, strong baselines, budget curve, ablations,
   scale/generalization, efficiency, and secondary analyses.
6. Limitations: representation dependence, label dependence of purity, privacy, evidence
   semantics, dataset scale, and unsuccessful hypotheses.
7. Conclusion.

## Predeclared tables and figures

- Table 1: classification and efficiency at the default matched memory budget.
- Figure 1: accuracy versus stored images, with per-seed uncertainty.
- Table 2: selection × voting × fusion ablation.
- Table 3: results across datasets and encoder families after the large-scale extension.
- Table 4: OOD and insertion analyses, clearly marked secondary.
- Figure 2: retrieved evidence successes and failures selected by a fixed sampling rule.

## Drafting gate

Do not write an abstract result sentence, headline comparison, or conclusion until:

- a corrected paper-mode manifest is complete;
- the primary matched-ablation gate passes;
- Tip-Adapter results are present at equal memory count;
- per-example predictions support paired uncertainty;
- the claim ledger is updated with exact artifact paths; and
- at least one author manually checks a sample of raw predictions and figures.

If the primary gate fails, rewrite the paper as a careful analysis of reliability failure
modes or stop. Do not search for a favourable metric or dataset after the fact.
