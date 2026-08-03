# Paper workspace

Target: NeurIPS 2026 VLM4RWD workshop, up to eight content pages excluding
references and appendices, double-blind.

The workshop requests the NeurIPS 2026 conference format. The LaTeX style is
not vendored yet because the official 2026 template must be copied in full and
compiled unchanged before paper content is added. Do not substitute an older
style file or edit the conference `.sty` file.

## Working title

**EvidenceMem: Reliability-Aware Prototype Memory for Updateable and
Uncertainty-Aware Vision-Language Classification**

## Section plan

1. Abstract — no numerical claims until experiments finish.
2. Introduction — problem, why naive cache/kNN is insufficient, contribution.
3. Related work — cache adapters, coresets/prototypes, continual CLIP, OOD.
4. Method — medoid construction, reliability, fusion, confidence, updates.
5. Experiments — classification, budget, OOD, continual, evidence, ablations.
6. Limitations — representation dependence, privacy, non-causal evidence,
   threshold shift, small-dataset scale.
7. Conclusion.
8. Broader impact and NeurIPS checklist.

`docs/experiment_log.md` will be created once the first real experiment batch
has results. Raw or invented numbers must never be placed in the draft.

