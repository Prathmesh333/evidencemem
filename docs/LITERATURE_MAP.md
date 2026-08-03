# Literature and novelty map

This is a working audit, not a finished related-work section. Bibliographic metadata must
be checked against the publisher or official repository before it enters a paper.

## Frozen vision-language representations

- **CLIP: Learning Transferable Visual Models From Natural Language Supervision.**
  Supplies the frozen joint image-text space and prompt-based zero-shot classifier.
  EvidenceMem builds on this representation.
  <https://arxiv.org/abs/2103.00020>

- **Learning to Prompt for Vision-Language Models (CoOp).** Learns prompt context rather
  than external memory. It is relevant as an adaptation baseline but changes parameters.
  <https://arxiv.org/abs/2109.01134>

- **CLIP-Adapter: Better Vision-Language Models with Feature Adapters.** Adds a small
  trainable feature adapter. A strong study should compare with it while reporting that
  its training and storage regime differs from a frozen external memory.
  <https://arxiv.org/abs/2110.04544>

## Closest cache and kernel methods

- **Tip-Adapter: Training-Free Adaptation of CLIP for Few-Shot Classification.** This is
  the closest established baseline. It stores visual features and labels, constructs a
  cache affinity, and combines cache logits with CLIP logits. EvidenceMem cannot claim
  novelty for cache retrieval or text fusion.
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/154_ECCV_2022_paper.php>

- **ProKeR: A Kernel Perspective on Few-Shot Adaptation of Large Vision-Language
  Models.** Recasts cache adaptation through local kernels and is a likely stronger
  theoretical or empirical comparator for a large-scale extension.
  <https://arxiv.org/abs/2501.11175>

## Submodular and coreset selection

Facility-location objectives and greedy submodular maximization are established tools.
EvidenceMem’s approximation statement is therefore not presented as a new theorem. The
research question is whether adding prototype reliability to that selection rule helps a
bounded vision-language memory under matched controls.

The final related-work section needs a verified review of:

- facility-location subset selection and data summarization;
- supervised coresets and prototype selection;
- cache compression or pruning in frozen vision-language spaces; and
- recent methods that learn which cache examples to retain.

## OOD detection

- **Out-of-Distribution Detection with Deep Nearest Neighbors.** Establishes nearest
  neighbour distance as a serious non-parametric baseline.
  <https://proceedings.mlr.press/v162/sun22d.html>

- **Nearest Neighbor Guidance for Out-of-Distribution Detection.** Combines classifier
  confidence with neighbourhood geometry and is relevant to any combined confidence
  analysis.
  <https://arxiv.org/abs/2309.14888>

The historical EvidenceMem confidence formula lost to predictive entropy, so OOD is a
secondary analysis unless the corrected protocol gives a clear and replicated result.

## Continual vision-language learning

- **Class Incremental Learning with Pre-trained Vision-Language Models.** Studies
  parameter-based continual adaptation. EvidenceMem’s distinction is that the encoder is
  unchanged, but that engineering property does not remove the need for fair old/new
  class accuracy and forgetting comparisons.
  <https://arxiv.org/abs/2310.20348>

Class insertion remains a supporting capability. A strong continual-learning claim would
need established protocols, additional methods, repeated task orders, average accuracy,
and forgetting curves.

## Evidence and explanation boundary

Nearest retrieved training examples are inputs to EvidenceMem’s explicit visual voting
rule. They can therefore be called decision evidence. They should not be called causal or
faithful explanations of everything encoded by CLIP without an intervention or human
utility study.

## Open novelty audit

Before submission, search and record work published after this map on:

- learned cache selection and cache compression for vision-language models;
- reliability- or uncertainty-aware prototype selection;
- submodular memories for continual or open-world classification; and
- training-example attribution for frozen multimodal encoders.

If an existing method already combines the same candidate construction, objective, and
voting rule, the contribution must be narrowed or changed.
