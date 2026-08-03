# Initial literature map

This is a living novelty audit, not a finished related-work section. Formal
BibTeX entries will be added only after metadata and claims are verified.

## Foundation

- **CLIP — Learning Transferable Visual Models From Natural Language
  Supervision.** Supplies the frozen joint image-text representation and
  prompt-based zero-shot classifier. EvidenceMem builds on, rather than
  contributes, this representation.
  <https://arxiv.org/abs/2103.00020>

## Closest adaptation work

- **Tip-Adapter — Training-Free Adaption of CLIP for Few-Shot
  Classification.** The most important novelty threat: it constructs a
  key-value cache from visual features and combines cache retrieval with CLIP
  knowledge. EvidenceMem must compare against an equivalent cache formulation
  and cannot claim retrieval/text fusion alone.
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/154_ECCV_2022_paper.php>
  and <https://arxiv.org/abs/2207.09519>

- **ProKeR — A Kernel Perspective on Few-Shot Adaptation of Large
  Vision-Language Models.** Reinterprets cache adaptation as a local kernel
  method. It may provide a stronger modern baseline or theoretical lens.
  <https://arxiv.org/abs/2501.11175>

## OOD detection

- **Out-of-Distribution Detection with Deep Nearest Neighbors.** Establishes
  non-parametric neighbor distance as a serious OOD baseline. EvidenceMem must
  compare its combined confidence against raw kNN distance/similarity.
  <https://proceedings.mlr.press/v162/sun22d.html> and
  <https://arxiv.org/abs/2204.06507>

- **Nearest Neighbor Guidance for Out-of-Distribution Detection.** Combines a
  classifier score with neighborhood geometry, making it especially relevant
  to EvidenceMem's combined confidence claim.
  <https://arxiv.org/abs/2309.14888>

## Continual vision-language learning

- **Class Incremental Learning with Pre-trained Vision-Language Models.** Uses
  adapters and parameter retention for continual CLIP learning. EvidenceMem's
  distinction is no encoder/adapter update, but the staged protocol and old/new
  accuracy reporting must be competitive and fair.
  <https://arxiv.org/abs/2310.20348>

- **Continual Learning on CLIP via Incremental Prompt Tuning with Intrinsic
  Textual Anchors.** A recent prompt-tuning direction that uses textual
  prototypes as anchors. It is a useful contrast to external non-parametric
  memory.
  <https://arxiv.org/abs/2505.20680>

## Open search gaps

- Prototype pruning and memory coreset selection in CLIP embedding spaces.
- Retrieval-based interpretability using actual training medoids.
- Open-world and class-incremental CLIP methods with bounded external memory.
- Confidence calibration that combines multimodal margins and neighbor purity.
- Post-2025 cache adapters that already optimize or compress stored examples.

