# Frozen UIE-22K experiment protocol

## Status

The confirmatory run is complete. The frozen protocol is `uie22k_evidencemem_v4`,
revision `2.0.1`. The result source is
`results/confirmatory/uie22k-confirmatory/`.

No parameter can be tuned on the confirmatory split after this run. A new hypothesis
requires a new protocol and untouched data.

## Questions

The protocol asks two questions:

1. Is a 40-image-per-class EvidenceMem non-inferior to full kNN at a predeclared
   one-percentage-point margin?
2. Does reliability-aware scoring improve an equal-count facility-selected memory?

The first question is primary. The second question is secondary. Both questions use the
same frozen encoder, train split, validation split, confirmatory split, class balance,
memory seeds, and image preprocessing.

## Frozen identifiers

- Protocol: `uie22k_evidencemem_v4`
- Protocol revision: `2.0.1`
- Data manifest:
  `f9eece5f3f489fd2b986ca89b797c2843e53e2618cd93361b730f0c77bff2c09`
- Development commit: `0b766243eb6352db067bde2815e4472cca0c6d2`
- Development package tree: `9fec2b475fe962bd92fd8d6e496ec7bcc0b1835e`
- Confirmatory source commit: `7ce2d2de4283a52c4a114352cfb9d211e0f9a426`
- Frozen repository tag: `uie22k-v4-confirmatory-2026-09-02`
- Sample seed: `2026`
- Memory seeds: `7`, `17`, `29`, `43`, and `61`
- Selected encoder: `google/siglip2-base-patch16-384`
- Encoder key: `siglip2_b16_384`
- Default memory budget: 40 images per class, 440 images total

## Data

The source is `rhtsingh/130k-images-512x512-universal-image-embeddings` on Kaggle.
The experiment uses the image files and computes new SigLIP 2 embeddings. It does not use
the supplied third-party embeddings.

The 11 labels are apparel, artwork, cars, dishes, furniture, illustrations, landmark,
meme, packaged, storefronts, and toys. Each label contributes:

- 1,200 training images;
- 200 validation images;
- 300 development images; and
- 300 confirmatory images.

The final manifest contains 22,000 images. The duplicate audit found four exact
duplicates, four within-label perceptual duplicates, and two cross-label perceptual
duplicates in the sampled candidate pool. Those entries were not retained in the fixed
split.

Each source file is 512 by 512 pixels. The selected checkpoint uses its native
384-by-384 preprocessing. The experiment is not a native 512-pixel encoder test.

## Method under test

EvidenceMem uses the following sequence for each memory seed:

1. Encode all images with the frozen SigLIP 2 image encoder.
2. Normalize every embedding.
3. Overcluster each class to twice the requested memory budget.
4. Use the nearest real image to each cluster center as a candidate medoid.
5. Select exactly 40 candidates per class with a coverage-only facility objective.
6. Compute compactness, neighborhood purity, and text alignment for each selected
   prototype.
7. Select the reliability weights on validation data.
8. Retrieve an equal top-k set inside every class.
9. Apply the selected reliability power to class-conditional visual scores.
10. Fuse visual and text probabilities with the selected fixed or continuous text weight.
11. Select the calibration temperature on validation negative log likelihood.
12. Evaluate the frozen configuration on the confirmatory split.

The treatment and facility control use the same prototype indices. Reliability changes
scoring and fusion. It does not change prototype selection in the confirmatory method.

## Baselines

The default comparison includes:

- zero-shot text classification;
- full similarity-weighted kNN;
- a fixed linear probe;
- random memory;
- one nearest-to-centroid example per class;
- K-means medoids;
- equal-count facility selection without reliability;
- the earlier reliability-selection method;
- global, visual-only, fixed-fusion, and continuous-fusion variants; and
- Tip-Adapter with the same cache size as the bounded memories.

Every method uses the same frozen embeddings. Validation data selects method-specific
settings. Confirmatory labels do not select a method or parameter.

## Validation grids

- Full-kNN and global-retrieval depth: `3`, `5`, `10`, or `20`
- Class-conditional depth: `1`, `3`, `5`, or `10` per class
- Fixed text weight or continuous intercept: `0`, `0.25`, `0.5`, `0.75`, or `1`
- Continuous gate slope: `-0.5`, `-0.25`, `0`, `0.25`, or `0.5`
- Reliability power: `0`, `0.5`, `1`, or `2`
- Calibration temperature: `0.02`, `0.03`, `0.05`, `0.07`, `0.10`, `0.15`, `0.20`,
  `0.50`, or `1`
- Memory budget curve: `20`, `40`, or `80` images per class

The code resolves exact validation ties deterministically.

## Metrics

Classification metrics are top-1 accuracy, balanced accuracy, macro F1, negative log
likelihood, Brier score, 15-bin expected calibration error, and area under the selective
risk--coverage curve. Efficiency fields include stored-image count, prototype-build
time, and scoring time per query.

Scoring time excludes image decoding, preprocessing, and encoder inference. The storage
ratio counts source images. It does not compare serialized bytes.

## Hypothesis decisions

The primary decision uses a hierarchical paired bootstrap with 5,000 draws. Each draw
resamples memory seeds and then examples within each selected seed. Non-inferiority is
supported only if the lower end of the 95% interval for EvidenceMem minus full kNN is
greater than `-0.0100`.

The secondary decision uses the same bootstrap structure. Superiority over facility
selection is supported only if the lower interval bound is greater than zero.

The primary result is -0.9818 points with a 95% interval of
[-1.2850, -0.6788] points. The primary hypothesis is not supported. The secondary
result is +0.1212 points with a 95% interval of [-0.1030, +0.3515] points. The secondary
hypothesis is not supported.

## Completion criteria

The confirmatory release is complete because:

- the frozen-protocol checks passed;
- the classification and budget rows are complete;
- all five seeds are present;
- the paired statistics are present and finite;
- the source run manifest has status `complete`;
- the executed notebook contains no error output; and
- the raw predictions reproduce the published metrics.

Run `python scripts/verify_confirmatory_release.py` to repeat the release audit.
