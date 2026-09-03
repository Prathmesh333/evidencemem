# EvidenceMem

EvidenceMem is a bounded visual memory for frozen vision--language encoders. It stores
real training images as class prototypes. At inference time, it returns a class
prediction and the retrieved prototypes that contributed to that prediction.

This repository contains the method, unit tests, Kaggle notebooks, a completed
confirmatory run, and a full research-paper draft. The result is useful but mixed:
EvidenceMem preserves most of full-kNN accuracy with 30 times fewer stored images, but
the predeclared non-inferiority and reliability-improvement hypotheses did not pass.

## Confirmatory result

The frozen experiment uses 22,000 images from 11 classes. Each source image is 512 by
512 pixels. The selected SigLIP 2 encoder processes each image at 384 by 384 pixels.
The split contains 13,200 training, 2,200 validation, 3,300 development, and 3,300
confirmatory images.

| Method | Stored images | Confirmatory accuracy | Macro F1 | ECE |
|---|---:|---:|---:|---:|
| Linear probe | Not image-based | 96.67% | 96.66% | 1.18% |
| Full kNN | 13,200 | 95.91% | 95.88% | 1.20% |
| EvidenceMem reliability-weighted fusion | 440 | 94.93% ± 0.22% | 94.88% ± 0.21% | 1.81% |
| Facility selection without reliability | 440 | 94.81% ± 0.10% | 94.76% ± 0.10% | 1.87% |
| K-means medoids | 440 | 94.53% ± 0.19% | 94.46% ± 0.19% | 2.54% |
| Random memory | 440 | 92.77% ± 0.33% | 92.67% ± 0.33% | 3.37% |
| Tip-Adapter with a matched cache | 440 | 91.58% ± 0.11% | 91.38% ± 0.13% | 4.10% |
| Zero-shot text classifier | 0 | 87.97% | 87.69% | 32.49% |

The values after `±` are standard deviations across five memory seeds. The linear probe,
full kNN, and zero-shot classifier are deterministic for the frozen embeddings.

The primary comparison is EvidenceMem minus full kNN. The mean difference is
`-0.9818` accuracy points. Its hierarchical paired 95% interval is
`[-1.2850, -0.6788]` points. The lower bound crosses the predeclared `-1.0`-point
boundary. Therefore, this experiment does not establish non-inferiority.

The secondary comparison is EvidenceMem minus the equal-count facility baseline. The
mean difference is `+0.1212` points. Its 95% interval is `[-0.1030, +0.3515]` points.
Therefore, this experiment does not establish a reliability-related accuracy gain.

## How the method works

The current method has five stages:

1. The frozen SigLIP 2 encoder maps each image to a normalized vector.
2. Each class is overclustered to twice the requested memory budget.
3. The nearest real image to each cluster center becomes a candidate prototype.
4. A facility-coverage objective selects exactly 40 prototypes per class.
5. Class-conditional retrieval combines visual scores, prototype reliability, and class
   text scores. Validation data selects the scoring and fusion settings.

Prototype reliability combines three signals:

- **Compactness:** similarity between a prototype and the members of its cluster.
- **Neighborhood purity:** fraction of nearby training vectors with the same label.
- **Text alignment:** similarity between the prototype and its prompted class text.

EvidenceMem uses the same selected prototype indices as the coverage-only facility
baseline. Reliability changes prototype scoring and visual--text fusion. It does not
change the support set in the confirmatory run. This distinction prevents an incorrect
claim that the current result validates reliability-aware prototype selection.

## How the idea developed

The project started as a vector-database classifier. A query image retrieved labeled
neighbors, and their labels determined the prediction. That design was simple but stored
every support example and provided no fixed memory budget.

The first compressed design replaced the full database with real-image medoids. Real images made
the memory inspectable, while a fixed number of medoids controlled storage. An early
method also used reliability during prototype selection. That design reached lower
accuracy and did not produce a stable gain.

The confirmatory design separates coverage from reliability. Facility selection fixes the real-image
support set. Reliability then changes class-conditional retrieval and fusion. The
confirmatory run tests whether this scoring mechanism improves the matched support set
and whether 440 prototypes remain within one point of full kNN. Both hypotheses were
decided with rules that were frozen before the confirmatory labels were evaluated.

## Memory-budget behavior

The 40-image-per-class setting is the frozen confirmatory setting. The 20- and
80-image settings are exploratory.

| Images per class | Total stored | Compression vs. full kNN | Accuracy |
|---:|---:|---:|---:|
| 20 | 220 | 60× | 94.26% ± 0.14% |
| 40 | 440 | 30× | 94.93% ± 0.22% |
| 80 | 880 | 15× | 95.44% ± 0.17% |

At 80 images per class, the facility baseline reaches 95.53%. The current reliability
mechanism does not improve every memory budget.

## Earlier small-benchmark experiments

The first completed T4 experiment tested an earlier memory design on CIFAR-10 and
CIFAR-100. These runs are useful for understanding how the idea developed, but they do
not use the final confirmatory method or encoder configuration.

| Dataset | Stored prototypes | Fused memory | Zero-shot CLIP | Linear probe |
|---|---:|---:|---:|---:|
| CIFAR-10 | 200 | 89.58% ± 0.11% | 85.25% | 93.84% |
| CIFAR-100 | 2,000 | 66.73% ± 0.20% | 63.18% | 74.85% |

The same notebook evaluated OOD rejection with CIFAR-10 as the in-distribution set.
Predictive entropy reached AUROC 0.907 on CIFAR-100 and 0.991 on SVHN. The historical
notebook used a noncanonical OpenCLIP activation configuration, so these values are
reported as exploratory observations rather than evidence for the frozen hypotheses.
The clean extract is
[`results/historical_small_benchmark_summary.csv`](results/historical_small_benchmark_summary.csv).

## Completed notebook and full results

The main release is in
[`results/confirmatory/uie22k-confirmatory/`](results/confirmatory/uie22k-confirmatory/).
It includes:

- the complete executed Kaggle notebook;
- the full run archive with all 65 raw prediction packets;
- the frozen protocol and exact 22,000-image split manifest;
- classification, calibration, memory-budget, and hypothesis tables;
- per-class accuracy and confusion summaries;
- PDF figures and a qualitative-evidence image; and
- a SHA-256 manifest for every published release file.

The executed notebook is
[`EvidenceMem_UIE22K_Confirmatory_T4_executed.ipynb`](results/confirmatory/uie22k-confirmatory/EvidenceMem_UIE22K_Confirmatory_T4_executed.ipynb).
The unexecuted source notebook is
[`notebooks/EvidenceMem_UIE22K_Confirmatory_T4.ipynb`](notebooks/EvidenceMem_UIE22K_Confirmatory_T4.ipynb).

The release audit checks the notebook, ZIP archive, source run manifest, and raw
predictions. It independently recalculates accuracy, balanced accuracy, macro F1,
negative log likelihood, Brier score, 15-bin ECE, and AURC. The largest observed
difference from the published CSV values is `2.22e-16`.

Run the audit from the repository root:

```bash
python scripts/verify_confirmatory_release.py
```

## Run the confirmatory notebook on Kaggle

Use the committed source notebook only if you need to reproduce the frozen run. Do not
change its encoder, split, memory budget, seeds, or hypotheses.

1. Upload `notebooks/EvidenceMem_UIE22K_Confirmatory_T4.ipynb` to Kaggle.
2. Select a GPU T4 accelerator and enable Internet access.
3. Add the dataset `rhtsingh/130k-images-512x512-universal-image-embeddings`.
4. Optionally add the successful development notebook as an input. The notebook can
   reuse verified training and validation caches.
5. Select **Run All** once.
6. Download the generated run ZIP after the claim gate reports a complete run.

The notebook uses one GPU even if Kaggle supplies two T4 devices. It sets the data-loader
worker count to zero and uses a batch size of 16 for SigLIP 2. These settings match the
completed run and avoid the worker and memory failures found in older notebooks.

## Local installation and tests

The core package does not require a GPU.

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests scripts
```

Install the optional vision and analysis dependencies when you need the full experiment
stack:

```bash
python -m pip install -e ".[vision,analysis,dev]"
```

Python 3.11 or later is required. The completed Kaggle run used Python 3.12.13,
PyTorch 2.10.0, torchvision 0.25.0, scikit-learn 1.6.1, FAISS 1.15.0, and one Tesla T4.

## Repository structure

```text
src/evidencemem/       Reusable prototype-memory, scoring, cache, and artifact code
tests/                 Unit and notebook-structure tests
notebooks/             Clean development and frozen confirmatory notebooks
scripts/               Notebook builders, validators, and release audit
results/               Historical runs and the verified confirmatory release
paper/                 Venue-neutral LaTeX paper, references, and figures
docs/                  Protocol, claim ledger, integrity policy, and experiment journal
```

## Paper

[`paper/main.tex`](paper/main.tex) is a full venue-neutral draft. The paper treats the
failed hypotheses as results. It does not claim that EvidenceMem is non-inferior to full
kNN or that reliability improves accuracy. The draft also states the main limits: one
selected encoder, one 11-class subset, five stochastic seeds, image-count rather than
byte-level storage, and scoring times that exclude encoder inference.

## What the evidence supports

The completed experiment supports these statements:

- A 440-image prototype memory reaches 94.93% mean accuracy on the frozen split.
- The memory stores 30 times fewer source images than full kNN.
- The accuracy point estimate is 0.982 points below full kNN.
- The tested reliability-aware scoring does not show a clear gain over matched facility
  selection.
- Most of the kNN shortfall occurs in artwork, illustrations, and storefronts.

The experiment does not support these statements:

- EvidenceMem is non-inferior to full kNN at a one-point margin.
- Reliability-aware scoring is more accurate than coverage-only facility selection.
- The retrieved images are causal explanations.
- The result generalizes to other datasets, encoders, or native 512-pixel models.

## Next experiment

Do not tune the current confirmatory split again. Register a new experiment on an
untouched dataset. The next study should isolate class-conditional scoring, reliability
weighting, and text fusion in a small factorial ablation. It should also report stored
bytes, encoder-inclusive latency, a second encoder family, and an evidence-fidelity or
human-audit measure.

## License

The repository code is available under the [MIT License](LICENSE). Dataset files and
pretrained model weights retain their original licenses and are not redistributed here.
