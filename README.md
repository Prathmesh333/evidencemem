# EvidenceMem

EvidenceMem is a bounded visual memory for frozen vision-language encoders. It stores a
small set of real training images, retrieves those images as decision evidence, combines
visual retrieval with text-label scores, and can add labelled examples or classes without
updating the encoder.

The repository contains the reusable Python package, a Google Colab T4 experiment
notebook, integrity checks, a venue-neutral paper plan, and the complete historical run
that motivated the corrected protocol.

## Current status

The software and experiment protocol have been hardened, but the corrected full GPU run
has not been completed yet. This distinction matters:

- The package has 16 passing tests covering selection, scoring, serialization, cache
  tampering, model-name resolution, and run manifests.
- The clean notebook uses the same package code as the tests. It no longer contains a
  second implementation of EvidenceMem.
- The old executed T4 notebook is preserved in [`results/`](results/) for transparency.
  Its numbers are historical observations, not final paper evidence.
- [`scripts/check_submission_readiness.py`](scripts/check_submission_readiness.py) blocks
  paper claims until a corrected three-seed run and its hash manifest are present.

No repository can guarantee paper acceptance. The practical goal here is narrower: make
the central claim falsifiable, compare it with the closest baselines, and leave enough raw
evidence for another researcher or reviewer to audit the result.

## Start here

| Resource | Purpose |
|---|---|
| [`notebooks/EvidenceMem_Colab_T4.ipynb`](notebooks/EvidenceMem_Colab_T4.ipynb) | Clean end-to-end notebook for a Colab T4 |
| [`results/`](results/) | Historical run, audit record, and location for corrected run archives |
| [`src/evidencemem/`](src/evidencemem/) | Reusable implementation used by the notebook |
| [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) | Frozen evaluation and reporting rules |
| [`docs/CLAIM_LEDGER.md`](docs/CLAIM_LEDGER.md) | Claims, required evidence, and present status |
| [`docs/ARTIFACT_INTEGRITY.md`](docs/ARTIFACT_INTEGRITY.md) | What is hashed, preserved, or excluded from claims |
| [`paper/README.md`](paper/README.md) | Paper workspace and drafting gate |

[Open the clean notebook in Google Colab](https://colab.research.google.com/github/Prathmesh333/evidencemem/blob/main/notebooks/EvidenceMem_Colab_T4.ipynb).
If the repository is private, Colab may request GitHub access. Downloading the notebook
and uploading it to Colab also works.

## How the idea developed

The starting idea was a “vector database for classification”:

```text
image -> encoder -> nearest stored vector -> label
```

That is useful engineering, but it is not enough for a research contribution. It stores
too many examples, a single nearest neighbour is brittle, and the method says little
about memory selection, unfamiliar inputs, or the role of the retrieved examples.

The next version compressed each class with K-means medoids. A medoid is an actual
training image nearest to a cluster centre, so it can be displayed. The first T4 run then
added reliability-weighted voting, CLIP text fusion, OOD scores, and class insertion.
That run exposed a more important question: reliability did not help when it was applied
only after plain medoids had already been selected.

The current method moves reliability into the memory-construction objective. It first
creates more candidate medoids than the final budget, then selects exactly the requested
number by balancing coverage of the class distribution against candidate reliability.
This creates a direct, controlled claim:

> At an equal stored-image budget and with the same validation protocol, does
> reliability-aware facility selection outperform coverage-only selection?

Everything else—fusion, evidence retrieval, insertion, and OOD analysis—is supporting
evaluation rather than a bundle of unrelated novelty claims.

## Method

### 1. Frozen image and text representations

The corrected notebook uses OpenCLIP `ViT-B-32-quickgelu` with the `openai` checkpoint.
The QuickGELU model definition matches the activation used to train that checkpoint.
Image and text vectors are L2-normalized float32 arrays, so cosine similarity is an inner
product.

Each class-text vector is the normalized average of several prompt embeddings. The image
encoder remains frozen in every EvidenceMem experiment.

### 2. Candidate medoids

For each class, MiniBatchKMeans creates an overcomplete candidate set. Every centre is
replaced by its nearest real training example. Candidates therefore retain source sample
identities and can be rendered as evidence.

For candidate (j), the reliability score combines:

- cluster compactness (q_j),
- local label purity (p_j), and
- alignment (a_j) with the class-text vector.

The default score is:

\[
r_j = 0.45q_j + 0.35p_j + 0.20a_j.
\]

Component inputs are mapped into bounded similarity ranges before the weighted average.

### 3. Exact-budget selection

For class examples (X_c), candidate medoids (V_c), and budget (B), the selector
greedily maximizes:

\[
F(S) = \lambda_{cov}\frac{1}{|X_c|}
       \sum_{x \in X_c}\max_{v \in S}\frac{1+x^\top v}{2}
       + \lambda_{rel}\frac{1}{B}\sum_{v \in S}r_v,
       \qquad |S|=B.
\]

The first term is facility-location coverage; the second is a modular reliability reward.
Both terms are non-negative and monotone, so the objective is monotone submodular and the
standard greedy cardinality solution has the usual (1-1/e) approximation guarantee.
That is a property of the selection objective, not evidence that the method improves
accuracy. The corrected GPU run must establish the empirical part.

The default coefficients are `coverage_weight=0.75`,
`reliability_weight=0.25`, with a `candidate_multiplier=2.0`.

### 4. Retrieval and prediction

For a query (z), the memory retrieves its top-(k) candidate vectors. Within the
retrieved set, similarity weights are multiplied by prototype reliability and accumulated
by class. This produces a normalized visual distribution (P_v(c\mid z)).

The frozen text prototypes produce (P_t(c\mid z)). The final distribution is:

\[
P(c\mid z) = (1-\alpha)P_v(c\mid z) + \alpha P_t(c\mid z).
\]

`text_weight` is the canonical package name for (alpha). `0.0` is visual-only and
`1.0` is text-only. The old ambiguous `fusion_weight` argument is deprecated.

The prediction object contains the final, visual, and text scores; confidence components;
an unknown flag; and the retrieved source images used by the visual rule.

### 5. Updates and bounded memory

The memory can:

- merge a labelled example into a nearby same-class prototype,
- insert a new prototype when it represents a different local mode,
- add a class from a small labelled support set,
- prune low-reliability, low-use entries while preserving a class floor, and
- save and reload all prototypes and selection settings as a compressed NumPy archive.

No update changes the frozen encoder.

## Experiment design

The notebook has two modes:

- `validation` runs one seed on bounded subsets. It checks the complete pipeline but its
  numbers must not be cited.
- `paper` runs seeds 7, 17, and 29, the full 10,000-image CIFAR-10 test set, the complete
  budget curve, secondary CIFAR-100 classification, OOD evaluation, class insertion,
  evidence precision, ablations, and robustness checks.

Every tunable fusion weight, retrieval depth, temperature, and Tip-Adapter coefficient is
selected on the validation split. Test labels are used once for reporting.

### Baselines

The corrected protocol includes:

- CLIP zero-shot,
- conventional full similarity-weighted kNN,
- one real centroid-nearest example per class,
- random equal-budget memory,
- plain K-means medoids,
- coverage-only facility selection with uniform voting,
- Tip-Adapter at the same stored-example budget,
- a fixed linear probe, and
- a separately labelled supervised ResNet-18 reference.

The primary ablation differs from EvidenceMem only in the reliability term and reliability
voting. All memory methods receive their own validation-selected (k) and text weight so
one method is not favoured by another method’s hyperparameters.

### Required reporting

The full run exports:

- per-seed metrics and summaries,
- per-example predictions, probabilities, and scores,
- selected validation hyperparameters,
- equal-budget accuracy and latency curves,
- paired bootstrap intervals and exact McNemar tests,
- OOD AUROC, AUPR, and FPR95,
- continual insertion metrics,
- evidence precision and qualitative examples,
- environment details and `pip freeze`, and
- a run manifest containing a hash for every required artifact.

The claim gate requires the reliability-aware selector to beat the matched coverage-only
selector on at least half of the tested budgets, plus at least one supporting signal. If
that condition fails, the project must narrow or become an analysis/negative-result paper.

## Historical T4 observations

The complete first run remains at [`results/evidencemem.ipynb`](results/evidencemem.ipynb).
It ran on a Tesla T4 with seeds 7, 17, and 29. These values explain why the protocol was
changed; they are not final results:

| Historical method or condition | Observation |
|---|---:|
| CIFAR-10 EvidenceMem fused accuracy | 89.58% mean |
| CIFAR-10 CLIP zero-shot accuracy | 85.25% |
| CIFAR-10 plain medoid accuracy | 86.09% mean |
| CIFAR-10 linear-probe accuracy | 93.84% |
| CIFAR-100 EvidenceMem fused accuracy | 66.73% mean |
| CIFAR-100 linear-probe accuracy | 74.85% |

That run also found that reliability-weighted visual voting underperformed unweighted
medoid voting at the default budget, and the proposed disagreement-aware OOD score lost
to predictive entropy. Those negative observations are preserved rather than hidden.

The notebook is excluded from final claims because it used the non-QuickGELU OpenCLIP
model definition with OpenAI weights, emitted repeated DataLoader child-process errors,
duplicated the method inside the notebook, lacked the closest cache baseline, and did not
produce a complete hash manifest. See
[`results/legacy_t4_summary.json`](results/legacy_t4_summary.json) for the audit record.

## Run on Google Colab

1. Open [`notebooks/EvidenceMem_Colab_T4.ipynb`](notebooks/EvidenceMem_Colab_T4.ipynb) in
   Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Leave `RUN_MODE = "validation"` and run all cells.
4. Download the generated validation archive and inspect `claim_validation.json` and
   `run_manifest.json`.
5. Set `RUN_MODE = "paper"`, restart the runtime, and run all cells.
6. Unzip the completed archive into `results/corrected/<run-id>/`.
7. Run `python scripts/check_submission_readiness.py` locally.

Set `USE_GOOGLE_DRIVE = True` if caches must survive Colab session resets. The notebook
uses `num_workers=0` deliberately because forked DataLoader workers caused cleanup errors
in the historical Colab runtime.

## Install locally

Core package:

```bash
python -m pip install -e .
```

Vision and analysis dependencies:

```bash
python -m pip install -e ".[vision,analysis]"
```

Development checks:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests scripts
python scripts/validate_artifacts.py
```

The core tests do not download datasets or model weights.

## Minimal package example

```python
import numpy as np

from evidencemem import EvidenceMemClassifier, EvidenceMemory, SelectionConfig

# Rows are frozen, normalized image embeddings.
train_embeddings = np.asarray(..., dtype=np.float32)
train_labels = np.asarray(..., dtype=np.int64)
text_prototypes = {
    0: np.asarray(..., dtype=np.float32),
    1: np.asarray(..., dtype=np.float32),
}

memory = EvidenceMemory(
    selection_config=SelectionConfig(
        candidate_multiplier=2.0,
        coverage_weight=0.75,
        reliability_weight=0.25,
    )
).build(
    train_embeddings,
    train_labels,
    class_names={0: "cat", 1: "dog"},
    prototypes_per_class=20,
    sample_ids=[str(index) for index in range(len(train_labels))],
    text_prototypes=text_prototypes,
    duplicate_threshold=None,
)

classifier = EvidenceMemClassifier(
    memory,
    text_prototypes,
    text_weight=0.25,
)
prediction = classifier.predict_embedding(np.asarray(..., dtype=np.float32), k=10)

print(prediction.class_name, prediction.confidence, prediction.is_unknown)
for item in prediction.evidence:
    print(item.sample_id, item.class_name, item.similarity, item.reliability)
```

## Artifact integrity

Embedding caches have adjacent schema-versioned manifests. Loading verifies sample
identity, shapes, dtype, normalization status, and SHA-256 hashes of the embedding and
label arrays. Old unverified caches are rejected instead of silently reused.

The notebook writes into a versioned protocol directory and only marks a run complete
after every required output exists. `run_manifest.json` records the git revision, resolved
encoder, full configuration, package environment, file sizes, and SHA-256 hashes.

Run the static audit at any time:

```bash
python scripts/validate_artifacts.py
```

It rejects the known failure modes from the first run: a mismatched OpenCLIP definition,
float16 embedding caches, notebook-local K-means fitting, multi-worker Colab loading,
retained outputs in the clean source notebook, or package-version drift.

## Repository layout

```text
.
├── configs/
│   └── cifar10.yaml
├── docs/
│   ├── ARTIFACT_INTEGRITY.md
│   ├── CLAIM_LEDGER.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── LITERATURE_MAP.md
│   └── RESEARCH_PLAN.md
├── notebooks/
│   └── EvidenceMem_Colab_T4.ipynb
├── paper/
│   └── README.md
├── results/
│   ├── corrected/
│   ├── evidencemem.ipynb
│   ├── legacy_t4_summary.json
│   └── README.md
├── scripts/
│   ├── check_submission_readiness.py
│   ├── extract_embeddings.py
│   ├── smoke_core.py
│   └── validate_artifacts.py
├── src/evidencemem/
└── tests/
```

## What EvidenceMem does not claim

- It does not invent CLIP, vector search, K-means, kNN, cache adaptation, or prototype
  explanations.
- Retrieved images are evidence used by the explicit decision rule. They are not a
  causal explanation of the encoder’s internal representation.
- CIFAR experiments do not establish web-scale, medical, safety-critical, or production
  performance.
- OOD detection and class insertion remain evaluated capabilities, not established
  contributions unless they beat their predeclared baselines.
- The objective’s greedy approximation guarantee does not imply better classification.
- Historical output does not become valid merely because it is reproducible.

## Licence

MIT. See [`LICENSE`](LICENSE).
