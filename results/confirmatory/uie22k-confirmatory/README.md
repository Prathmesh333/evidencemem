# UIE-22K confirmatory release

This directory contains the completed EvidenceMem confirmatory run. The run used one
Tesla T4 and evaluated a frozen SigLIP 2 encoder on 3,300 confirmatory images. It used
five memory seeds and 40 stored prototypes per class.

## Main decision

EvidenceMem reached 94.9273% mean accuracy with 440 stored images. Full kNN reached
95.9091% with 13,200 stored images. The mean difference was -0.9818 accuracy points.
The hierarchical paired 95% interval was [-1.2850, -0.6788] points. The lower bound did
not exceed the predeclared -1.0-point boundary. The primary non-inferiority hypothesis
was not supported.

EvidenceMem exceeded the equal-count facility baseline by 0.1212 points. Its 95%
interval was [-0.1030, 0.3515] points. The secondary superiority hypothesis was not
supported.

## Files

| File | Purpose |
|---|---|
| `EvidenceMem_UIE22K_Confirmatory_T4_executed.ipynb` | Complete Kaggle execution with outputs |
| `full_run_with_predictions.zip` | Original 91-member run archive, including 65 raw prediction packets |
| `release_manifest.json` | SHA-256 and byte count for every published file in this directory |
| `run_manifest.json` | Original run-level manifest for 23 required artifacts |
| `frozen_protocol.json` | Frozen encoder, package tree, split, method, budget, seeds, and hypotheses |
| `uie22k_manifest.csv` | Exact 22,000-image split with paths and content hashes |
| `classification_results.csv` | Per-seed classification and efficiency metrics |
| `classification_summary.csv` | Mean and standard deviation by method |
| `publication_classification_summary.csv` | Paper-facing table with descriptive method names and percentages |
| `confirmatory_hypotheses.csv` | Primary and secondary decisions with paired intervals |
| `memory_budget_summary.csv` | Exploratory 20-, 40-, and 80-image-per-class results |
| `per_class_accuracy.csv` | Derived class-level accuracy for EvidenceMem, full kNN, and facility selection |
| `confusion_summary.csv` | Derived EvidenceMem confusion counts across five seeds |
| `paper_analysis.json` | Independent notebook, archive, prediction, and error-analysis audit |
| `main_accuracy.pdf` | Main method comparison |
| `memory_budget_accuracy.pdf` | Accuracy as a function of memory budget |
| `calibration_ece.pdf` | Calibration comparison before and after temperature selection |
| `qualitative_evidence_siglip2_b16_384.png` | Fixed qualitative retrieval panel from the run |

The remaining CSV and JSON files preserve validation choices, calibration values,
encoder timings, reliability searches, paired tests, and the experiment journal.

## Integrity notes

The executed notebook SHA-256 is
`647c6b61470d7826a5ae54c916c2a2e72832743ee8b937f77808f25ec973a22b`.
It contains 14 executed code cells with sequential execution counts and no error output.

The full run ZIP SHA-256 is
`4b4525b8dba195233afb2d0bc473645b08b37295fcdb8898020e57f7ab023da4`.
The ZIP passes a full CRC check. The independent audit reproduces every published
classification row from the raw predictions with a maximum absolute difference of
`2.22e-16`.

The original run manifest did not list the prediction packets or the qualitative PNG.
The files were still present and valid inside the ZIP. `release_manifest.json` closes
that publication gap by hashing the complete GitHub release. The updated notebook export
cell now hashes every top-level run file before it creates a future archive.

## Verification

From the repository root, run:

```bash
python scripts/verify_confirmatory_release.py
```

Use `--refresh` only when you intentionally rebuild the derived per-class tables and
release manifest from the unchanged source archive.
