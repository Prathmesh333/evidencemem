# Experiment artifacts

This directory separates the historical run from evidence that may be used in a paper.

## Historical run

[`evidencemem.ipynb`](evidencemem.ipynb) is the complete, executed T4 notebook from the
first implementation. It is preserved byte-for-byte for transparency. Its outputs are
useful for debugging and for showing how the project evolved, but they are not valid as
final paper results because the run used an activation-mismatched OpenCLIP model
definition, produced DataLoader child-process errors, implemented a second copy of the
method inside the notebook, and omitted a matched Tip-Adapter comparison.

The machine-readable audit is in [`legacy_t4_summary.json`](legacy_t4_summary.json).

## Corrected runs

Place the unzipped archive from the current Colab notebook under:

```text
results/corrected/<run-id>/
```

A corrected run is complete only when `run_manifest.json` says `status: complete` and
all listed file hashes verify. Run this check before drafting numerical claims:

```bash
python scripts/check_submission_readiness.py
```

Validation-mode output tests code and protocol only. It must not be cited as final
evidence. A paper-mode run requires three seeds and the full 10,000-example test split.
