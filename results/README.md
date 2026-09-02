# Experiment artifacts

This directory separates the verified confirmatory release from historical notebooks.

## Verified confirmatory release

[`confirmatory/uie22k-v4-7ce2d2de/`](confirmatory/uie22k-v4-7ce2d2de/) is the result
source for the paper and the root README. It contains the complete executed notebook,
the full prediction archive, all tables, the split manifest, figures, and a release-level
SHA-256 manifest.

Verify the release from the repository root:

```bash
python scripts/verify_confirmatory_release.py
```

The verifier checks these properties:

- every code cell in the published notebook completed without an error output;
- the full run ZIP passes its CRC check and contains no duplicate member names;
- all 23 artifacts listed by the source run manifest match their hashes;
- all 65 prediction packets have the required shapes and finite values;
- predictions equal the probability argmax; and
- the published classification metrics can be recomputed from the raw predictions.

## Historical runs

[`evidencemem.ipynb`](evidencemem.ipynb) is the first executed T4 notebook. It remains
unchanged for transparency. It is not valid evidence for the paper because it used an
activation-mismatched OpenCLIP model definition, reported data-loader child-process
errors, duplicated method code inside the notebook, and omitted a matched Tip-Adapter
comparison. [`legacy_t4_summary.json`](legacy_t4_summary.json) records that audit.

Other untracked local notebooks can be useful during development. A notebook becomes a
paper source only after it has a frozen protocol, raw predictions, a complete artifact
manifest, and an explicit claim decision.
