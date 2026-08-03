# Artifact integrity policy

## Why the old run is retained

The first T4 notebook produced useful observations, but it also exposed protocol and
implementation defects. Deleting it would hide that history; silently replacing it would
make the record impossible to audit. It therefore remains unchanged at
`results/evidencemem.ipynb`, accompanied by a SHA-256 digest and a machine-readable issue
list in `results/legacy_t4_summary.json`.

The file is labelled `historical_invalid_for_final_claims`. It may be discussed as the
reason for redesigning the method, but it is not a source for final tables.

## Source notebook rules

The clean notebook:

- contains no retained execution counts or outputs;
- installs and imports the repository package;
- uses the activation-compatible OpenCLIP model name;
- uses one DataLoader process on Colab;
- stores verified float32 embeddings;
- includes the matched selector and cache baselines; and
- writes a complete run manifest only after required outputs exist.

`scripts/validate_artifacts.py` enforces these rules statically.

## Embedding cache manifest

Schema version 2 records:

- dataset and split labels;
- resolved model and pretrained checkpoint;
- preprocessing fingerprint and source revision;
- row count, dimension, dtype, and normalization status; and
- SHA-256 hashes over embeddings, labels, and ordered sample IDs.

The array hash includes dtype and shape. A reordered, converted, truncated, or edited
array is rejected on load. Earlier unverified cache schemas must be regenerated.

## Run manifest

`run_manifest.json` records the full experiment and encoder configuration, configuration
hash, git revision, UTC start and completion times, environment snapshot, and hashes and
sizes for every required artifact.

The lifecycle is:

```text
start -> status: running -> write incremental outputs -> verify required files
      -> hash outputs -> status: complete
```

A crash before finalization leaves the run in `running` state. The result archive also
contains `pip_freeze.txt` so exact versions can be recovered after Colab changes its base
image.

## What remains outside the manifest

Downloaded datasets and pretrained weights are not copied into the repository. Their
public identifiers and resolved model names are recorded, but future large-scale runs
should also record upstream checksums where the provider publishes them.

Google Drive and Colab are external systems. Users should keep the generated zip archive
and verify its extracted files against `run_manifest.json` after transfer.
