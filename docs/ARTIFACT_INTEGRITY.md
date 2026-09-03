# Artifact integrity policy

## Evidence levels

The repository uses three evidence levels:

1. **Historical:** useful for project history, but invalid for final claims.
2. **Development:** valid for method and encoder selection, but not for final claims.
3. **Confirmatory:** frozen before evaluation and valid for a final claim decision.

The UIE-22K release is confirmatory. The first T4 notebook remains historical. The
repository does not delete or overwrite historical results.

## Source notebook

The frozen confirmatory notebook contains these safeguards:

- an immutable repository tag;
- a fixed package-tree hash;
- a fixed data-manifest identifier;
- a fixed SigLIP 2 encoder and 384-pixel preprocessing path;
- a fixed 40-image-per-class memory budget;
- fixed memory seeds and hypotheses;
- zero data-loader worker processes; and
- no environment override for the confirmatory split or encoder.

Every code cell in the clean source notebook has no output and no execution count. The
published executed notebook has 14 code cells with execution counts 1 through 14 and no
error output.

## Data and embedding provenance

The split manifest records the dataset slug, class label, relative source path, SHA-256,
perceptual hash, dimensions, file size, split, sample identifier, and manifest row. The
manifest fixes 22,000 unique retained images.

An embedding cache records the model, checkpoint, preprocessing fingerprint, source
revision, row count, vector dimension, data type, normalization state, sample order, and
array hashes. The loader rejects a reordered, converted, truncated, or edited array.

## Source run manifest

The original confirmatory `run_manifest.json` records the experiment configuration,
environment, source revision, start and completion time, and hashes for 23 required
files. The release verifier checks each file both inside the original ZIP and in the
published result directory.

The original manifest did not list 65 raw prediction packets, the qualitative PNG, or
the final journal line. Those files were present and valid, but the omission created a
publication-integrity gap. The repository addresses the gap in two ways:

1. `release_manifest.json` hashes every file published in the confirmatory release,
   including the complete source ZIP and executed notebook.
2. The updated notebook export cell writes the completion event before finalization and
   hashes every top-level run file. Future run manifests will therefore include raw
   predictions, qualitative figures, and the final journal.

The original ZIP remains unchanged. Its SHA-256 is
`4b4525b8dba195233afb2d0bc473645b08b37295fcdb8898020e57f7ab023da4`.

## Independent release audit

Run this command from the repository root:

```bash
python scripts/verify_confirmatory_release.py
```

The audit performs these checks:

- verify every release-level file hash and byte count;
- verify the ZIP CRC and member-name uniqueness;
- verify all source-manifest hashes;
- inspect notebook execution counts and error outputs;
- load all 65 prediction packets;
- verify array shapes, finite values, probability sums, and prediction argmax; and
- recompute every classification metric that the stored predictions support.

The maximum difference between the recomputed metrics and published CSV values is
`2.22e-16`.

## External dependencies

The release does not redistribute the Kaggle dataset or pretrained model weights. It
records the public dataset slug and model identifier. Future releases should also record
an upstream checksum when the provider publishes one.

Kaggle and the model host are external systems. Keep the committed release and verify
its hashes after any transfer.
