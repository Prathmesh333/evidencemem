# Compute budget

## Available machine

- GPU: NVIDIA GeForce GTX 1650, 4 GB VRAM.
- Local Python: 3.13.2; the reproducible experiment environment is pinned to
  Python 3.11 for wider OpenCLIP compatibility.
- Core packages already present: PyTorch, torchvision, FAISS, NumPy,
  scikit-learn, PyYAML, pytest.
- Missing at project start: `open_clip_torch`.

## Budget policy

The encoder remains frozen. Image embeddings are extracted once per dataset and
stored as normalized float32 arrays with a manifest recording model, weights,
transform, split hash, dimension, and sample count. All seeds, memory budgets,
fusion sweeps, and OOD scores reuse those arrays.

Approximate raw embedding storage for a 512-dimensional float32 encoder is
2 KiB per image. CIFAR-10, CIFAR-100, and the SVHN test set together require
well under 0.5 GB for embeddings before metadata and backups.

## Pre-flight pilot

1. Encode 512 CIFAR-10 images at batch sizes 32 and 64.
2. Record images/second, peak VRAM, and output checksum.
3. Select the largest batch size with at least 10% VRAM headroom.
4. Extrapolate full extraction time and add a 40% rerun contingency.
5. Run the full pipeline on 1,000 cached embeddings before downloading or
   processing optional datasets.

## Scope controls

- Mandatory first: CIFAR-10, SVHN OOD, memory budgets, fusion, confidence,
  continual insertion, and ablations.
- CIFAR-100 is the first extension and near-OOD dataset.
- SigLIP, Tiny ImageNet, HNSW, product quantization, and LLaVA reranking remain
  optional until every mandatory result is reproducible.
- No paid API or cloud-GPU spend is authorized by this plan.

