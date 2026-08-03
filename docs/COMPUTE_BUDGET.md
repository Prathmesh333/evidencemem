# Compute budget

## Primary runtime

The corrected notebook targets a Google Colab NVIDIA T4 with 16 GB VRAM. Colab’s base
environment changes over time, so the notebook records its actual Python, CUDA, PyTorch,
torchvision, OpenCLIP, FAISS, and package versions instead of claiming a permanent image.

The local machine can run tests and synthetic checks. Full image encoding and the
three-seed paper-mode experiment are assigned to the T4.

## Storage estimate

A 512-dimensional float32 embedding uses 2 KiB. CIFAR-10 train/validation/test,
CIFAR-100 test, and SVHN test embeddings fit comfortably below 0.5 GB before archive
overhead. Embeddings remain float32 because the historical float16 cache introduced an
unnecessary numerical mismatch between fresh and resumed runs.

## Runtime controls

- Encode each exact dataset split once and reuse its verified cache.
- Use mixed precision only inside the frozen encoder forward pass.
- Use `num_workers=0` on Colab to avoid worker cleanup failures.
- Use batch FAISS search for purity and retrieval when available.
- Use lazy greedy selection for the submodular objective.
- Save every long-table row incrementally.
- Run bounded validation mode before the full protocol.

## Expansion policy

The current T4 run is a go/no-go pilot for the primary claim. Large-scale datasets,
additional encoders, and paid compute are justified only after the matched CIFAR result
is positive. A strong archival submission will require that extension; the small run is
not presented as sufficient by itself.
