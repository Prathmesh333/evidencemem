"""Extract and cache OpenCLIP embeddings for CIFAR or SVHN splits."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from evidencemem.cache import save_embedding_cache
from evidencemem.data import save_split, stratified_train_validation_indices
from evidencemem.encoder import OpenClipEncoder
from evidencemem.utils import normalize_vector

PROMPT_TEMPLATES = (
    "a photo of a {}",
    "an image of a {}",
    "a close-up photo of a {}",
    "a visual example of a {}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100", "svhn"), default="cifar10")
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def dataset_bundle(name: str, root: Path, transform: Any) -> tuple[Any, Any, np.ndarray, list[str]]:
    from torchvision import datasets

    if name == "cifar10":
        training = datasets.CIFAR10(root, train=True, transform=transform, download=True)
        testing = datasets.CIFAR10(root, train=False, transform=transform, download=True)
        return training, testing, np.asarray(training.targets), list(training.classes)
    if name == "cifar100":
        training = datasets.CIFAR100(root, train=True, transform=transform, download=True)
        testing = datasets.CIFAR100(root, train=False, transform=transform, download=True)
        return training, testing, np.asarray(training.targets), list(training.classes)
    training = datasets.SVHN(root, split="train", transform=transform, download=True)
    testing = datasets.SVHN(root, split="test", transform=transform, download=True)
    return training, testing, np.asarray(training.labels), [str(index) for index in range(10)]


def encode_loader(encoder: OpenClipEncoder, loader: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    embedding_batches = []
    label_batches = []
    for images, labels in loader:
        embedding_batches.append(encoder.encode_images(images))
        label_batches.append(np.asarray(labels, dtype=np.int64))
    return np.concatenate(embedding_batches), np.concatenate(label_batches)


def limit_indices(indices: np.ndarray, maximum: int | None) -> np.ndarray:
    return indices if maximum is None else indices[:maximum]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise SystemExit("batch-size must be positive and workers non-negative")

    import torch
    from torch.utils.data import DataLoader, Subset

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    encoder = OpenClipEncoder(args.model, args.pretrained)
    training, testing, training_labels, class_names = dataset_bundle(
        args.dataset, args.root, encoder.preprocess
    )
    train_indices, validation_indices = stratified_train_validation_indices(
        training_labels,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    dataset_output = args.output / args.dataset
    save_split(
        dataset_output,
        dataset=args.dataset,
        labels=training_labels,
        train_indices=train_indices,
        validation_indices=validation_indices,
        seed=args.seed,
    )

    test_indices = np.arange(len(testing), dtype=np.int64)
    splits = {
        "train": (Subset(training, limit_indices(train_indices, args.max_samples)), train_indices),
        "validation": (
            Subset(training, limit_indices(validation_indices, args.max_samples)),
            validation_indices,
        ),
        "test": (Subset(testing, limit_indices(test_indices, args.max_samples)), test_indices),
    }
    for split_name, (dataset, original_indices) in splits.items():
        selected_indices = limit_indices(original_indices, args.max_samples)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )
        embeddings, labels = encode_loader(encoder, loader)
        sample_ids = np.array(
            [f"{args.dataset}:{split_name}:{index}" for index in selected_indices], dtype=np.str_
        )
        save_embedding_cache(
            dataset_output / f"{split_name}.npz",
            embeddings,
            labels,
            sample_ids,
            dataset=args.dataset,
            split=split_name,
            model_name=args.model,
            pretrained=args.pretrained,
        )
        print(f"saved {split_name}: {embeddings.shape}")

    prompt_embeddings = []
    for class_name in class_names:
        prompts = [template.format(class_name.replace("_", " ")) for template in PROMPT_TEMPLATES]
        prompt_embeddings.append(normalize_vector(encoder.encode_texts(prompts).mean(axis=0)))
    np.savez_compressed(
        dataset_output / "text_prototypes.npz",
        embeddings=np.stack(prompt_embeddings),
        class_names=np.asarray(class_names, dtype=np.str_),
    )


if __name__ == "__main__":
    main()
