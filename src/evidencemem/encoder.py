"""Lazy OpenCLIP wrapper used by the embedding extraction scripts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .utils import normalize_rows


class VisionLanguageEncoder(Protocol):
    """Minimal image/text encoder contract required by EvidenceMem."""

    dimension: int
    preprocess: Any

    def encode_images(self, images: Any) -> NDArray[np.float32]: ...

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]: ...


class OpenClipEncoder:
    """Frozen OpenCLIP encoder with normalized NumPy outputs.

    Heavy dependencies are imported only when this class is instantiated, so
    the memory core and its unit tests remain lightweight.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        *,
        device: str | None = None,
        precision: str = "fp16",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP support requires `python -m pip install -e .[vision]`"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision if self.device == "cuda" else "fp32"

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model_name)

        visual_dimension = getattr(self.model.visual, "output_dim", None)
        if visual_dimension is None:
            raise RuntimeError("could not determine OpenCLIP output dimension")
        self.dimension = int(visual_dimension)

    def _autocast_context(self) -> Any:
        if self.device == "cuda" and self.precision == "fp16":
            return self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
        return self._torch.autocast(device_type=self.device, enabled=False)

    def encode_images(self, images: Any) -> NDArray[np.float32]:
        """Encode a preprocessed image tensor into normalized float32 rows."""
        batch = images.to(self.device, non_blocking=self.device == "cuda")
        with self._torch.inference_mode(), self._autocast_context():
            features = self.model.encode_image(batch)
        return normalize_rows(features.float().cpu().numpy(), name="image embeddings")

    def encode_texts(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Encode prompts into normalized float32 rows."""
        tokens = self.tokenizer(list(texts)).to(self.device)
        with self._torch.inference_mode(), self._autocast_context():
            features = self.model.encode_text(tokens)
        return normalize_rows(features.float().cpu().numpy(), name="text embeddings")
