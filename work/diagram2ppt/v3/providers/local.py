"""Local model provider for v3.

Wraps locally downloaded models (Grounding DINO, SAM3, Pix2Tex/Nougat, DePlot)
so agents can use them through the same Provider interface as remote APIs.
"""
from __future__ import annotations

from typing import Any

from PIL import Image

from work.diagram2ppt.v3 import models as local_models
from .base import Provider


class LocalModelProvider(Provider):
    """Provider backed by local downloaded models."""

    name = "local_model"

    def __init__(self, model_name: str = "grounding-dino-tiny",
                 cache_dir: str | None = None, **kwargs: Any) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._loaded = False
        self.extra = kwargs

    def _ensure_loaded(self) -> Any:
        if not self._loaded:
            self._model = local_models.get_local_model(
                self.model_name, cache_dir=self.cache_dir)
            self._loaded = True
        return self._model

    def ask(self, image: Image.Image, prompt: str,
            temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Generic ask: dispatch by model type."""
        model = self._ensure_loaded()
        if self.model_name in ("grounding-dino-tiny",):
            # prompt is the object-detection text query
            results = model(image, prompt)
            return str(results)
        if self.model_name in ("pix2tex", "nougat-base"):
            return model(image)
        raise NotImplementedError(
            f"ask() not implemented for local model {self.model_name}")

    def ask_json(self, image: Image.Image, prompt: str,
                 temperature: float = 0.0, max_tokens: int = 4096) -> dict[str, Any]:
        import json
        text = self.ask(image, prompt, temperature, max_tokens)
        try:
            return json.loads(text)
        except Exception:
            return {"raw": text}

    def detect(self, image: Image.Image, prompt: str,
               threshold: float = 0.3, nms_threshold: float = 0.5) -> list[dict]:
        """Object-detection entry point for local vision models."""
        model = self._ensure_loaded()
        return model(image, prompt, threshold=threshold, nms_threshold=nms_threshold)

    def ocr(self, image: Image.Image,
            prompt: str = "Convert the document to markdown.") -> str:
        """OCR entry point for local OCR/formula models."""
        model = self._ensure_loaded()
        return model(image)
