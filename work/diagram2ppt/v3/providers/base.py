"""Base provider protocol for v3.

A Provider encapsulates one backend (SiliconFlow, OpenAI-compatible, local model,
etc.) and exposes capability-specific calls used by agents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image


class Provider(ABC):
    """Abstract provider for VLM/OCR/formula/chart/icon tasks."""

    name: str = "base"

    @abstractmethod
    def ask(self, image: Image.Image, prompt: str,
            temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Send an image + text prompt and return the model's text response."""
        raise NotImplementedError

    @abstractmethod
    def ask_json(self, image: Image.Image, prompt: str,
                 temperature: float = 0.0, max_tokens: int = 4096) -> dict[str, Any]:
        """Send an image + text prompt and parse the response as JSON."""
        raise NotImplementedError

    def ocr(self, image: Image.Image, prompt: str = "Convert the document to markdown.") -> str:
        """Default OCR path falls back to ask(); providers may override."""
        return self.ask(image, prompt, temperature=0.0, max_tokens=4096)
