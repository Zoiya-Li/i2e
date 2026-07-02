"""SiliconFlow (硅基流动 / 轨迹流动) provider for v3.

Supports:
  - DeepSeek-OCR for text extraction
  - Qwen / DeepSeek vision models for layout/content understanding

Recommended SiliconFlow vision models for diagram tasks:
  - Qwen/Qwen3.6-35B-A3B           (preferred current VL model)
  - Qwen/Qwen3-VL-32B-Instruct     (previous default)
  - Qwen/Qwen3-VL-8B-Instruct      (faster, cheaper)
  - Qwen/Qwen-VL-Max               (production multimodal)
  - deepseek-ai/deepseek-vl2       (alternative vision model)

Environment variables (with I2E_OCR_ or I2E_VLM_ prefix from registry):
  - BASE_URL  default https://api.siliconflow.cn/v1
  - API_KEY   SiliconFlow API key
  - MODEL     e.g. deepseek-ai/DeepSeek-OCR or Qwen/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image

from .openai_compat import OpenAICompatProvider


class SiliconFlowProvider(OpenAICompatProvider):
    """SiliconFlow provider; DeepSeek-OCR uses PNG + a fixed markdown prompt."""

    name = "siliconflow"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Default to a capable vision model.  OCR calls explicitly override this
        # to deepseek-ai/DeepSeek-OCR in self.ocr().
        super().__init__(
            base_url=base_url or "https://api.siliconflow.cn/v1",
            api_key=api_key,
            model=model or "Qwen/Qwen3.6-35B-A3B",
            **kwargs,
        )

    def _encode_image(self, image: Image.Image) -> str:
        # OCR wants PNG fidelity; general VLM calls can use smaller JPEG.
        buf = BytesIO()
        if "OCR" in self.model:
            image.save(buf, format="PNG")
        else:
            image.convert("RGB").save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def ocr(self, image: Image.Image,
            prompt: str = "Convert the document to markdown.") -> str:
        """DeepSeek-OCR: image → markdown text."""
        original_model = self.model
        if "OCR" not in self.model:
            self.model = "deepseek-ai/DeepSeek-OCR"
        try:
            return self.ask(image, prompt, temperature=0.0, max_tokens=4096)
        finally:
            self.model = original_model
