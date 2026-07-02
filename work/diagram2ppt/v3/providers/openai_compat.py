"""OpenAI-compatible provider (OpenRouter, DeepSeek, SiliconFlow, etc.)."""
from __future__ import annotations

import base64
import json
import os
import re
from io import BytesIO
from typing import Any

from PIL import Image

from .base import Provider


class OpenAICompatProvider(Provider):
    """Generic OpenAI-compatible vision API provider."""

    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.base_url = base_url or "https://api.openai.com/v1"
        self.api_key = api_key or ""
        self.model = model or "gpt-4o"
        self.extra = kwargs
        try:
            self.timeout = float(kwargs.get("timeout") or os.environ.get("I2E_VLM_TIMEOUT", "60"))
        except (TypeError, ValueError):
            self.timeout = 60.0

    def _encode_image(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _call(self, image: Image.Image, prompt: str,
              temperature: float, max_tokens: int) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("openai package required for openai_compat provider") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        b64 = self._encode_image(image)
        kwargs: dict[str, Any] = {
            "model": self._model_for_image(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        extra_body = self._provider_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        response = client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        return (
            (msg.content or "")
            or str(getattr(msg, "reasoning_content", "") or "")
        )

    def ask(self, image: Image.Image, prompt: str,
            temperature: float = 0.0, max_tokens: int = 4096) -> str:
        return self._call(image, prompt, temperature, max_tokens)

    def ask_json(self, image: Image.Image, prompt: str,
                 temperature: float = 0.0, max_tokens: int = 4096) -> dict[str, Any]:
        text = self._call(image, prompt + "\n\nOutput ONLY valid JSON.",
                          temperature, max_tokens)
        return self._extract_json(text)

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        fence = re.search(r"```(?:json)?\s*\n?([\s\S]*?)```", text)
        if fence:
            try:
                return json.loads(fence.group(1).strip())
            except json.JSONDecodeError:
                pass
        brace = re.search(r"\{[\s\S]*\}", text)
        if brace:
            try:
                return json.loads(brace.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not extract JSON from response: {text[:200]}")

    def _provider_extra_body(self) -> dict[str, Any]:
        """Vendor-specific options for OpenAI-compatible endpoints."""
        if "siliconflow.cn" not in self.base_url.lower():
            return {}
        # Some SiliconFlow Qwen endpoints reject unsupported thinking options
        # with HTTP 400.  Keep the default payload portable; allow an explicit
        # opt-in for models/endpoints known to support these controls.
        if os.environ.get("I2E_SILICONFLOW_SEND_THINKING_OPTIONS", "0") != "1":
            return {}
        return {"enable_thinking": False, "thinking": {"type": "disabled"}}

    def _model_for_image(self) -> str:
        """Use a real vision model when a text-only reasoning model is configured."""
        if self.model in {"Qwen/Qwen3.5-397B-A17B"}:
            return os.environ.get("I2E_VISION_MODEL", "Qwen/Qwen3.6-35B-A3B")
        return self.model
