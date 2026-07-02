"""VLM providers for Node ②. Each returns the SAME intermediate shape: a list of
raw element dicts matching EXTRACTION_SCHEMA. The pipeline never depends on which
provider produced them (model-agnostic by design).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Protocol

from .extract_prompt import EXTRACTION_SCHEMA, JSON_INSTRUCTION, SYSTEM_PROMPT, USER_INSTRUCTION

# Skill mandate: default to the most capable model. For high-volume extraction
# you may switch to "claude-sonnet-4-6" / "claude-haiku-4-5" for cost — that is
# your call, not a default we make for you.
MODEL = "claude-opus-4-7"
MODEL_VERSION = "anthropic:claude-opus-4-7"

# Extraction is a structured perception task; structured outputs already
# constrain the result, so we keep thinking off for speed/cost and to avoid
# truncating the JSON under max_tokens. Flip to {"type": "adaptive"} if hard
# layouts (z-order, grouping) need more reasoning.
THINKING = {"type": "disabled"}

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a project .env into os.environ (no dependency).
    Existing env vars win, so an explicit `export` still overrides .env."""
    import os
    for p in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        return


class Provider(Protocol):
    name: str
    model_version: str
    def extract(self, image_path: str) -> list[dict]: ...


class MockProvider:
    """No API key needed. Returns a representative element list for a generic
    flat marketing image so the full pipeline (assemble -> validate -> capture)
    can be exercised offline. Ignores pixel content by design."""

    name = "mock"
    model_version = "mock-0.1"

    def extract(self, image_path: str) -> list[dict]:
        return [
            {"type": "background", "name": "background", "bbox": {"x": 0, "y": 0, "w": 1080, "h": 1350},
             "confidence": 0.6, "text": None, "raster": None, "logo": None, "vector": None, "children": None},
            {"type": "logo", "name": "brand logo", "bbox": {"x": 60, "y": 60, "w": 220, "h": 80},
             "confidence": 0.7, "text": None, "raster": None, "logo": {"brand_guess": None},
             "vector": None, "children": None},
            {"type": "text", "name": "headline", "bbox": {"x": 60, "y": 220, "w": 700, "h": 180},
             "confidence": 0.83,
             "text": {"content": "夏日新品\n限时上市", "font_family": "思源黑体", "font_size_px": 88,
                      "color": "#FFFFFF", "align": "left", "lang": "zh-Hans"},
             "raster": None, "logo": None, "vector": None, "children": None},
            {"type": "text", "name": "subhead", "bbox": {"x": 62, "y": 420, "w": 520, "h": 44},
             "confidence": 0.9,
             "text": {"content": "清爽一夏 · 全场直降", "font_family": "思源黑体", "font_size_px": 32,
                      "color": "#EAEAEA", "align": "left", "lang": "zh-Hans"},
             "raster": None, "logo": None, "vector": None, "children": None},
            {"type": "raster", "name": "product shot", "bbox": {"x": 560, "y": 520, "w": 460, "h": 620},
             "confidence": 0.92, "text": None, "raster": {"kind": "product"}, "logo": None,
             "vector": None, "children": None},
            {"type": "text", "name": "CTA", "bbox": {"x": 60, "y": 1180, "w": 300, "h": 90},
             "confidence": 0.95,
             "text": {"content": "立即抢购", "font_family": "思源黑体", "font_size_px": 40,
                      "color": "#1A1A1A", "align": "center", "lang": "zh-Hans"},
             "raster": None, "logo": None, "vector": None, "children": None},
        ]


class AnthropicProvider:
    """Real extraction via Claude vision + structured outputs.

    NOTE: requires ANTHROPIC_API_KEY. Untested without a key in this session —
    the wire shape follows the claude-api skill docs (base64 vision, cached
    system prompt, output_config.format)."""

    name = "anthropic"
    model_version = MODEL_VERSION

    def __init__(self) -> None:
        import anthropic  # imported lazily so `mock` runs without the dep configured
        self._client = anthropic.Anthropic()

    def extract(self, image_path: str) -> list[dict]:
        data = Path(image_path).read_bytes()
        media_type = _MEDIA_TYPES.get(Path(image_path).suffix.lower(), "image/png")
        b64 = base64.standard_b64encode(data).decode("utf-8")

        resp = self._client.messages.create(
            model=MODEL,
            max_tokens=16000,
            thinking=THINKING,
            system=[{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},  # stable prefix -> cache it
            }],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": USER_INSTRUCTION},
                ],
            }],
            output_config={"format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return _loads_lenient(text)["elements"]


def _loads_lenient(text: str) -> dict:
    """Parse JSON, tolerating accidental markdown fences."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1:
            return json.loads(text[s:e + 1])
        raise


class OpenAICompatProvider:
    """Any OpenAI-compatible vision endpoint — the 'be Switzerland' provider.
    Point it at the cheapest capable model you have. Configured by env:
        I2E_VLM_BASE_URL   e.g. https://openrouter.ai/api/v1
                                https://dashscope-intl.aliyuncs.com/compatible-mode/v1   (Qwen-VL)
                                https://generativelanguage.googleapis.com/v1beta/openai  (Gemini)
                                http://localhost:11434/v1                                (Ollama, free/local)
        I2E_VLM_MODEL      e.g. qwen/qwen2.5-vl-72b-instruct | google/gemini-2.0-flash | gpt-4o-mini
        I2E_VLM_API_KEY    bearer token (omit for most local servers)
    """

    name = "openai-compat"

    def __init__(self) -> None:
        import os
        _load_dotenv()
        self.base_url = os.environ.get("I2E_VLM_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("I2E_VLM_API_KEY", "")
        self.model = os.environ.get("I2E_VLM_MODEL", "")
        if not self.base_url or not self.model:
            raise RuntimeError(
                "openai-compat provider needs env I2E_VLM_BASE_URL and I2E_VLM_MODEL "
                "(and usually I2E_VLM_API_KEY)."
            )
        self.model_version = f"openai-compat:{self.model}"

    MAX_EDGE = 1024  # downscale before sending: VLMs give approximate boxes anyway,
                     # and full-res payloads can exceed server limits / time out.

    def extract(self, image_path: str) -> list[dict]:
        import io as _io
        import httpx  # pulled in transitively; declared in requirements
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        W, H = im.size
        if max(W, H) > self.MAX_EDGE:
            f = self.MAX_EDGE / max(W, H)
            send = im.resize((max(1, int(W * f)), max(1, int(H * f))))
        else:
            send = im
        rw, rh = send.size
        buf = _io.BytesIO(); send.save(buf, "JPEG", quality=90)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": self.model,
            "max_tokens": 8000,
            "temperature": 0,  # determinism; cheap models support this (unlike Opus 4.7)
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + JSON_INSTRUCTION},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # stream — servers disconnect on long non-streaming generations
        payload["stream"] = True
        parts = []
        with httpx.stream("POST", self.base_url + "/chat/completions", json=payload,
                          headers=headers, timeout=240) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    continue
                if delta:
                    parts.append(delta)
        elements = _loads_lenient("".join(parts)).get("elements", [])

        # bbox comes back as fractions of width/height [0,1] (scale-invariant) ->
        # convert to original pixels; clamp so a stray >1 can't break the IR.
        def cl(v):
            try:
                return min(max(float(v), 0.0), 1.0)
            except (TypeError, ValueError):
                return 0.0
        for el in elements:
            b = el.get("bbox")
            if isinstance(b, dict):
                b["x"], b["w"] = cl(b.get("x", 0)) * W, cl(b.get("w", 0)) * W
                b["y"], b["h"] = cl(b.get("y", 0)) * H, cl(b.get("h", 0)) * H
        return elements


def get_provider(name: str) -> Provider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai-compat":
        return OpenAICompatProvider()
    if name == "gemini-web":
        from .gemini_provider import GeminiWebProvider
        return GeminiWebProvider()
    raise ValueError(f"unknown provider: {name!r} (use 'mock', 'openai-compat', 'anthropic', or 'gemini-web')")
