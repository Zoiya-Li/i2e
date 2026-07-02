"""OpenAI-compatible VLM client for the iterative loop.

The iterative loop makes many small calls (global pass + per-element refine +
missing-region identify), which browser automation cannot sustain — this is
the API replacement for work/gen_decompose/driver.py.

Config comes from .env / env vars (same contract as extractor.providers):
    I2E_VLM_BASE_URL   e.g. https://api.siliconflow.cn/v1
    I2E_VLM_MODEL      e.g. Qwen/Qwen3.5-397B-A17B
    I2E_VLM_API_KEY    bearer token
"""
from __future__ import annotations

import base64
import io
import json
import os
import signal
import threading
import time

from PIL import Image

from extractor.providers import _load_dotenv


SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_REASONING_MODEL = "Qwen/Qwen3.5-397B-A17B"


class VLMClient:
    """Thin streaming chat-with-image client. One instance per run."""

    MAX_EDGE = 1280       # global pass downscale
    CROP_MAX_EDGE = 1024  # refine/identify crops are already small

    def __init__(self, max_tokens: int = 14000, temperature: float = 0.0,
                 timeout: int = 90, retries: int = 0,
                 model: str | None = None) -> None:
        _load_dotenv()
        _load_recorrect_env()
        self.base_url = (
            os.environ.get("I2E_VLM_BASE_URL", SILICONFLOW_BASE_URL)
        ).rstrip("/")
        self.api_key = (
            os.environ.get("I2E_VLM_API_KEY")
            or os.environ.get("SILICONFLOW_API_KEY")
            or ""
        )
        self.model = model or os.environ.get(
            "I2E_VLM_MODEL", DEFAULT_SILICONFLOW_MODEL)
        if not self.base_url or not self.model:
            raise RuntimeError(
                "VLMClient needs I2E_VLM_BASE_URL and I2E_VLM_MODEL "
                "(and usually I2E_VLM_API_KEY) in env or .env"
            )
        self._validate_provider_config()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = int(os.environ.get("I2E_VLM_TIMEOUT", timeout))
        self.total_timeout = int(os.environ.get("I2E_VLM_TOTAL_TIMEOUT", "120") or 0)
        self.retries = int(os.environ.get("I2E_VLM_RETRIES", retries))
        self.calls = 0          # observability: the loop reports call count

    def _validate_provider_config(self) -> None:
        """Enforce the project-wide provider choice."""
        base = self.base_url.lower()
        if "openrouter.ai" in base:
            raise RuntimeError(
                "OpenRouter is disabled for diagram2ppt. Set "
                "I2E_VLM_BASE_URL=https://api.siliconflow.cn/v1 and use a "
                "SiliconFlow API key/model."
            )
        if "siliconflow.cn" in base and self.api_key.startswith("sk-or-"):
            raise RuntimeError(
                "I2E_VLM_API_KEY looks like an OpenRouter key (sk-or-*). "
                "Replace it with a SiliconFlow API key before running v3."
            )

    # -- public ---------------------------------------------------------

    def chat(self, prompt: str, image: "Image.Image | str",
             max_edge: int | None = None, max_tokens: int | None = None,
             temperature: float | None = None,
             frequency_penalty: float | None = None) -> str:
        """Send one prompt + one image, return the model's text.

        max_tokens / temperature / frequency_penalty override the instance
        defaults for this one call (decompose wants a small token cap +
        frequency_penalty to stop degenerate repetition at temperature 0)."""
        b64 = self._encode(image, max_edge or self.MAX_EDGE)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": _env_bool("I2E_VLM_STREAM", default=False),
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        }
        self._apply_provider_options(payload)
        if frequency_penalty is not None and self._supports_frequency_penalty():
            payload["frequency_penalty"] = frequency_penalty
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._stream_with_deadline(payload, headers)
            except Exception as e:  # httpx errors, malformed stream, HTTP 5xx
                last_err = e
                if attempt < self.retries:
                    time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"VLM call failed after {self.retries + 1} attempts: {last_err}")

    # -- internals ------------------------------------------------------

    def _encode(self, image: "Image.Image | str", max_edge: int) -> str:
        im = Image.open(image) if isinstance(image, str) else image
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_edge:
            f = max_edge / max(w, h)
            im = im.resize((max(1, int(w * f)), max(1, int(h * f))))
        w, h = im.size
        if min(w, h) < 64:  # Qwen-VL endpoints 400 on tiny crops (min side ~28)
            f = 64 / min(w, h)
            im = im.resize((max(64, int(w * f)), max(64, int(h * f))))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    def _stream(self, payload: dict, headers: dict) -> str:
        import httpx
        self.calls += 1
        if not payload.get("stream", True):
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(self.base_url + "/chat/completions",
                                json=payload, headers=headers)
                if r.status_code >= 400:
                    raise RuntimeError(
                        f"HTTP {r.status_code} from VLM: {r.text[:500]}"
                    )
                data = r.json()
                msg = data.get("choices", [{}])[0].get("message", {})
                return (msg.get("content") or msg.get("reasoning_content") or "")
        parts: list[str] = []
        reasoning_parts: list[str] = []
        with httpx.stream("POST", self.base_url + "/chat/completions",
                          json=payload, headers=headers, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta_obj = json.loads(data)["choices"][0].get("delta", {})
                except Exception:
                    continue
                delta = delta_obj.get("content")
                if delta:
                    parts.append(delta)
                reasoning = delta_obj.get("reasoning_content")
                if reasoning:
                    reasoning_parts.append(reasoning)
        return "".join(parts) or "".join(reasoning_parts)

    def _stream_with_deadline(self, payload: dict, headers: dict) -> str:
        if self.total_timeout <= 0 or threading.current_thread() is not threading.main_thread():
            return self._stream(payload, headers)
        old_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _raise_vlm_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.total_timeout)
            return self._stream(payload, headers)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def _supports_frequency_penalty(self) -> bool:
        if os.environ.get("I2E_VLM_ALLOW_FREQUENCY_PENALTY") == "1":
            return True
        return "siliconflow.cn" not in self.base_url.lower()

    def _apply_provider_options(self, payload: dict) -> None:
        """Set vendor options required for stable structured vision output.

        SiliconFlow's Qwen thinking models return useful visual analysis in
        reasoning_content and may exhaust max_tokens before emitting content
        unless thinking is explicitly disabled.  For this pipeline the model is
        a structured perception worker, so we default to no-thinking responses.
        """
        if "siliconflow.cn" not in self.base_url.lower():
            return
        if _is_text_only_model(str(payload.get("model", ""))):
            payload["model"] = os.environ.get(
                "I2E_VISION_MODEL", DEFAULT_SILICONFLOW_MODEL)
        if os.environ.get("I2E_SILICONFLOW_SEND_THINKING_OPTIONS", "0") == "1":
            payload.setdefault("enable_thinking", False)
            payload.setdefault("thinking", {"type": "disabled"})


def _raise_vlm_timeout(signum, frame) -> None:
    raise TimeoutError("VLM call exceeded I2E_VLM_TOTAL_TIMEOUT")


def _load_recorrect_env() -> None:
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        explicit = os.environ.get("I2E_PROVIDER_ENV_FILE")
        if explicit:
            load_dotenv(Path(explicit).expanduser(), override=False)
        recorrect = Path("/Users/lizeyan/Desktop/recorrect/.env")
        if recorrect.exists():
            load_dotenv(recorrect, override=False)
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_text_only_model(model: str) -> bool:
    """Known strong reasoning models that do not accept image_url content."""
    return model in {
        DEFAULT_REASONING_MODEL,
        "Qwen/Qwen3.5-397B-A17B",
    }
