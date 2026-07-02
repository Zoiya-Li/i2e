"""Provider registry for v3.

Agents do not call APIs directly. They request a provider by capability:
  - "vlm"        → vision-language model (layout, content understanding)
  - "ocr"        → text recognition
  - "chart"      → chart data extraction
  - "icon"       → icon semantic recognition

Each capability can be configured independently via environment variables:
  I2E_VLM_*        default VLM config (used by layout/vlm agent)
  I2E_OCR_*        OCR config (falls back to I2E_VLM_* if no API key)
  I2E_CHART_*      chart VLM config (falls back to I2E_VLM_*)
  I2E_ICON_*       icon VLM config (falls back to I2E_VLM_*)

The provider type (siliconflow / openai_compat / local_model) is selected by:
  I2E_V3_VLM_PROVIDER  for vlm/chart/icon
  I2E_V3_OCR_PROVIDER  for ocr
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load project/provider .env files so users do not have to export variables
# manually. Existing env vars win, so explicit exports still override files.
try:
    from dotenv import load_dotenv

    here = Path(__file__).resolve()
    for parent in here.parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    explicit_env = os.environ.get("I2E_PROVIDER_ENV_FILE")
    if explicit_env:
        load_dotenv(Path(explicit_env).expanduser(), override=False)
    recorrect_env = Path("/Users/lizeyan/Desktop/recorrect/.env")
    if recorrect_env.exists():
        load_dotenv(recorrect_env, override=False)
except Exception:
    pass

from .base import Provider
from .local import LocalModelProvider
from .openai_compat import OpenAICompatProvider
from .siliconflow import SiliconFlowProvider


_DEFAULT_VLM_PROVIDER = os.environ.get("I2E_V3_VLM_PROVIDER", "siliconflow")
_DEFAULT_OCR_PROVIDER = os.environ.get("I2E_V3_OCR_PROVIDER", "siliconflow")

# Capability → env prefix for its own settings.
_CAPABILITY_PREFIX = {
    "vlm": "I2E_VLM_",
    "ocr": "I2E_OCR_",
    "chart": "I2E_CHART_",
    "icon": "I2E_ICON_",
}

# Capabilities that are served by the VLM provider type.
_VLM_CAPABILITIES = {"vlm", "chart", "icon"}


def _load_env(prefix: str) -> dict[str, Any]:
    """Load provider config from environment variables with a prefix."""
    out: dict[str, Any] = {}
    for key, val in os.environ.items():
        if key.startswith(prefix):
            out[key[len(prefix):].lower()] = val
    return out


def _provider_name_for_capability(capability: str) -> str:
    if capability == "ocr":
        return _DEFAULT_OCR_PROVIDER
    return _DEFAULT_VLM_PROVIDER


def _config_for_capability(capability: str) -> dict[str, Any]:
    """Build config for a capability, falling back to the default VLM config."""
    prefix = _CAPABILITY_PREFIX.get(capability)
    if not prefix:
        raise ValueError(f"unknown capability: {capability}")

    cfg = _load_env(prefix)

    # OCR falls back to VLM credentials if OCR-specific ones are not set.
    if capability == "ocr" and not cfg.get("api_key"):
        cfg = {**_load_env("I2E_VLM_"), **cfg}

    # Chart/icon can override the VLM model while keeping the same credentials.
    if capability in ("chart", "icon") and not cfg.get("api_key"):
        cfg = {**_load_env("I2E_VLM_"), **cfg}

    # Keep compatibility with projects that store SiliconFlow credentials using
    # the vendor-native name, e.g. recorrect's SILICONFLOW_API_KEY.
    if not cfg.get("api_key") and os.environ.get("SILICONFLOW_API_KEY"):
        cfg["api_key"] = os.environ["SILICONFLOW_API_KEY"]

    base_url = str(cfg.get("base_url") or "")
    api_key = str(cfg.get("api_key") or "")
    if "openrouter.ai" in base_url.lower():
        raise RuntimeError(
            "OpenRouter is disabled for diagram2ppt. Use SiliconFlow "
            "(https://api.siliconflow.cn/v1) credentials instead."
        )
    if "siliconflow.cn" in base_url.lower() and api_key.startswith("sk-or-"):
        raise RuntimeError(
            "I2E provider is configured for SiliconFlow but the API key looks "
            "like an OpenRouter key (sk-or-*)."
        )

    return cfg


def get_provider(capability: str, **kwargs: Any) -> Provider:
    """Return a provider for the given capability.

    Args:
        capability: one of vlm/ocr/chart/icon.

    Returns:
        A Provider instance.
    """
    name = _provider_name_for_capability(capability)
    cfg = _config_for_capability(capability)

    merged = {**cfg, **kwargs}
    if name == "siliconflow":
        return SiliconFlowProvider(**merged)
    if name == "openai_compat":
        return OpenAICompatProvider(**merged)
    if name == "local_model":
        return LocalModelProvider(**merged)
    raise ValueError(f"unknown provider: {name}")
