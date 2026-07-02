"""Semantic validation for v3 patch accept/rollback decisions.

The pixel-level Verifier is not enough: a patch can make the rendered image
closer to the original while destroying the semantic content (e.g. replacing a
formula with "0").  This module evaluates whether a patch preserves or improves
the *meaning* of the elements it touches, independently of pixel metrics.
"""
from __future__ import annotations

from typing import Any

from work.diagram2ppt.v3 import ir as IR


CONTENT_AGENTS = {"TextAgent", "FormulaAgent", "ChartAgent", "IconAgent"}


def validate_patch(ir_after: dict, patch: dict, ir_before: dict | None = None
                   ) -> dict[str, Any]:
    """Evaluate a patch for semantic correctness and targeted defect repair.

    Returns a dict:
      - ok: True if the patch should be kept from a semantic standpoint.
      - reason: human-readable explanation.
      - degraded: list of element ids whose content degraded.
      - target_defects_remaining: ids of expected fixes still present.
    """
    agent = patch.get("agent", "")
    changed = patch.get("changed", []) or []
    expected = patch.get("expected_fixes", []) or []

    result: dict[str, Any] = {
        "ok": True,
        "reason": "no semantic issues",
        "degraded": [],
        "target_defects_remaining": [],
    }

    # 1. Content degradation for content agents.
    if agent in CONTENT_AGENTS and ir_before is not None:
        for eid in changed:
            old_el = IR.get_element(ir_before, eid)
            new_el = IR.get_element(ir_after, eid)
            if old_el is None or new_el is None:
                continue
            if _is_content_degraded(old_el, new_el):
                result["degraded"].append(eid)

    if result["degraded"]:
        result["ok"] = False
        result["reason"] = (
            f"content degraded for {len(result['degraded'])} element(s): "
            f"{', '.join(result['degraded'][:3])}"
        )
        return result

    # 2. Target-defect attribution for content agents.
    if agent in CONTENT_AGENTS and expected:
        after_ids = {d.get("id") for d in ir_after.get("defects", [])}
        remaining = [did for did in expected if did in after_ids]
        result["target_defects_remaining"] = remaining
        if remaining and len(remaining) == len(expected):
            # The patch did not remove any of the defects it was meant to fix.
            result["ok"] = False
            result["reason"] = (
                f"target defect(s) still present: {', '.join(remaining[:3])}"
            )

    return result


def _is_content_degraded(old_el: dict, new_el: dict) -> bool:
    """Return True if the element's semantic content got worse."""
    el_type = new_el.get("type")
    if el_type in ("text", "formula"):
        return _text_degraded(old_el, new_el)
    if el_type == "chart":
        return _chart_degraded(old_el, new_el)
    if el_type == "icon":
        return _icon_degraded(old_el, new_el)
    return False


def _text_degraded(old_el: dict, new_el: dict) -> bool:
    old_text = _normalize_text(old_el.get("text", ""))
    new_text = _normalize_text(new_el.get("text", ""))

    # Emptying previously meaningful content is a degradation.
    if not new_text and len(old_text) >= 2:
        return True

    old_len = max(1, len(old_text))
    new_len = len(new_text)

    # Catastrophic shortening: losing >70% of characters.
    if new_len < old_len * 0.30 and old_len > 3:
        return True

    # Collapsing text to a meaningless token like "0" or "-".
    if new_len <= 2 and old_len > 3 and _is_meaningless_token(new_text):
        return True

    # For formulas, dropping all math tokens is a degradation.
    if new_el.get("type") == "formula":
        old_math = _has_math_tokens(old_text)
        new_math = _has_math_tokens(new_text)
        if old_math and not new_math and old_len > 3:
            return True

    return False


def _chart_degraded(old_el: dict, new_el: dict) -> bool:
    old_chart = old_el.get("chart") or old_el.get("ext", {}).get("chart") or {}
    new_chart = new_el.get("chart") or new_el.get("ext", {}).get("chart") or {}

    old_kind = old_chart.get("kind") or old_chart.get("type")
    new_kind = new_chart.get("kind") or new_chart.get("type")

    # Changing to none/unknown is degradation.
    if not new_kind or str(new_kind).lower() == "none":
        return bool(old_kind)

    # Losing all series or categories is degradation.
    old_series = old_chart.get("series") or []
    new_series = new_chart.get("series") or []
    if old_series and not new_series:
        return True

    old_cats = old_chart.get("categories") or []
    new_cats = new_chart.get("categories") or []
    if old_cats and not new_cats:
        return True

    return False


def _icon_degraded(old_el: dict, new_el: dict) -> bool:
    old_icon = old_el.get("icon") or old_el.get("ext", {}).get("icon") or {}
    new_icon = new_el.get("icon") or new_el.get("ext", {}).get("icon") or {}

    old_kind = old_icon.get("kind") or old_icon.get("name") or ""
    new_kind = new_icon.get("kind") or new_icon.get("name") or ""

    # Unknown/empty kind when we previously had one is degradation.
    if old_kind and (not new_kind or str(new_kind).lower() in {"unknown", "none", "other"}):
        return True

    return False


def _normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def _is_meaningless_token(text: str) -> bool:
    """A token that is almost never a valid replacement for real text/formula."""
    if not text:
        return True
    return bool(text) and text in {
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "-", "—", "–", ".", ",", "?", "!", "*", "#", "$", "%", "&",
        "(", ")", "[", "]", "{", "}", "\\", "/", "+", "=", "<", ">",
    }


def _has_math_tokens(text: str) -> bool:
    math_tokens = {
        "\\", "frac", "sum", "int", "prod", "sqrt", "alpha", "beta",
        "gamma", "delta", "epsilon", "theta", "lambda", "mu", "nu", "pi",
        "sigma", "tau", "phi", "omega", "Gamma", "Delta", "Theta",
        "Lambda", "Pi", "Sigma", "Phi", "Omega", "nabla", "infty",
        "rightarrow", "leftarrow", "leftrightarrow", "geq", "leq",
        "neq", "approx", "times", "cdot", "pm", "mp", "in", "subset",
        "cup", "cap", "exists", "forall", "ldots", "cdots",
    }
    lowered = text.lower()
    return any(tok in lowered for tok in math_tokens)
