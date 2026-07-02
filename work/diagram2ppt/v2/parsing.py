"""Lenient JSON extraction from VLM responses (API port of v1 analyze.py).

VLMs wrap JSON in prose, code fences, or 'JSON\\n' prefixes, and sometimes
invent their own field names. parse_elements() finds the JSON and returns the
raw element list; field normalization to IR happens in ir.from_vlm_elements.
"""
from __future__ import annotations

import json
import re


def parse_elements(raw: str) -> list[dict]:
    """Extract a list of element dicts from a noisy VLM response.

    Accepts: {"elements": [...]}, a bare [...] array, or either embedded in
    surrounding text / markdown fences. Raises ValueError if nothing parses.
    """
    raw = raw.strip()
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # 1. direct parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("elements"), list):
            return data["elements"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 2. {"elements": ...} embedded in text (whitespace tolerant)
    m = re.search(r'\{\s*"elements"', raw)
    if m:
        obj = _balanced(raw, m.start(), "{", "}")
        if obj is not None:
            try:
                return json.loads(obj)["elements"]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # 3. bare array embedded in text
    a = raw.find("[{")
    if a != -1:
        arr = _balanced(raw, a, "[", "]")
        if arr is not None:
            try:
                parsed = json.loads(arr)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

    # 4. truncated stream (max_tokens cut the JSON mid-element): salvage the
    #    complete element objects — the coverage loop recovers whatever's lost
    salvaged = _salvage_elements(raw)
    if salvaged:
        return salvaged

    raise ValueError(f"cannot parse elements JSON from response:\n{raw[:400]}...")


def _salvage_elements(raw: str) -> list[dict]:
    """Collect complete {...} objects inside a truncated elements array."""
    m = re.search(r'"elements"\s*:\s*\[', raw)
    start = m.end() if m else (raw.find("[{") + 1 if raw.find("[{") != -1 else -1)
    if start < 0:
        return []
    out: list[dict] = []
    i = start
    while True:
        j = raw.find("{", i)
        if j == -1:
            break
        obj = _balanced(raw, j, "{", "}")
        if obj is None:   # the truncated tail — stop here
            break
        try:
            out.append(json.loads(obj))
        except json.JSONDecodeError:
            break
        i = j + len(obj)
    return out


def _balanced(s: str, start: int, open_ch: str, close_ch: str) -> str | None:
    """Return the balanced bracket substring starting at `start`, or None."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None
