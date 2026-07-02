"""Step 1: Analyze diagram image via Gemini VLM → structured JSON.

Sends the diagram image to Gemini with a structured extraction prompt.
Returns a dict of elements, each with type, position, text, colors, and
connections.

Key design: the output describes VISUAL STRUCTURE, not pixels.
Each element maps to a native PPT shape type (rectangle, oval, connector…).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from work.gen_decompose.driver import GeminiWebDriver


DIAGRAM_PROMPT = (
    'Analyze this diagram. List ALL elements as JSON: '
    '{"elements":[{"id":"el-1","type":"rounded_rect",'
    '"x":0.1,"y":0.2,"width":0.25,"height":0.12,'
    '"text":"...","fill":"#hex","border_color":"#hex",'
    '"text_color":"#hex","bold":true}]} '
    'Types: rect,rounded_rect,oval,diamond,hexagon,text,arrow,line. '
    'For arrows use from_id/to_id instead of position. '
    'x/y/w/h are 0-1 fractions of image size. '
    'Output ONLY the JSON.'
)


def analyze_diagram(image_path: str,
                    driver: Optional[GeminiWebDriver] = None,
                    timeout: int = 180) -> dict:
    """Analyze a diagram image and return structured element data.

    Args:
        image_path: path to the diagram image
        driver: existing GeminiWebDriver (creates one if None)
        timeout: max wait for Gemini response

    Returns:
        {"elements": [{"id": ..., "type": ..., "x": ..., ...}, ...]}
    """
    own_driver = driver is None
    if own_driver:
        driver = GeminiWebDriver(timeout=timeout)
        driver.connect()

    try:
        raw = driver.analyze(image_path, DIAGRAM_PROMPT, single_turn=True)
    finally:
        if own_driver:
            driver.disconnect()

    data = _parse_response(raw)
    data = _normalize(data)
    return data


def _parse_response(raw: str) -> dict:
    """Parse Gemini's text response into structured dict.

    Handles three formats:
      1. {"elements": [...]}
      2. Bare array [{...}, ...] wrapped into {"elements": [...]}
      3. Gemini's own schema (element_type, shape, etc.) → normalized
    """
    # --- Try 1: direct JSON parse ---
    try:
        data = json.loads(raw)
        if "elements" in data:
            return data
        if isinstance(data, list):
            return {"elements": _normalize_gemini_elements(data)}
    except json.JSONDecodeError:
        pass

    # --- Try 2: find {"entities": ...} or {"elements": ...} in text ---
    # Handles whitespace: {"elements" or {\n  "elements"
    import re
    m = re.search(r'\{\s*"entities"', raw)
    if not m:
        m = re.search(r'\{\s*"elements"', raw)
    if m:
        start = m.start()
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        break

    # --- Try 3: find bare array [...] in text ---
    arr_start = raw.find('[{')
    if arr_start != -1:
        depth = 0
        for i in range(arr_start, len(raw)):
            if raw[i] == '[':
                depth += 1
            elif raw[i] == ']':
                depth -= 1
                if depth == 0:
                    try:
                        arr = json.loads(raw[arr_start:i + 1])
                        return {"elements": _normalize_gemini_elements(arr)}
                    except json.JSONDecodeError:
                        break

    # Debug: save raw response for inspection
    debug_path = Path("work/diagram2ppt/_raw_response.txt")
    debug_path.write_text(raw, encoding="utf-8")
    raise ValueError(
        f"Cannot parse JSON from response (saved to {debug_path}):\n{raw[:500]}..."
    )


def _normalize_gemini_elements(elements: list) -> list:
    """Normalize Gemini's own schema into our canonical format.

    Gemini often uses: element_type, shape, color, position, direction, etc.
    We map to: id, type, x, y, width, height, text, fill, etc.
    """
    # Shape name → our type mapping
    SHAPE_MAP = {
        "rectangle": "rect",
        "rounded rectangle": "rounded_rect",
        "rounded_rect": "rounded_rect",
        "rect": "rect",
        "oval": "oval",
        "circle": "oval",
        "diamond": "diamond",
        "hexagon": "hexagon",
        "parallelogram": "parallelogram",
    }
    # element_type → our type
    TYPE_MAP = {
        "node": None,       # resolved via shape field
        "box": None,
        "connector": "arrow",
        "arrow": "arrow",
        "line": "line",
        "text": "text",
        "label": "text",
        "icon": "icon",
        "badge": "badge",
    }

    normalized = []
    for i, el in enumerate(elements):
        out = {"id": el.get("id", f"el-{i+1}")}

        # Determine element type
        el_type = el.get("type", "")
        element_type = el.get("element_type", "")
        shape = el.get("shape", "")

        if el_type in ("connector", "arrow") or shape == "arrow":
            out["type"] = "arrow"
            out["from_id"] = el.get("from_id", el.get("from", ""))
            out["to_id"] = el.get("to_id", el.get("to", ""))
            out["label"] = el.get("label", "")
            out["color"] = el.get("color", "#333333")
        elif el_type == "text" or el_type == "label":
            out["type"] = "text"
        elif shape:
            out["type"] = SHAPE_MAP.get(shape.lower(), "rect")
        elif el_type:
            mapped = TYPE_MAP.get(el_type)
            out["type"] = mapped if mapped else "rect"
        else:
            out["type"] = "rect"

        # Position
        pos = el.get("position", {})
        out["x"] = el.get("x", pos.get("x", 0))
        out["y"] = el.get("y", pos.get("y", 0))
        out["width"] = el.get("width", pos.get("width", 0.1))
        out["height"] = el.get("height", pos.get("height", 0.05))

        # Text
        out["text"] = el.get("text", el.get("label", el.get("content", "")))

        # Colors
        color = el.get("color", el.get("fill", ""))
        if color and not color.startswith("#"):
            # Gemini sometimes uses color names
            NAME_TO_HEX = {
                "blue": "#4A90D9", "dark blue": "#2C5F8A",
                "light blue": "#87CEEB", "green": "#4CAF50",
                "red": "#E74C3C", "orange": "#FF9800",
                "yellow": "#FFEB3B", "purple": "#9B59B6",
                "gray": "#888888", "grey": "#888888",
                "black": "#000000", "white": "#FFFFFF",
                "dark gray": "#444444", "light gray": "#CCCCCC",
            }
            color = NAME_TO_HEX.get(color.lower(), color)
        out["fill"] = color
        out["border_color"] = el.get("border_color", el.get("border", ""))
        out["text_color"] = el.get("text_color", "")
        out["bold"] = el.get("bold", False)
        out["rounded"] = el.get("rounded", False)

        normalized.append(out)

    return normalized


def _normalize(data: dict) -> dict:
    """Ensure IDs exist, assign sequential IDs if missing, validate connectors."""
    elements = data.get("elements", [])
    shape_ids = set()
    next_id = 1

    # Assign IDs to shapes first
    for el in elements:
        if el.get("type") in ("arrow", "line"):
            continue
        if not el.get("id"):
            el["id"] = f"el-{next_id}"
            next_id += 1
        shape_ids.add(el["id"])

    # Assign IDs to connectors, and fix dangling references
    for el in elements:
        if el.get("type") not in ("arrow", "line"):
            continue
        if not el.get("id"):
            el["id"] = f"conn-{next_id}"
            next_id += 1
        # If connector references non-existent IDs, try to infer
        # (just flag it for now)
        fid = el.get("from_id", "")
        tid = el.get("to_id", "")
        if fid and fid not in shape_ids:
            print(f"  ⚠ Connector {el['id']} from_id '{fid}' not found in shapes")
        if tid and tid not in shape_ids:
            print(f"  ⚠ Connector {el['id']} to_id '{tid}' not found in shapes")

    return data


def save_analysis(data: dict, output_path: str):
    """Save structured analysis to JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"✓ Analysis saved → {p}")
