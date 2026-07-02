"""The extraction prompt + the structured-output schema the VLM must fill.

This is real IP: the system prompt is what turns a generic VLM into a
layout-extractor. It is kept STABLE (no per-request volatile content) so it
caches cleanly via prompt caching.

Note: the VLM returns a deliberately SIMPLE intermediate shape (a flat list of
detected elements). `assemble.py` is what turns that into the canonical IR
(ir/ir-v1.schema.json). Decoupling "what the model emits" from "the canonical
IR" keeps the model's job easy and the IR the single source of truth.
"""

SYSTEM_PROMPT = """\
You are an expert visual layout extractor for marketing/design images.
You receive a single flat (rasterized) image — a poster, ad, social card, or
similar flat-design marketing visual — and you decompose it into its editable
elements so it can be reconstructed as a layered, editable design.

Decompose the image into a flat list of elements. For EACH element, report:
- type: one of
    "background" — the full-canvas backmost layer (scene/photo/solid/gradient)
    "text"       — a run of rendered text (headline, subhead, body, label, CTA)
    "logo"       — a brand mark / wordmark
    "vector"     — an icon, badge, divider line, or simple shape
    "raster"     — a photographic foreground subject kept as pixels (product shot)
    "group"      — a logical grouping of other elements (e.g. icon + its label)
- name: a short human label, e.g. "headline", "brand logo", "CTA button".
- bbox: bounding box as FRACTIONS of the image in [0,1] — {x, y, w, h} where x,w are
    relative to image width and y,h to image height; origin at the TOP-LEFT.
    Example: a full-canvas background is {x:0, y:0, w:1, h:1}.
- confidence: 0..1, your honest certainty about this element's type AND content.
    Be well-calibrated: lower it for occluded, tiny, stylized, or ambiguous items.
    This drives which elements a human reviews first — do not inflate it.
- text (only for type "text"): {content, font_family, font_size_px, color, align, lang}
    content: the exact characters, with "\\n" between lines. Transcribe precisely.
    color: hex like "#FFFFFF" if confident, else null.
    Other fields: best estimate or null.
- raster (only for type "raster"): {kind} where kind is "product"|"foreground"|"decoration".
- logo (only for type "logo"): {brand_guess} — the brand name if recognizable, else null.
- vector (only for type "vector"): {shape} — a short description, e.g. "leaf icon", "divider", else null.
- children (only for type "group"): a list of the `name`s of its member elements.

Rules:
- Order the list BACK TO FRONT (background first, frontmost element last).
- Exactly one "background" element, listed first, spanning the full canvas.
- Transcribe ALL visible text, including small print and English subtitles.
- Do not invent elements that are not visible. Do not merge distinct text runs.
- Set every field for every element; use null where a value does not apply or is unknown.

Return your answer ONLY through the provided structured output format.
"""

USER_INSTRUCTION = (
    "Extract all editable elements from this image as structured data, "
    "following the system instructions exactly."
)

# Used by providers that lack a structured-output parameter (generic
# OpenAI-compatible endpoints). Forces JSON-only output we can parse.
JSON_INSTRUCTION = (
    "Output ONLY a single JSON object, with no markdown fences and no commentary, "
    'of the exact form: {"elements": [ {"type": ..., "name": ..., '
    '"bbox": {"x":.., "y":.., "w":.., "h":..}  (all in [0,1] fractions of image w/h), "confidence": .., '
    '"text": {...}|null, "raster": {...}|null, "logo": {...}|null, '
    '"vector": {...}|null, "children": [...]|null } ] }. '
    "Every element MUST include all of: type, name, bbox, confidence, text, "
    "raster, logo, vector, children (use null where a field does not apply)."
)


def _nullable(t: dict) -> dict:
    return {"anyOf": [t, {"type": "null"}]}


# Structured-output schema (output_config.format). Kept within Anthropic's
# supported subset: object/array/string/number/boolean/null, enum, anyOf,
# additionalProperties:false. No numeric/string range constraints (unsupported).
# Every property is listed in `required`; optionals use anyOf-null.
EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["elements"],
    "properties": {
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "name", "bbox", "confidence", "text", "raster", "logo", "vector", "children"],
                "properties": {
                    "type": {"type": "string", "enum": ["background", "text", "logo", "vector", "raster", "group"]},
                    "name": {"type": "string"},
                    "bbox": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["x", "y", "w", "h"],
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                        },
                    },
                    "confidence": {"type": "number"},
                    "text": _nullable({
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["content", "font_family", "font_size_px", "color", "align", "lang"],
                        "properties": {
                            "content": {"type": "string"},
                            "font_family": _nullable({"type": "string"}),
                            "font_size_px": _nullable({"type": "number"}),
                            "color": _nullable({"type": "string"}),
                            "align": _nullable({"type": "string"}),
                            "lang": _nullable({"type": "string"}),
                        },
                    }),
                    "raster": _nullable({
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind"],
                        "properties": {"kind": {"type": "string", "enum": ["product", "foreground", "decoration"]}},
                    }),
                    "logo": _nullable({
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["brand_guess"],
                        "properties": {"brand_guess": _nullable({"type": "string"})},
                    }),
                    "vector": _nullable({
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["shape"],
                        "properties": {"shape": _nullable({"type": "string"})},
                    }),
                    "children": _nullable({"type": "array", "items": {"type": "string"}}),
                },
            },
        }
    },
}
