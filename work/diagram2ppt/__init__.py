"""diagram2ppt — Convert diagram/framework images into native editable PPTX.

Pipeline:
  image → Gemini VLM analysis → structured JSON → python-pptx → .pptx

Every element becomes a native PowerPoint shape (rectangle, text, connector),
not a raster crop — so it's truly editable.
"""
