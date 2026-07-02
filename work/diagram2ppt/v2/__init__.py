"""diagram2ppt v2 — iterative, type-aware diagram → native-editable PPTX.

The v1 pipeline (work/diagram2ppt/) was single-pass: one Gemini-web call →
JSON → python-pptx. v2 is the "做好" architecture from
docs/positioning-20260610.md:

  Phase 0  global VLM analysis (API, not browser automation)
  Phase 1  type routing (this slice: shapes / connectors / text)
  Phase 2  render the IR back to pixels (PIL proxy of the PPT output)
  Phase 3  render-diff iteration: per-element residual + ink-coverage scan →
           re-query worst elements on CROPPED regions → converge, or demote
           to a pixel-faithful raster crop of the ORIGINAL image.

Invariants (the bar):
  * recomposite fidelity is guaranteed — anything that doesn't converge ships
    as a faithful crop, never as a wrong native shape;
  * the north-star metric is native fraction at that fixed fidelity, reported
    per run in report.json and expected to rise version over version.
"""
