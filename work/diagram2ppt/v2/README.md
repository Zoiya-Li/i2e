# diagram2ppt v2 — Production Baseline

**Status:** ✅ **Stable / maintenance only**. This is the version that produced the usable `v22_out/diagram_final.pptx`.

**Philosophy:** "Extract, then refine or demote." Get the diagram structure right, make as much as possible native-editable, and fall back to faithful pixel crops for elements that cannot converge. **Fidelity is guaranteed; native fraction is optimized.**

---

## What it does

1. `decompose.py` — VLM extracts panels, shapes, text, charts, formulas, arrows, surfaces
2. `handlers.py` — per-type experts fill content (OCR for text, LaTeX-OCR for formulas, chart parser, etc.)
3. `loop.py` / `postprocess.py` — render-diff loop refines or demotes stubborn elements
4. `build_pptx.py` / `svg_export.py` — emit native PPTX or SVG

---

## How to run

```bash
# New default pipeline → SVG (all-native vector)
python -m work.diagram2ppt.v2.run framework.png -o work/diagram2ppt/v2_out

# Legacy loop → PPTX (hybrid native + faithful crops)
python -m work.diagram2ppt.v2.run framework.png --legacy -o work/diagram2ppt/v2_out

# With remote OCR on the A800 box
python -m work.diagram2ppt.v2.run framework.png --legacy --ocr remote -o work/diagram2ppt/v2_out
```

---

## Best delivered artifact

- `work/diagram2ppt/v22_out/diagram_final.pptx`
- 81 shapes + 402 dots + 12 icons + 2 charts + 10 formulas + 6 connectors
- Native fraction: **0.70 / 0.696** (count / area)
- Coverage: **0.972**

---

## Relationship to v3

- **v2 is the baseline.** If you need a result today, use v2.
- **v3 is the research successor.** It reuses v2 for low-level extraction, tracing, diffing, and PPTX assembly, but adds agentic orchestration and a stricter all-native policy.
- **Bug fixes in v2** are fine, but new major features should probably go into v3 unless they are needed for an immediate demo/product.
