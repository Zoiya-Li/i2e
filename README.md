# i2e — image → editable

Turn flat AI-generated visuals into editable, on-brand design assets.

**Today (Current):** an **image → editable** pipeline — currently focused on **structured diagrams → native PowerPoint / SVG**. The original poster flywheel and the editable-omnimatte superpower are **Frozen** (completed); see [STATUS.md](STATUS.md) for the full picture.

**Long-term (Target):** a **Visual Design Decompiler** — recover any flat visual asset into an editable, auditable, iterable, cross-format **design source structure**, not just one SVG or PPTX:

```
Visual Artifact → Evidence → Components → Editable Design IR → SVG / PPTX / (future) Figma-like JSON / HTML
```

AI participates during decompile, but **post-export editing is native / deterministic / fast / auditable / not model-dependent** — changing text, color, shapes, connectors, formulas, or charts never re-invokes an image model.

> **State labels used across the docs:** **Current** = already in the repo · **Active** = current main dev line · **Frozen** = done, maintenance only · **Target** = final architecture direction, **not yet implemented**. Do not treat `Target` items as existing files, APIs, or stable behavior.

---

## What lives here

| Line | Location | Status | Notes |
|---|---|---|---|
| **diagram2ppt** | `work/diagram2ppt/` | 🚧 **Active** | Framework/tech diagrams → native PPTX/SVG (v2 stable, v3 rewrite) |
| **core i2e poster pipeline** | `extractor/` `editor/` `render/` `capture/` etc. | ❄️ **Frozen** | Marketing poster → IR → editor → corrections flywheel |
| **editable omnimatte** | `work/poster/` `work/omnimatte.py` `work/plate.py` | ❄️ **Frozen (paused)** | `IMG_9493.jpg` → layered omnimatte + 4 edit classes |
| **Visual Design Decompiler** | `work/diagram2ppt/v3/` (evolving) | 🎯 **Target** | Evidence → Components → Editable IR → multi-format; audit-driven refinement — *not yet implemented* |

See `work/diagram2ppt/STATUS.md` for the diagram2ppt v2/v3 split and current blockers.

---

## Quick start

### diagram2ppt v2（稳定可用）

```bash
# v2 baseline: framework.png → native PPTX
python -m work.diagram2ppt.v2.run work/diagram2ppt/v2_out/framework_2x.png

# Best delivered artifact:
# work/diagram2ppt/v22_out/diagram_final.pptx
```

### diagram2ppt v3（活跃开发，尚未稳定）

```bash
# v3 agentic pipeline
python -m work.diagram2ppt.v3.run <image.png> -o work/diagram2ppt/v3_out --max-rounds 5
```

v3 currently defaults to SiliconFlow `Qwen/Qwen3.6-35B-A3B` and may timeout on large diagrams.

### core i2e poster pipeline（冻结维护）

```bash
pip install -r requirements.txt

# Offline smoke test
python -m extractor.extract poster.png -o out.ir.json --provider mock

# Edit in browser
python -m editor.server out.ir.json        # http://127.0.0.1:8765
```

---

## Repository map

```
ir/                     IR v1 schema (load-bearing wall) + example + contract
extractor/              Node ② — flat image -> IR  (mock | openai-compat | anthropic)
ocr/                    Text geometry (RapidOCR / Paddle / Tesseract)
segment/                Foreground cutouts (rembg | SAM-2 | MobileSAM)
inpaint/                Background reconstruction (flat | opencv | LaMa)
editor/                 Node ③ — browser editor + correction capture backend
render/                 Node ④ — layered / fallback PNG export
capture/                Node ⑤ — correction capture (the moat data)
bench/                  Flywheel north-star metric
tests/                  Offline smoke + pytest suites
work/diagram2ppt/       Active: diagram → native PPTX/SVG pipeline
work/poster/            Completed: omnimatte-style poster de-layering
work/lib/               Shared math helpers
archive/                Older experiments and superseded code
```

---

## Test status

```bash
python -m pytest tests/ work/diagram2ppt/tests/ -q
# 104 passed, 0 failed
```

---

## Key documents

| Document | What it covers |
|---|---|
| [STATUS.md](STATUS.md) | Project-wide current state and strategy |
| [BUILD-PLAN.md](BUILD-PLAN.md) | Original MVP scope (poster flywheel) |
| [docs/positioning-20260610.md](docs/positioning-20260610.md) | Strategic pivot: from RGBA layers to semantic native structure |
| [docs/diagram2ppt-progress.md](docs/diagram2ppt-progress.md) | diagram2ppt v1→v3 evolution |
| [work/diagram2ppt/STATUS.md](work/diagram2ppt/STATUS.md) | diagram2ppt v2/v3 status |
| [work/diagram2ppt/DEFECTS.md](work/diagram2ppt/DEFECTS.md) | framework.png reconstruction defect ledger |
