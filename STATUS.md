# i2e — Status (2026-07-01)

**One line:** turn flat AI-generated visuals into editable, on-brand design assets — currently focused on **structured diagrams → native PowerPoint / SVG**.

**North star:** make the IR the missing interchange standard between generative pixels and production-ready design — earned by winning a wedge, not declared.

**Long-term target (Target):** evolve i2e from an *image → editable* converter into a **Visual Design Decompiler** — recover any flat visual asset into an editable, auditable, iterable, cross-format design source: `Visual Artifact → Evidence → Components → Editable Design IR → SVG / PPTX / (future) Figma-like JSON / HTML`. AI participates only during decompile; post-export editing stays native / deterministic / fast / auditable / model-independent. This is a **Target**, not current behavior — see the state legend below.

> **State legend (used throughout):** **Current** = already in the repo · **Active** = current main dev line · **Frozen** = done, maintenance only · **Target** = final architecture direction, **not yet implemented**. Genuine `Target` items are: a true multi-layer IR (Evidence/Component/Editable/Correction as separate schemas), the component-level local render/diff/**refine execution** loop, an executable refinement task **queue**, and cross-format lowering (Figma/HTML). Already **Current** (see `work/diagram2ppt/STATUS.md` §1.6): the run manifest, multi-dimensional metrics, fallback audit, Component/audit-task **scaffolds**, the SVG loop, build profiles, regression suite, and the **v3 runtime state-machine kernel** (`PlannerKernel` + `RuntimeState` + operators + `state_log.json` + `kernel.replay()`).

> **Strategic pivot (2026-06-10):** pixel-level decomposition (RGBA layers) is being commoditized by Lovart/OmniPSD and Qwen-Image-Layered. Our wedge is **semantic native structure**: text as text boxes, boxes as shapes, arrows as connectors, formulas as OMML, charts as data. See `docs/positioning-20260610.md`.

---

## Three lines

### 1. diagram2ppt — active (current main effort)

**Goal:** reconstruct academic/technical framework diagrams as native, editable PowerPoint / SVG.

**Location:** `work/diagram2ppt/`

**Status:**
- **v2 baseline** is usable and delivers real PPTX (`v22_out/diagram_final.pptx`)
- **v3** is a research rewrite toward an agentic/audit pipeline; not yet stable

| Version | State | Native fraction | Notes |
|---|---|---|---|
| v2 | ✅ Production baseline | 0.70 / 0.696 (count/area) | Hybrid: native objects + faithful raster crops |
| v2.4 | ✅ Best v2 delivery | 0.70 / 0.696 | Formula + chart experts online |
| v3 | 🚧 Active rewrite | N/A | All-native policy; no accepted output yet |

**Key blocker:** v3 (CLI default `Qwen/Qwen3-VL-32B-Instruct`) produces native output but still has not reached `status: accepted` end-to-end — it can time out or fail to converge on large diagrams. Every run now writes a diagnosable `run_manifest.json`; the remaining gap is convergence/fidelity, not editability.

**See:** `work/diagram2ppt/STATUS.md` for v2/v3 architecture and current outputs.

---

### 2. core i2e poster pipeline — completed, maintenance only

**Goal:** turn AI-generated marketing posters into a layered, editable IR with a correction flywheel.

**Location:** `extractor/`, `ocr/`, `segment/`, `inpaint/`, `verify/`, `editor/`, `render/`, `capture/`, `bench/`

**Status:** ✅ **Stage-complete (2026-05-27)**. All 7 smoke tests green. End-to-end demonstrated on the 风油精 poster: OCR text, rembg/SAM foreground cutouts, LaMa background reconstruction, font matching, browser editor, correction capture.

**Current mode:** **frozen / maintenance**. No active feature work. Kept because it validates the IR v1 contract and the flywheel loop.

**Run:**
```bash
python -m extractor.extract poster.png -o out.ir.json \
  --provider mock --ocr rapid --assets rembg --inpaint flat
python -m editor.server out.ir.json
```

---

### 3. editable omnimatte superpower — completed, paused

**Goal:** de-layer `IMG_9493.jpg` (风油精 × Häagen-Dazs poster) into true RGBA layers with smoke/shadow following, supporting 4 edit classes.

**Location:** `work/poster/`, `work/omnimatte.py`, `work/plate.py`, `work/assemble_omnimatte.py`, `work/edit_demo.py`, `work/lib/omnimatte_math.py`

**Status:** ✅ **Demonstration-complete (2026-06-07)**. Produced `poster/omnimatte.ir.json` and `poster_i2e_final.svg` with 12 curated layers + editable text on a cleaned plate.

**Current mode:** **paused**. Validated the omnimatte + plate-critic approach; not generalized to arbitrary posters.

---

## Repository health

### Tests

```bash
python -m pytest tests/ work/diagram2ppt/tests/ -q
# 191 passed  (latest known; rerun: pytest tests/ work/diagram2ppt/tests/ -q)
```

### Output organization

- `work/diagram2ppt/archive_202506/` contains 140 archived v3 experiment directories (2026-06-19 ~ 06-21)
- 59 recent v3 output directories remain in `work/diagram2ppt/` for active reference
- `work/poster_sd15_0739/`, `work/derender_IMG_9493/`, `work/gen_decompose/` are historical experiments

### Code relationship

```
                    ┌─────────────────────────────────────┐
                    │  i2e                                │
                    │  (shared IR v1 schema, tests,       │
                    │   correction-capture primitives)    │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼
  core i2e poster          work/poster/              work/diagram2ppt/
  (frozen)                 omnimatte                 (active)
                                                       ├─ v2/ baseline
                                                       └─ v3/ next-gen
```

---

## Current priorities

1. **Stabilize v3 end-to-end** — get it to run through a new image without timeout
2. **Clarify v2/v3 boundary** — v2 is the production baseline; v3 is the research successor
3. **Decide on poster vs diagram** — if we return to poster market, reuse v2/omnimatte learning; if we stay on diagrams, fold poster work into archival mode
4. **Update top-level docs** — this file and README.md now reflect reality (done 2026-06-27)

---

## Key documents

| Document | Covers |
|---|---|
| `docs/positioning-20260610.md` | Strategic pivot and competitive positioning |
| `docs/diagram2ppt-progress.md` | diagram2ppt v1→v3 detailed evolution |
| `docs/first-step-outward-20260607.md` | Early plan to find design-industry co-owner |
| `work/diagram2ppt/STATUS.md` | v2/v3 current state |
| `work/diagram2ppt/DEFECTS.md` | framework.png defect ledger |
| `docs/superpowers/plans/2026-06-04-i2e-editable-omnimatte.md` | Omnimatte implementation plan |
