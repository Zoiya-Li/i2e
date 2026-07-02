# i2e — image → editable

Turn flat AI-generated visuals into **editable, on-brand design assets**.

**Today (Current):** an *image → editable* pipeline, currently focused on **structured diagrams → native PowerPoint / SVG**. The original marketing-poster flywheel and the editable-omnimatte prototype are **Frozen** (complete). See [STATUS.md](STATUS.md).

**Long-term (Target):** a **Visual Design Decompiler** — recover any flat visual asset into an editable, auditable, iterable, cross-format **design source structure**, not just one SVG or PPTX:

```
Visual Artifact → Evidence → Components → Editable Design IR → SVG / PPTX / (future) Figma-like JSON / HTML
```

The product contract: **AI is used only during decompile.** After export, editing is **native, deterministic, fast, auditable, and model-independent** — changing text, color, shapes, connectors, formulas, or charts never re-invokes an image model. This is the wedge against pixel/RGBA layer decomposition (Lovart / OmniPSD / Qwen-Image-Layered): we output semantic native objects (text boxes, shapes, connectors, OMML formulas, native charts), not raster layers.

> **State labels used throughout the docs:** **Current** = already in the repo · **Active** = current main dev line · **Frozen** = done, maintenance only · **Target** = final architecture direction, **not yet implemented**. Do not treat `Target` items as existing files, APIs, or stable behavior.

---

## What lives here

| Line | Location | Status | Notes |
|---|---|---|---|
| **diagram2ppt** | `work/diagram2ppt/` | 🚧 **Active** | Framework/tech diagrams → native PPTX/SVG (v2 stable, v3 rewrite) |
| **core i2e poster pipeline** | `extractor/` `editor/` `render/` `capture/` … | ❄️ **Frozen** | Marketing poster → IR → browser editor → correction flywheel |
| **editable omnimatte** | `work/omnimatte.py` `work/lib/` … | ❄️ **Frozen (paused)** | `IMG_9493.jpg` → layered omnimatte + 4 edit classes |
| **Visual Design Decompiler** | `work/diagram2ppt/v3/` (evolving) | 🎯 **Target** | Evidence → Components → Editable IR → multi-format; audit-driven refinement — *not yet implemented* |

The **IR** is the load-bearing wall: [`ir/ir-v1.schema.json`](ir/ir-v1.schema.json). Any persisted IR must validate against it.

---

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/Zoiya-Li/i2e.git
cd i2e

python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Model weights and caches are NOT in the repo** (they are large and git-ignored). Formula/segmentation models download on first use via `work/diagram2ppt/v3/models/loader.py` (Hugging Face). In mainland China, set a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**API key** (only needed for the live VLM pipelines — all smoke/pytest tests run fully offline). Create a `.env` at the repo root (it is git-ignored):

```dotenv
# SiliconFlow (default provider) — or set SILICONFLOW_API_KEY
I2E_VLM_API_KEY=sk-xxxxxxxx
I2E_VLM_BASE_URL=https://api.siliconflow.cn/v1
I2E_VLM_MODEL=Qwen/Qwen3-VL-32B-Instruct
```

Heavy/optional extras (install only when needed, ideally in a separate env): `simple-lama-inpainting` (photographic inpaint), `ultralytics` (SAM-2 / MobileSAM), `diffusers`+`torch` (SD-1.5). `python-pptx` (PPTX export) is in `requirements.txt`.

---

## Quick start

### diagram2ppt — v2 (stable baseline)

```bash
# default: native SVG, zero raster
python -m work.diagram2ppt.v2.run framework.png -o work/diagram2ppt/v2_out

# legacy PPTX export (native objects + faithful-crop fallback)
python -m work.diagram2ppt.v2.run framework.png --legacy -o work/diagram2ppt/v2_out
```

### diagram2ppt — v3 (active research pipeline)

```bash
python -m work.diagram2ppt.v3.run test.png -o work/diagram2ppt/v3_out --max-rounds 5
```

v3 defaults to SiliconFlow `Qwen/Qwen3-VL-32B-Instruct`. Every run — success, timeout, `SIGTERM` kill, or crash — writes a diagnosable `run_manifest.json` (`outcome ∈ accepted / partial / rejected / error / interrupted`). v3 is **not yet stable**: it produces native output but has not reached `status: accepted` end-to-end.

### core i2e poster pipeline (frozen, offline)

```bash
# extract IR offline with the mock provider (no API key)
python -m extractor.extract poster.png -o out.ir.json --provider mock

# browser editor + correction capture
python -m editor.server out.ir.json        # http://127.0.0.1:8765
```

---

## v3 tooling / infrastructure (Current)

Offline, deterministic infra built around the v3 pipeline:

| Module | Purpose |
|---|---|
| `v3/runtime/` | Runtime state-machine kernel: `RuntimeState` as single source of truth, `PlannerKernel` dispatching operators, `state_log.json` transitions, `kernel.replay()` |
| `v3/pptx_stats.py` | Deterministic PPTX fingerprint: shape histogram, picture/OMML counts, `native_object_ratio`, sha256 |
| `v3/baselines/v2_framework.json` | Frozen, **measured** v2 delivery locked as a regression baseline |
| `v3/metrics.py` | Multi-dimensional metrics: `native_element_ratio`, `fallback_area_ratio`, `editability_score`, coverage |
| `v3/fallback.py` | Fallback audit: flags raster fallbacks that are undocumented or full-page |
| `v3/components.py` | Component IR: promote strategy regions into first-class components (lifecycle, local metrics, crops, sub-IR) → `components.json` |
| `v3/audit_tasks.py` | Unify verifier + visual_review defects into one executable `AuditTask` schema → `audit_tasks.json` |
| `v3/svg_loop.py` | SVG canonical loop: IR → SVG → PNG (`rsvg-convert`) → pixel diff vs source (debug/preview renderer) |
| `v3/builder.py` | Build profiles: `all_native` (zero raster) vs `product_delivery` (documented local fallback) via `--profile` / `I2E_BUILD_PROFILE` |
| `v3/triage.py` | Non-destructive scan/index of `v3_out*` run dirs → `v3_out_index.{json,md}` |
| `v3/docs/execution-semantics-spec.md` | Honest audit of the v3 runtime: TSI (traceable state interpreter) today vs state-machine kernel Target |

```bash
# index every v3 run directory by outcome + editability (writes nothing destructive)
python -m work.diagram2ppt.v3.triage -r work/diagram2ppt -o work/diagram2ppt/v3_out_index
```

---

## Testing

All core tests are **offline** (mock providers, stub segmenters, synthetic images — no API key, no network):

```bash
python -m pytest tests/ work/diagram2ppt/tests/ -q
# 191 passed  (latest known; rerun locally/CI before release)
```

---

## Repository map

```
ir/                     IR v1 schema (load-bearing wall) + example + contract
extractor/              Node ② — flat image → IR  (mock | openai-compat | anthropic)
ocr/                    Text geometry (RapidOCR / Paddle / Tesseract)
segment/                Foreground cutouts (rembg | SAM-2 | MobileSAM)
inpaint/                Background reconstruction (flat | opencv | LaMa)
editor/                 Node ③ — browser editor + correction-capture backend
render/                 Node ④ — layered / fallback PNG export
capture/                Node ⑤ — correction capture (the moat data)
bench/                  Flywheel north-star metric
fonts/  label/  verify/  Font/color match · VLM labeling · needs_review checks
tests/                  Offline smoke + pytest suites
work/diagram2ppt/v2/    Stable diagram → native PPTX/SVG baseline
work/diagram2ppt/v3/    Active agentic/audit rewrite (Decompiler experiment field)
work/lib/               Shared math/geometry helpers
docs/                   Positioning, progress, and design notes
```

Not in the repo (git-ignored, regenerated locally): `.env`, model weights & HF/MS caches, generated `*_out*` output directories, `snapshots/`, `archive/`, and `external/` third-party submodules.

---

## Status & honest limitations

- **v2** delivers usable PPTX/SVG and is the current production baseline (frozen, bug-fix only). The delivered `diagram_final.pptx` is a hybrid: native objects + ~26% raster fallback area.
- **v3** enforces all-native output (editability 1.0, zero fallback on current runs) but **loses on visual fidelity** (best `visual_delta` ≈ 0.36) and has **not** reached `status: accepted` end-to-end. The bottleneck is convergence/fidelity, not editability.
- Target-tier work (multi-layer IR, component-level local loop, audit-driven refinement, model-in-the-loop generation) is designed but **not implemented** — see the roadmap in `work/diagram2ppt/STATUS.md`.

---

## Key documents

| Document | Covers |
|---|---|
| [STATUS.md](STATUS.md) | Project-wide state, Decompiler target, delivered infra |
| [AGENTS.md](AGENTS.md) | Guide for AI coding agents (state vocabulary, contracts, commands) |
| [ir/README.md](ir/README.md) | IR v1 design contract and validation |
| [docs/positioning-20260610.md](docs/positioning-20260610.md) | Strategic pivot: RGBA layers → semantic native structure |
| [docs/diagram2ppt-progress.md](docs/diagram2ppt-progress.md) | diagram2ppt v1→v3 evolution |
| [work/diagram2ppt/STATUS.md](work/diagram2ppt/STATUS.md) | v2/v3 status + Target roadmap (P0–P5) |
| [work/diagram2ppt/DEFECTS.md](work/diagram2ppt/DEFECTS.md) | `framework.png` reconstruction defect ledger |
