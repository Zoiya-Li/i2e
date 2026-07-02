# diagram2ppt v3 — Next-Gen Agentic Pipeline

**Status:** 🚧 **Active research / not stable**. No `status: accepted` output yet.

**Philosophy:** "Search and audit." Use a blackboard IR, semantic region plans, competing specialist agents, and real PowerPoint rendering verification to rebuild a diagram as **100% native editable objects** (no screenshots, no raster crops, no pictures).

---

## What it adds over v2

| Capability | v2 | v3 |
|---|---|---|
| Fidelity model | Native + faithful raster crops | All-native only |
| Optimization | Per-element residual | Region-level semantic reconstruction |
| Control loop | Global render → refine/demote | Agent proposals + accept/rollback audit |
| Surfaces | CV-traced or hybrid raster | Procedural / SVG vector surfaces |
| Auditability | Informal checkpoints | `audit_trace.json`, `task_graph.json`, method contracts |

---

## Architecture

```
run.py
  └── AuditAgentSystem
        └── Planner
              ├── PerceptionOrchestrator  (CV + VLM + OCR evidence fusion)
              ├── ContentOrchestrator     (v2 handlers as typed agents)
              ├── Strategy                (semantic region plan)
              ├── Quality Gate + Typography
              ├── Initial IR selection by real render
              ├── render_and_verify       (true PowerPoint render)
              ├── Proposal Phase          (region-level agent search)
              └── Single-defect fallback
```

---

## How to run

```bash
python -m work.diagram2ppt.v3.run <image.png> -o work/diagram2ppt/v3_out --max-rounds 5
```

**Current issue:** the CLI default model is `Qwen/Qwen3-VL-32B-Instruct` via SiliconFlow (set by `run.py`; the provider-level fallback, used only when `I2E_VLM_MODEL` is unset, is the older `Qwen/Qwen3.6-35B-A3B`). It can still time out or fail to converge on large diagrams — but every run writes a diagnosable `run_manifest.json` regardless. If you hit timeouts, try:

```bash
I2E_VLM_TIMEOUT=240 I2E_VLM_TOTAL_TIMEOUT=300 \
  python -m work.diagram2ppt.v3.run <image.png> -o work/diagram2ppt/v3_out --max-rounds 3
```

---

## Runtime semantics

v3 is currently a **traceable state interpreter (TSI)**, not a graph runtime or
compiler backend. The `PlannerKernel` / `RuntimeState` / `state_log.json` layer
adds auditable transitions around the existing Planner, but operators are still
mostly wrappers around in-place mutations of the Global Native IR.

See [`docs/execution-semantics-spec.md`](docs/execution-semantics-spec.md) for
an honest audit of what is real today versus what is required for a true
state-machine kernel.

---

## Relationship to v2

- **v3 depends on v2.** It imports v2 for decompose fallback, handlers, build_pptx, render, diff, and snapshot.
- **v2 is the production baseline.** Use v2 if you need a result today.
- **v3 is where new architecture work happens.** When v3 is stable and beats v2, it will become the default.
