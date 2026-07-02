# v3 Execution Semantics Specification

> Version: execution-semantics-v1
> Scope: `work/diagram2ppt/v3/` runtime behavior, not future Target architecture.
> Purpose: describe what the system *actually* does today, define the smallest
> transition contract that makes it auditable, and mark the gap to a true
> execution-graph runtime.

---

## 1. Current system verdict

The v3 runtime is **not** an OS-style kernel, a graph scheduler, or a compiler
backend. It is a **traceable state interpreter (TSI)**:

```text
RuntimeState       := mutable snapshot of the reconstruction run
Planner            := heuristic policy that proposes mutations
Operator           := named wrapper around a Planner/module procedure
AuditAgentSystem   := loop controller that picks the next operator
state_log.json     := append-only trace of state snapshots
kernel.replay()    := snapshot replay, not execution replay
```

The dominant operation is in-place mutation:

```text
S ← mutate(S)
S ← mutate(S)
S ← mutate(S)
```

not:

```text
S(t+1) = f(S(t), event)
```

This document makes that fact explicit, then defines the minimal contract that
already exists and the contract that must be added to reach a deterministic
state-machine kernel.

---

## 2. State formalism

### 2.1 RuntimeState fields (canonical)

These fields are owned and written by the kernel/operator layer:

| Field | Type | Semantics |
|-------|------|-----------|
| `input_image` | str | Source image path (immutable after init). |
| `out_dir` | str | Output directory (immutable after init). |
| `round` | int | Number of repair/agent rounds completed. |
| `stage` | str | Current lifecycle stage (idle/planning/composing/auditing/refining/accepted/failed/interrupted). |
| `ir` | dict\|None | The Global Native IR blackboard. |
| `strategy_plan` | dict\|None | Semantic region plan derived from entities. |
| `components` | list\|None | Derived Component IR (post-run tooling). |
| `task_graph` | dict\|None | Region task graph built from IR. |
| `audit_tasks` | list\|None | Unified executable audit tasks. |
| `defects` | list | Current verifier/visual-review defects. |
| `metrics` | dict | Current scalar metrics (visual_delta, coverage_explained, ...). |
| `renderer_mode` | str\|None | `true_powerpoint`, `proxy`, or `unavailable`. |
| `last_verify_result` | dict\|None | Last `render_and_verify` return value. |
| `last_proposal_result` | dict\|None | Last `run_proposal_phase` return value. |
| `last_pptx` | str\|None | Path to last built PPTX. |
| `last_compare_png` | str\|None | Path to last comparison PNG. |
| `last_svg` | str\|None | Path to last SVG canonical output. |
| `transitions` | list[Transition] | Append-only mutation log. |
| `config` | dict | Run config (max_rounds, etc.). |
| `run_memory` | dict | Same-source memory bookkeeping. |
| `artifacts` | dict[str,bool] | Existence map of known artifact files. |

### 2.2 Derived vs canonical

Derived fields are recomputed from the IR or the filesystem:

- `metrics` is **derived** from `ir["metrics"]` but copied into state for speed.
- `defects` is **derived** from `ir["defects"]`.
- `components`/`task_graph`/`audit_tasks` are **derived** from `ir` + `strategy_plan`.
- `artifacts` is **derived** from `out_dir` file existence.

Rule: derived fields must be reconstructible from canonical fields + files.
If they drift, the drift is a bug.

### 2.3 Who mutates RuntimeState

Today:

1. `PlannerKernel.__init__` copies Planner fields into state.
2. `Operator.run()` creates a deep copy of state, mutates the copy, returns it.
3. `PlannerKernel.transition()` assigns the returned copy to `self.state`,
   calls `_sync_from_planner()`, appends a Transition, writes `state_log.json`.
4. Legacy code and many modules mutate `planner.ir` directly.

This means RuntimeState is **not** the single source of truth for the IR.
The Planner's `self.ir` is. The kernel is a follower, not an owner.

---

## 3. Transition formalism

### 3.1 What is a Transition

```python
Transition(
    id,              # unique transition id
    timestamp,       # wall-clock time
    stage_from,      # stage before operator
    stage_to,        # stage after operator
    operator,        # operator name
    inputs,          # arguments passed to operator
    outputs,         # summary of operator outputs
    artifact_paths,  # files the operator is expected to touch
    error,           # optional error payload
    checkpoint_path, # optional checkpoint
)
```

A Transition records that an operator ran. It does **not** record the full
state delta. Reconstructing the delta requires diffing IR snapshots.

### 3.2 Current transition contract (weak)

The only guarantees today:

- `stage_from` equals the state stage before the operator runs.
- `stage_to` is set by the operator's declared `target_stage` unless the
  operator explicitly changes it.
- One Transition is appended per `kernel.transition()` call.
- `state_log.json` is written after every transition.

What is **not** guaranteed:

- Operators are not pure functions.
- Inputs/outputs do not capture the full pre/post state.
- No validation that the operator only touches declared fields.
- No rollback of filesystem artifacts if an operator fails mid-way.

### 3.3 Required transition contract (target)

To become a true state-machine kernel, every operator must satisfy:

```text
Given:  state(t)  and  inputs
Return: state(t+1)
Where:  state(t+1) differs from state(t) only in declared write-fields
And:    side effects are limited to declared artifact paths
And:    operator is deterministic given state(t) + inputs + external files
```

This is **not** satisfied today.

---

## 4. Operator taxonomy

### 4.1 Current operators and their real behavior

| Operator | Declared target | Real behavior | Pure? | Notes |
|----------|-----------------|---------------|-------|-------|
| `perceive` | planning | Runs `planner.plan()`: perception, content handling, candidate selection, writes files. | No | Mutates Planner.ir, strategy_plan, filesystem. |
| `compose` | composing | No-op placeholder. | Yes | Currently folded into `perceive`. |
| `render_verify_audit` | auditing | Builds PPTX, renders, verifies, runs visual review, mutates IR defects/metrics. | No | Heavy side effects; writes PPTX/PNG/JSON. |
| `task_graph` | refining | Builds task graph from IR. | Yes-ish | Only writes to state.task_graph; no IR mutation. |
| `proposal_phase` | refining | Runs multi-agent region proposals, commits verified candidates, mutates IR. | No | May write candidate files. |
| `component_cleanup` | refining | Removes redundant native components. | No | Mutates IR.elements. |
| `repair` | refining | Runs one specialist agent round, records patch. | No | Mutates IR via agent.run(). |
| `rollback_or_accept` | refining | Decides patch fate; may restore IR from snapshot. | No | Mutates IR, snapshots, defects. |
| `derive_components` | auditing | Builds Component IR from IR + strategy_plan. | Yes-ish | Writes files; reads state. |
| `audit_tasks` | auditing | Unifies defects into AuditTasks. | Yes-ish | Writes files; reads state. |
| `svg_loop` | auditing | Renders IR to SVG/PNG/diff. | No | Calls v2 render/export; writes files. |
| `accept` | accepted | Sets IR.status = accepted. | No | Mutates IR.status. |
| `fail` | failed | Sets IR.status = failed. | No | Mutates IR.status. |
| `finalize` | finalizing | Best-effort post-run derivation. | No | Chains other operators. |
| `legacy_planner_loop` | finalizing | Runs original `planner.run()` end-to-end. | No | Encapsulates the whole old loop. |

### 4.2 Three real operator categories

#### A. Wrappers around Planner methods (mutators)

- `perceive`, `render_verify_audit`, `proposal_phase`, `repair`,
  `rollback_or_accept`, `component_cleanup`

These operators exist so the kernel can name and log planner actions, but they
do not isolate state. They are the main source of "mutable interpreter" behavior.

#### B. Pure-ish derived-state operators

- `task_graph`, `derive_components`, `audit_tasks`

These read canonical state and write derived state. They are closest to pure
reducers but still write files as a side effect.

#### C. Lifecycle terminal operators

- `accept`, `fail`, `finalize`, `legacy_planner_loop`

These mark outcomes or bridge to legacy behavior.

---

## 5. Execution model

### 5.1 Current model: loop with operator dispatch

```text
AuditAgentSystem.run()
  ├─ kernel.transition("perceive")           # one big step
  ├─ derive_components / audit_tasks / svg_loop
  ├─ kernel.transition("render_verify_audit")
  ├─ derive_artifacts
  ├─ if passed: accept
  └─ for iteration in max_rounds:
        decision = _decide_next_action(state)
        if proposal_phase:
           task_graph → proposal_phase → component_cleanup? → render_verify_audit
        elif single_repair:
           repair → render_verify_audit → rollback_or_accept
        elif stop: break
        derive_artifacts
        if passed: accept
  └─ finalize → accept|fail
```

This is a **linear loop** with branching inside the loop body. There is no
execution DAG, no parallelism, no dependency resolution.

### 5.2 Decision logic

`_decide_next_action()` is a heuristic policy:

- No actionable defects + no visual review defects → stop.
- Iteration == 1, or visual defects > 0, or missing coverage → proposal_phase.
- Otherwise, if a concrete repair exists → single_repair.

This policy is hard-coded, not learned or declarative.

### 5.3 Gap to graph execution

A true graph runtime would:

1. Build a DAG of operations where edges are data dependencies.
2. Schedule independent nodes concurrently when safe.
3. Re-run only changed subtrees after a state mutation.
4. Cache results by input hash.

v3 does none of this. The "graph" is implicit in the AuditAgentSystem loop.

---

## 6. Replay semantics

### 6.1 Current replay

`kernel.replay(state_log_path)` loads the last serialized `RuntimeState` and
restores `planner.ir` and `planner.strategy_plan` from it.

### 6.2 What replay can do

- Restore the kernel to the saved state.
- Provide a debugging/inspection checkpoint.
- Let downstream tools read the saved state without re-running.

### 6.3 What replay cannot do

- Re-execute the operator sequence deterministically.
- Reconstruct intermediate decisions from inputs.
- Simulate "what if agent X had run instead of agent Y".
- Guarantee the same result on a different machine or after code changes.

### 6.4 Why it is snapshot replay

Because Transitions do not record:

- Full pre-state.
- Full post-state delta.
- Deterministic seeds or external file versions.
- The exact code version that produced each transition.

To make replay an execution replay, every operator would need to be pure and
deterministic, and every Transition would need to capture enough information to
recompute the next state.

---

## 7. Module ownership and coupling

### 7.1 Layer map

```text
┌──────────────────────────────────────────────┐
│ AuditAgentSystem                               │  loop policy / tool picker
├──────────────────────────────────────────────┤
│ PlannerKernel + RuntimeState + operators     │  traceable state interpreter
├──────────────────────────────────────────────┤
│ Planner                                        │  blackboard orchestrator
├──────────────────────────────────────────────┤
│ strategy / quality_gate / task_graph           │  semantic mutation layer
├──────────────────────────────────────────────┤
│ builder / renderer / verifier / visual_review  │  execution/evaluation layer
├──────────────────────────────────────────────┤
│ components / audit_tasks / svg_loop            │  post-run derived tools
└──────────────────────────────────────────────┘
```

### 7.2 Coupling analysis

| Module | Reads | Writes | Coupled to |
|--------|-------|--------|------------|
| `Planner.plan()` | image, VLM, OCR, memory | `self.ir`, `self.strategy_plan`, files | perception_orchestrator, content_orchestrator, strategy, migrate, quality_gate, builder, renderer, verifier |
| `strategy.apply_ir_strategy()` | IR, plan | `ir["strategy_plan"]`, element.ext.strategy | IR structure |
| `strategy.apply_defect_strategy()` | IR defects, element strategy | defect.suggested_agent, defect.strategy | IR structure |
| `quality_gate.apply()` | IR, image | IR elements (text/font/color/...) | IR structure |
| `builder.build_pptx()` | IR | PPTX file, build_stats | IR structure, python-pptx |
| `renderer` | PPTX | PNG files | PowerPoint / PIL |
| `verifier.verify()` | IR, image, rendered PNG | `ir["defects"]`, `ir["metrics"]` | IR structure |
| `visual_review.review()` | compare PNG | review dict | VLM |
| `components.build_components()` | IR, strategy_plan | components.json, crops, sub-IRs | IR structure |
| `audit_tasks.unify_tasks()` | IR, components | audit_tasks.json | IR structure |

Key observation: **the IR is the global mutable database**. Almost every module
reads and writes it. The kernel only observes the IR after the fact.

### 7.3 Hidden state channels

Besides the IR, several hidden channels carry state:

- `Planner._snapshots` and `_patch_preimages` enable rollback.
- `Planner._visual_review_cache` avoids duplicate VLM calls.
- `Planner._agent_attempts` and `_failed_routes` implement quarantine.
- Filesystem artifacts carry state that the kernel does not track.
- Environment variables (`I2E_USE_RUN_MEMORY`, `I2E_BUILD_PROFILE`) change behavior.

These channels are not in RuntimeState, so `state_log.json` is an incomplete
picture of the runtime.

---

## 8. Effect isolation violations

### 8.1 Current violations of a clean transition contract

1. **Operators mutate Planner.ir directly.** The kernel copies state back, but
the source of truth remains in the Planner.

2. **Verifier writes back to IR.** `verifier.verify()` accepts `ir` and mutates
`ir["defects"]` and `ir["metrics"]` as a side effect.

3. **Strategy mutates IR in-place.** `apply_ir_strategy` and `apply_defect_strategy`
modify the IR without returning a new object.

4. **Visual review mutates IR.** `visual_review.attach_to_ir()` writes into
`ir["visual_review"]`.

5. **Rollback restores from private Planner snapshots.** The kernel does not
own the rollback mechanism.

6. **Filesystem writes are not transactional.** If an operator fails mid-way,
partial files may remain.

7. **Environment variables change semantics.** The same code can produce
different transitions depending on env flags.

### 8.2 Why these are acceptable today

- The system is an experimental research pipeline, not a production service.
- The goal is rapid iteration on agent/policy ideas, not formal correctness.
- The kernel was introduced incrementally to add traceability, not to replace
the Planner.

### 8.3 Why these block a true kernel

- Deterministic replay is impossible.
- Module interactions cannot be statically analyzed.
- Testing is integration-heavy because state boundaries are unclear.
- Parallel execution is unsafe.
- Policy learning is hard because the state space is implicit.

---

## 9. Minimal spec to make the kernel real

To move from TSI to a deterministic state-machine kernel, enforce the following
incrementally:

### 9.1 Phase A: IR immutability inside operators

- Every operator receives a deep-copied IR.
- Every operator returns a new IR (or None for no change).
- The kernel is the only entity that assigns `planner.ir = new_ir`.

### 9.2 Phase B: effect declarations

- Each operator declares:
  - `reads: list[str]` (state fields it reads)
  - `writes: list[str]` (state fields it writes)
  - `artifacts: list[str]` (files it writes)
  - `idempotent: bool`
- The kernel validates that an operator does not write outside its declared
  `writes` set.

### 9.3 Phase C: pure derived operators

- `task_graph`, `derive_components`, `audit_tasks` become pure functions:
  `f(ir, strategy_plan) -> derived_object`.
- They do not write files; the kernel writes files from the returned objects.

### 9.4 Phase D: deterministic replay

- Each Transition records:
  - `input_state_hash` (hash of canonical pre-state)
  - `output_state_hash` (hash of canonical post-state)
  - `code_version` (git hash or module version)
  - `seed` (for any stochastic operation)
- `kernel.replay()` can re-run operators from recorded inputs and verify hashes.

### 9.5 Phase E: execution graph

- Replace the AuditAgentSystem loop with a declarative task graph.
- Nodes = operators or agent proposals.
- Edges = data dependencies (e.g., `render_verify_audit` needs `build_pptx`).
- Scheduler = topological execution + caching.

---

## 10. Module rewrite vs wrapper map

| Module | To reach Phase A/B | Effort |
|--------|--------------------|--------|
| `Planner.plan()` | Split into `perceive`, `compose`, `select_initial_ir` operators that return new IRs. | Large |
| `Planner.render_and_verify()` | Split into `build`, `render`, `verify`, `visual_review` operators. | Medium |
| `Planner.run_round()` | Make agent.run() return a patch; apply patch in kernel. | Medium |
| `Planner.accept_or_rollback()` | Move snapshot/rollback into kernel; make it operate on IR copies. | Medium |
| `strategy.apply_*` | Return new IR instead of mutating. | Small |
| `quality_gate.apply()` | Return new IR instead of mutating. | Small |
| `verifier.verify()` | Return defects/metrics instead of mutating IR. | Small |
| `visual_review.attach_to_ir()` | Return review object; kernel attaches. | Small |
| `components / audit_tasks / svg_loop` | Make pure; kernel handles file writes. | Small |
| `builder / renderer` | Keep as effectful execution layer, but kernel owns artifact paths. | Small |

---

## 11. What to preserve

The current system already has valuable properties that the refactor must keep:

- **Offline testability**: most operators have mock/stub tests.
- **Run manifest diagnostics**: every run records outcome and blockers.
- **Component/audit/SVG scaffolds**: derived-state tooling is already built.
- **Fallback policy**: local, documented, tracked fallback is enforced.
- **Proxy vs true PowerPoint distinction**: acceptance gate is correct.

---

## 12. Summary

| Question | Current answer |
|----------|----------------|
| Is kernel a real scheduler? | No. It is a state wrapper + logger. |
| Is RuntimeState a state machine? | No. It is a mutable snapshot follower. |
| Are operators pure reducers? | No. Most are mutator wrappers. |
| Is replay execution replay? | No. It is snapshot replay. |
| Is there an execution graph? | No. There is a linear loop with heuristic branching. |
| Is the system auditable? | Partially. Transitions exist but do not capture full deltas. |
| What is the next jump? | Enforce IR immutability + effect declarations + deterministic replay hashes. |

## 13. Executable contract layer (Current as of 2026-07-02)

A concrete, opt-in contract layer now lives in `work/diagram2ppt/v3/runtime/contract.py`.
It does not yet replace the legacy operators, but it defines the interfaces and
has been validated with two pure operators and offline tests.

### 13.1 Core classes

- `ImmutableOperator`: abstract base that receives a deep-copied `RuntimeState`
  and returns `(new_state, effects)`.
- `SideEffect` / `WriteFileEffect` / `UpdatePlannereffect` / `NoEffect`:
  first-class descriptions of what the kernel should commit.
- `Transaction`: encapsulates one operator call with pre/post state hashes and
  effect list.
- `commit_effects()`: the only place in the contract layer allowed to perform
  side effects.
- `state_hash()`: stable hash of canonical state fields for replay/cache keys.

### 13.2 Declarative operator contract

Each immutable operator declares:

```python
reads = ("ir", "strategy_plan")
writes = ("task_graph",)
artifacts = ("task_graph.json",)
idempotent = True
```

This lets the kernel:

- Validate preconditions (`reads` must be non-None).
- Detect undeclared writes (future validation hook).
- Build a dependency graph from `reads`/`writes`.
- Cache results by `transition_hash()`.

### 13.3 First pure operators

| Operator | Reads | Writes | Artifacts | Status |
|----------|-------|--------|-----------|--------|
| `ImmutableTaskGraphOperator` | `ir`, `strategy_plan` | `task_graph` | `task_graph.json` | ✅ tested |
| `ImmutableAuditTasksOperator` | `ir`, `components` | `audit_tasks` | `audit_tasks.json` | ✅ tested |

These operators:

- Do not touch `Planner`.
- Do not write files directly.
- Return a new `RuntimeState` and a list of `SideEffect` objects.
- Are deterministic given the same input state.

### 13.4 Test coverage

`work/diagram2ppt/tests/test_runtime_contract.py` covers:

- `state_hash` stability and sensitivity.
- `Transaction` pre/post hash recording.
- Immutable operator non-mutation of input state.
- `commit_effects` for file writes and planner updates.
- `ImmutableTaskGraphOperator` / `ImmutableAuditTasksOperator` purity.
- Deterministic `transition_hash`.
- Error capture in `Transaction.execute()`.

### 13.5 Adoption path

The legacy `Operator` registry remains the default.  New pure operators can be
added side-by-side.  The next incremental steps are:

1. Add a kernel method `transition_immutable(op_name, **inputs)` that uses the
   `Transaction` / `commit_effects` path instead of the legacy operator path.
2. Migrate `derive_components` to an immutable operator.
3. Wrap `strategy.apply_*`, `quality_gate.apply`, `verifier.verify`, and
   `visual_review.attach_to_ir` so they return deltas instead of mutating IR.
4. Once enough operators are immutable, switch `AuditAgentSystem` to dispatch
   through the immutable registry and remove legacy mutation operators.

### 13.6 What is still missing

- The kernel does not yet commit immutable-operator effects by default.
- No write-field validation hook.
- No code-version / seed recording in `Transaction`.
- No transactional rollback of partial file writes.

These remain Target work; the contract layer makes them implementable
incrementally without rewriting the whole system.

---

## 14. Execution graph kernel (Current as of 2026-07-02)

The next jump is implemented in `work/diagram2ppt/v3/runtime/graph.py`:
`PlannerKernel` can now execute a dependency graph of operators via
`execute_graph(graph, cache)`.

### 14.1 Core classes

- `ExecutionGraph`: a DAG of `GraphNode`s and `DependencyEdge`s.
- `GraphNode`: one operator invocation with `inputs`, `depends_on`, and declared
  `produced_fields` / `produced_artifacts`.
- `DependencyEdge`: a data dependency between two nodes on a state field or
  artifact path.
- `GraphScheduler`: topologically sorts the graph, executes nodes, records a
  `GraphExecutionTrace`, and caches results by deterministic node keys.

### 14.2 Scheduler behavior

```text
for each node in topological_order(graph):
    cache_key = node.compute_cache_key(kernel.state)
    if cache_key in cache:
        kernel.state = cached_state
        mark cache hit
    else:
        if operator is ImmutableOperator:
            new_state = op.run(kernel.state)
        else:
            new_state = kernel.transition(op_name)
        cache[cache_key] = new_state
    append transition to trace
```

Independent nodes are grouped into waves by `independent_groups()`; the current
scheduler runs waves serially and nodes inside a wave serially.  A parallel
executor can be swapped in later without changing graph semantics.

### 14.3 Cache key semantics

`GraphNode.compute_cache_key()` hashes:

- node id
- operator name
- node inputs
- canonical upstream state (input image, out_dir, round, IR, strategy_plan,
  components, renderer_mode)

It deliberately excludes derived outputs (`task_graph`, `audit_tasks`) and the
runtime `stage` so that the same upstream state hits the cache even after a
previous graph execution mutated the stage or produced derived fields.

### 14.4 First graph execution tests

`work/diagram2ppt/tests/test_runtime_graph.py` covers:

- Topological order respects dependency edges.
- Cycle detection raises `RuntimeError`.
- Independent groups split nodes into parallelizable waves.
- Graph scheduler runs `immutable_task_graph` → `immutable_audit_tasks` in order.
- Cache hits on re-execution with the same upstream state.
- Cache key changes when upstream state changes.
- Graph serialization via `ExecutionGraph.to_dict()`.

### 14.5 What this unlocks

- **Parallel execution**: independent nodes can run concurrently once a thread/
  process executor is added.
- **Partial recompute**: after an IR mutation, only downstream nodes need
  re-execution.
- **Deterministic caching**: expensive operators (render/verify/VLM) can be
  memoized.
- **Auditable plans**: the graph itself is a first-class artifact that can be
  saved, diffed, and reviewed.

### 14.6 What is still missing

- `AuditAgentSystem` does not yet build or execute a graph; it still uses
  linear `kernel.transition()` calls.
- No automatic graph construction from operator `reads`/`writes`.
- No parallel executor.
- No graph-level rollback or checkpointing.
- Effects from immutable operators are not yet committed through the kernel.

These are the next Target increments after the graph kernel foundation.

---

**Bottom line:** v3 is now an event-driven state machine kernel with a
concrete graph execution layer on top. The remaining gap to a full Visual
Design Compiler runtime is populating `ExecutionGraph` automatically from the
declarative operator contracts and wiring `AuditAgentSystem` to schedule via
the graph instead of linear transitions.

---

**Bottom line:** v3 is a strong, traceable AI pipeline. To become a Visual
Design Compiler runtime, it needs a transition semantics layer that makes the
IR immutable at operator boundaries and the kernel the sole owner of state
mutation. The executable contract layer is the first concrete step in that
direction.
