"""Runtime state and transition records for the v3 state machine kernel.

`RuntimeState` is the single source of truth for everything that changes during
a reconstruction run. `Transition` records every state mutation so the run is
auditable and replayable.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Transition:
    """One state mutation executed by the PlannerKernel."""

    id: str
    timestamp: float
    stage_from: str
    stage_to: str
    operator: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str | None] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    checkpoint_path: str | None = None

    @classmethod
    def create(
        cls,
        stage_from: str,
        stage_to: str,
        operator: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        artifact_paths: dict[str, str | None] | None = None,
        error: dict[str, Any] | None = None,
        checkpoint_path: str | None = None,
    ) -> Transition:
        return cls(
            id=f"t_{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            stage_from=stage_from,
            stage_to=stage_to,
            operator=operator,
            inputs=inputs or {},
            outputs=outputs or {},
            artifact_paths=artifact_paths or {},
            error=error,
            checkpoint_path=checkpoint_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeState:
    """Single source of truth for a v3 reconstruction run.

    The kernel keeps this object serialized after every transition. Agents and
    operators should read from it and return an updated copy; they should not
    mutate it in place.
    """

    version: str = "runtime-v1"
    input_image: str = ""
    out_dir: str = ""
    round: int = 0
    stage: str = "idle"

    # Core design source (v3 Global Native IR)
    ir: dict[str, Any] | None = None

    # Semantic structure (promoted to runtime in later phases)
    strategy_plan: dict[str, Any] | None = None
    components: list[dict[str, Any]] | None = None
    task_graph: dict[str, Any] | None = None
    audit_tasks: list[dict[str, Any]] | None = None

    # Evidence / audit
    defects: list[dict[str, Any]] = field(default_factory=list)
    visual_review: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    # Render provenance
    renderer_mode: str | None = None  # true_powerpoint | proxy | unavailable
    last_render_png: str | None = None
    last_compare_png: str | None = None
    last_pptx: str | None = None
    last_svg: str | None = None

    # Configuration
    config: dict[str, Any] = field(default_factory=dict)
    run_memory: dict[str, Any] = field(default_factory=dict)

    # Transition log
    transitions: list[Transition] = field(default_factory=list)
    artifacts: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["transitions"] = [t.to_dict() for t in self.transitions]
        return payload

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        import json

        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return p
