"""Planner-controlled content extraction orchestration.

This turns the old v2 process_all() hard flow into explicit content tasks.
The concrete extractors still reuse the stable v2 handlers, but scheduling,
reporting, and failure accounting are planner-visible.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from work.diagram2ppt.v2 import handlers as legacy_handlers
from work.diagram2ppt.v2.vlm import VLMClient


@dataclass
class ContentContext:
    original: Image.Image
    vlm: Any
    ocr: Any
    use_ocr_prompt: bool
    all_entities: list[dict[str, Any]]


class ContentAgent:
    name = "ContentAgent"
    handled_types: set[str] = set()

    def can_handle(self, entity: dict[str, Any]) -> bool:
        return str(entity.get("type") or "") in self.handled_types

    def run(self, entity: dict[str, Any], ctx: ContentContext) -> None:
        legacy_handlers._dispatch(
            entity,
            ctx.original,
            ctx.vlm,
            ctx.all_entities,
            ctx.ocr,
            ctx.use_ocr_prompt,
        )


class TextContentAgent(ContentAgent):
    name = "TextContentAgent"
    handled_types = {"text", "formula"}


class ShapeContentAgent(ContentAgent):
    name = "ShapeContentAgent"
    handled_types = {"shape", "container", "rounded_rect"}


class ConnectorContentAgent(ContentAgent):
    name = "ConnectorContentAgent"
    handled_types = {"arrow"}


class ChartContentAgent(ContentAgent):
    name = "ChartContentAgent"
    handled_types = {"chart"}


class IconContentAgent(ContentAgent):
    name = "IconContentAgent"
    handled_types = {"icon"}


class SurfaceContentAgent(ContentAgent):
    name = "SurfaceContentAgent"
    handled_types = {"surface", "dotcloud"}


def default_agents() -> list[ContentAgent]:
    return [
        TextContentAgent(),
        ShapeContentAgent(),
        ConnectorContentAgent(),
        ChartContentAgent(),
        IconContentAgent(),
        SurfaceContentAgent(),
    ]


def run(
    entities: list[dict[str, Any]],
    original: Image.Image,
    vlm: Any,
    out_dir: Path,
    *,
    max_workers: int = 3,
    log: Callable[[str], None] = print,
    agents: list[ContentAgent] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ocr_model = os.environ.get("I2E_OCR_MODEL")
    use_ocr_prompt = bool(ocr_model)
    ocr = VLMClient(model=ocr_model) if ocr_model else vlm
    agent_list = agents or default_agents()
    tasks = _build_tasks(entities, agent_list)
    ctx = ContentContext(
        original=original,
        vlm=vlm,
        ocr=ocr,
        use_ocr_prompt=use_ocr_prompt,
        all_entities=entities,
    )

    if os.environ.get("I2E_BATCH_OCR", "0") == "1":
        legacy_handlers._batch_ocr_texts(
            entities,
            original,
            ocr or vlm,
            use_ocr_prompt,
            log=lambda msg: log(f"[Content:BatchTextOCR] {msg}"),
        )

    def work(task: dict[str, Any]) -> dict[str, Any]:
        idx = int(task["entity_index"])
        entity = entities[idx]
        agent = task["_agent"]
        before_type = entity.get("type")
        try:
            agent.run(entity, ctx)
            task["status"] = "ok"
            task["after_type"] = entity.get("type")
            task["changed_type"] = before_type != entity.get("type")
            task["content"] = entity.get("content")
        except Exception as exc:
            entity["content"] = None
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            log(f"[Content:{agent.name}] {entity.get('id')} "
                f"({before_type}) failed: {type(exc).__name__}: {exc}")
        task.pop("_agent", None)
        return task

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        completed = list(pool.map(work, tasks))

    summary = _summarize(completed)
    report = {
        "version": "content-orchestrator-v1",
        "tasks": completed,
        "summary": summary,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "content_tasks.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log(f"[ContentOrchestrator] tasks={summary['tasks']} "
        f"ok={summary['ok']} failed={summary['failed']} "
        f"by_agent={summary['by_agent']}")
    return entities, report


def _build_tasks(entities: list[dict[str, Any]],
                 agents: list[ContentAgent]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for idx, entity in enumerate(entities):
        agent = _agent_for_entity(entity, agents)
        if agent is None:
            continue
        tasks.append({
            "id": f"content_{idx:04d}_{entity.get('id') or idx}",
            "entity_index": idx,
            "element_id": entity.get("id"),
            "type": entity.get("type"),
            "agent": agent.name,
            "capability": _capability_for_type(str(entity.get("type") or "")),
            "status": "planned",
            "_agent": agent,
        })
    return tasks


def _agent_for_entity(entity: dict[str, Any],
                      agents: list[ContentAgent]) -> ContentAgent | None:
    for agent in agents:
        if agent.can_handle(entity):
            return agent
    return None


def _capability_for_type(typ: str) -> str:
    if typ in {"text", "formula"}:
        return "read_text_or_formula"
    if typ in {"shape", "container", "rounded_rect"}:
        return "sample_native_shape_style"
    if typ == "arrow":
        return "trace_connector_geometry"
    if typ == "chart":
        return "extract_chart_spec"
    if typ == "icon":
        return "recognize_icon"
    if typ in {"surface", "dotcloud"}:
        return "vectorize_surface"
    return "unknown"


def _summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    by_agent: dict[str, int] = {}
    by_type: dict[str, int] = {}
    failed = 0
    for task in tasks:
        by_agent[task["agent"]] = by_agent.get(task["agent"], 0) + 1
        by_type[str(task.get("type") or "")] = by_type.get(str(task.get("type") or ""), 0) + 1
        if task.get("status") != "ok":
            failed += 1
    return {
        "tasks": len(tasks),
        "ok": len(tasks) - failed,
        "failed": failed,
        "by_agent": by_agent,
        "by_type": by_type,
    }
