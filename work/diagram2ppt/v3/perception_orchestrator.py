"""Planner-controlled perception orchestration.

This replaces the hard-coded "decompose does everything" entry path with a
small agent blackboard: each perception agent submits evidence, then the
planner fuses the evidence into initial entities.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from work.diagram2ppt.v2 import decompose as legacy_decompose

from . import evidence as E


@dataclass
class PerceptionContext:
    image_path: str
    original: Image.Image
    vlm: Any
    log: Callable[[str], None]


class PerceptionAgent:
    name = "PerceptionAgent"
    capability = "generic"

    def run(self, ctx: PerceptionContext) -> list[dict[str, Any]]:
        raise NotImplementedError


class CVGeometryAgent(PerceptionAgent):
    name = "CVGeometryAgent"
    capability = "geometry"

    def run(self, ctx: PerceptionContext) -> list[dict[str, Any]]:
        seeds = legacy_decompose._cv_structure_seeds(ctx.original, log=ctx.log)
        return [
            E.make(
                source="cv_geometry",
                typ=s["type"],
                bbox=s["bbox"],
                confidence=0.82,
                payload={"agent": self.name},
            )
            for s in seeds
        ]


class VLMStructureAgent(PerceptionAgent):
    name = "VLMStructureAgent"
    capability = "structure_semantics"

    def run(self, ctx: PerceptionContext) -> list[dict[str, Any]]:
        max_edge = int(os.environ.get("I2E_DECOMPOSE_MAX_EDGE", "1280"))
        max_tokens = int(os.environ.get("I2E_STRUCTURE_MAX_TOKENS", "1200"))
        attempts = int(os.environ.get("I2E_STRUCTURE_ATTEMPTS", "2"))
        w, h = ctx.original.size
        f = max_edge / max(w, h) if max(w, h) > max_edge else 1.0
        seen = (max(1, round(w * f)), max(1, round(h * f)))
        out: list[dict[str, Any]] = []
        for attempt in range(attempts):
            raw = ctx.vlm.chat(
                legacy_decompose.STRUCTURE_PROMPT,
                ctx.original,
                max_tokens=max_tokens,
                max_edge=max_edge,
                frequency_penalty=0.5,
            )
            entities = legacy_decompose._parse_entities(raw, w, h, seen)
            ctx.log(f"[Perception:{self.name}] pass {attempt + 1}: "
                    f"{len(entities)} boxes")
            out.extend(
                E.make(
                    source="vlm_structure",
                    typ=e["type"],
                    bbox=e["bbox"],
                    confidence=0.68,
                    payload={"agent": self.name, "attempt": attempt + 1},
                )
                for e in entities
                if e.get("type") not in {"text", "formula"}
            )
            if len(entities) >= 24:
                break
        return out


class VLMTextAgent(PerceptionAgent):
    name = "VLMTextAgent"
    capability = "text_detection"

    def run(self, ctx: PerceptionContext) -> list[dict[str, Any]]:
        max_edge = int(os.environ.get("I2E_DECOMPOSE_MAX_EDGE", "1280"))
        max_tokens = int(os.environ.get("I2E_TEXT_DETECT_MAX_TOKENS", "2600"))
        attempts = int(os.environ.get("I2E_TEXT_DETECT_ATTEMPTS", "1"))
        w, h = ctx.original.size
        f = max_edge / max(w, h) if max(w, h) > max_edge else 1.0
        seen = (max(1, round(w * f)), max(1, round(h * f)))
        out: list[dict[str, Any]] = []
        for attempt in range(attempts):
            raw = ctx.vlm.chat(
                legacy_decompose.TEXT_PROMPT,
                ctx.original,
                max_tokens=max_tokens,
                max_edge=max_edge,
                frequency_penalty=0.5,
            )
            entities = legacy_decompose._parse_entities(raw, w, h, seen)
            ctx.log(f"[Perception:{self.name}] pass {attempt + 1}: "
                    f"{len(entities)} boxes")
            out.extend(
                E.make(
                    source="vlm_text",
                    typ=e["type"],
                    bbox=e["bbox"],
                    confidence=0.70,
                    text=e.get("text"),
                    payload={"agent": self.name, "attempt": attempt + 1},
                )
                for e in entities
                if e.get("type") in {"text", "formula"}
            )
            if len(entities) >= 12:
                break
        return out


class OCRTextAgent(PerceptionAgent):
    name = "OCRTextAgent"
    capability = "ocr_geometry"

    def run(self, ctx: PerceptionContext) -> list[dict[str, Any]]:
        try:
            lines = legacy_decompose._get_tesseract_lines(
                ctx.image_path,
                conf_threshold=float(os.environ.get("I2E_TESSERACT_CONF", "25.0")),
            )
        except Exception as exc:
            ctx.log(f"[Perception:{self.name}] unavailable: {exc}")
            return []
        out = []
        for ln in lines:
            b = ln["bbox"]
            bbox = [b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]]
            text = str(ln.get("content") or "").strip()
            if not text or legacy_decompose._is_bad_ocr_line(text, bbox):
                continue
            typ = "formula" if legacy_decompose._MATH_CHARS.search(text) else "text"
            out.append(E.make(
                source="ocr_text",
                typ=typ,
                bbox=bbox,
                confidence=float(ln.get("conf") or 0) / 100.0,
                text=text,
                payload={"agent": self.name},
            ))
        ctx.log(f"[Perception:{self.name}] {len(out)} usable OCR lines")
        return out


def default_agents() -> list[PerceptionAgent]:
    agents: list[PerceptionAgent] = [CVGeometryAgent()]
    if os.environ.get("I2E_DISABLE_VLM_PERCEPTION", "0") != "1":
        agents.extend([VLMStructureAgent(), VLMTextAgent()])
    if os.environ.get("I2E_DISABLE_OCR_PERCEPTION", "0") != "1":
        agents.append(OCRTextAgent())
    return agents


def run(
    image_path: str,
    original: Image.Image,
    vlm: Any,
    out_dir: Path,
    log: Callable[[str], None] = print,
    agents: list[PerceptionAgent] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run perception agents and return fused entities plus a report."""
    ctx = PerceptionContext(
        image_path=image_path,
        original=original,
        vlm=vlm,
        log=log,
    )
    all_evidence: list[dict[str, Any]] = []
    tasks = []
    for agent in agents or default_agents():
        task = {
            "agent": agent.name,
            "capability": agent.capability,
            "status": "running",
            "evidence": 0,
        }
        try:
            ev = agent.run(ctx)
            all_evidence.extend(ev)
            task["status"] = "ok"
            task["evidence"] = len(ev)
        except Exception as exc:
            task["status"] = "failed"
            task["error"] = f"{type(exc).__name__}: {exc}"
            log(f"[Perception:{agent.name}] failed: {type(exc).__name__}: {exc}")
        tasks.append(task)

    w, h = original.size
    entities = E.fuse_to_entities(all_evidence, w, h, log=log)
    report = {
        "version": "perception-blackboard-v1",
        "image": {"path": image_path, "width": w, "height": h},
        "tasks": tasks,
        "evidence": all_evidence,
        "entities": entities,
        "summary": {
            "evidence": len(all_evidence),
            "entities": len(entities),
            "sources": _count_by(all_evidence, "source"),
            "types": _count_by(entities, "type"),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "perception_blackboard.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return entities, report


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        val = str(item.get(key) or "")
        out[val] = out.get(val, 0) + 1
    return out
