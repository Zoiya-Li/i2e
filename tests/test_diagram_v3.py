"""Regression tests for diagram2ppt v3 foundation (Planner + IR + builder)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from work.diagram2ppt.v3 import builder, ir, migrate, planner, renderer, verifier


@pytest.fixture
def sample_v2_entities() -> list[dict]:
    return [
        {
            "id": "e1",
            "type": "text",
            "bbox": [100, 100, 300, 150],
            "text": "Hello",
            "font_size": 24,
            "text_color": "#000000",
            "z": 1,
        },
        {
            "id": "e2",
            "type": "shape",
            "bbox": [90, 90, 310, 160],
            "fill": "#FFFFFF",
            "border_color": "#333333",
            "z": 0,
        },
    ]


@pytest.fixture
def tmp_v3_out(tmp_path: Path) -> Path:
    out = tmp_path / "v3_out"
    out.mkdir()
    return out


def test_ir_creation() -> None:
    blackboard = ir.new_ir("img.png", 1920, 1080)
    assert blackboard["version"] == "d2p-3"
    assert blackboard["canvas"]["width_px"] == 1920
    assert blackboard["canvas"]["slide_height_in"] > 0
    assert blackboard["elements"] == []


def test_ir_element_factory() -> None:
    el = ir.element(
        id="t1",
        type="text",
        bbox=[10, 20, 100, 40],
        provenance=ir.provenance("TestAgent", "create"),
        text="test",
    )
    assert el["type"] == "text"
    assert el["provenance"]["agent"] == "TestAgent"
    assert el["status"] == "native"


def test_ir_rejects_non_native_element() -> None:
    with pytest.raises(ValueError, match="non-native"):
        ir.element("x1", "raster_crop", [0, 0, 10, 10], ir.provenance("x", "x"))


def test_migrate_from_v2_entities(sample_v2_entities: list[dict], tmp_v3_out: Path) -> None:
    out = migrate.from_v2_entities(sample_v2_entities, "img.png", 400, 300)
    assert out["version"] == "d2p-3"
    assert len(out["elements"]) == 2
    types = {e["type"] for e in out["elements"]}
    assert types == {"text", "rect"}
    assert all(e["provenance"]["agent"] == "MigrateAgent" for e in out["elements"])


def test_migrate_rejects_raster_crop(tmp_v3_out: Path) -> None:
    entities = [
        {"id": "r1", "type": "raster_crop", "bbox": [0, 0, 100, 100]},
    ]
    out = migrate.from_v2_entities(entities, "img.png", 100, 100)
    assert len(out["elements"]) == 0
    assert len(out["defects"]) == 1
    assert out["defects"][0]["type"] == "unsupported_element"


def test_builder_rejects_raster_crop() -> None:
    blackboard = ir.new_ir("img.png", 400, 300)
    blackboard["elements"].append(ir.element(
        "r1", "text", [0, 0, 100, 100],
        ir.provenance("TestAgent", "create"),
        text="ok",
    ))
    # sneaky non-native type should be caught
    blackboard["elements"][0]["type"] = "raster_crop"
    with pytest.raises(builder.BuildBlockedError):
        builder.build_pptx(blackboard, "/tmp/should_not_create.pptx")


def test_builder_produces_native_pptx(sample_v2_entities: list[dict],
                                     tmp_v3_out: Path) -> None:
    blackboard = migrate.from_v2_entities(sample_v2_entities, "img.png", 400, 300)
    pptx_path = tmp_v3_out / "test.pptx"
    stats = builder.build_pptx(blackboard, str(pptx_path))
    assert pptx_path.exists()
    assert stats["pictures"] == 0
    assert stats["native"] is True


def test_renderer_availability_detected() -> None:
    # macOS with osascript is expected in the dev environment.
    assert isinstance(renderer.is_available(), bool)


def test_planner_plan_creates_ir() -> None:
    # Use a tiny synthetic image to avoid heavy VLM calls.
    from PIL import Image
    img_path = Path("/tmp/v3_test_plan.png")
    Image.new("RGB", (200, 100), "white").save(img_path)
    p = planner.Planner(str(img_path), "/tmp/v3_test_out")
    # Replace decompose to avoid real VLM.
    fake_entities = [
        {"id": "t1", "type": "text", "bbox": [10, 10, 100, 40], "text": "Hi"},
    ]
    out = ir.new_ir(str(img_path), 200, 100)
    out["elements"].append(ir.element("t1", "text", [10, 10, 100, 40],
                                       ir.provenance("TestAgent", "setup"),
                                       text="Hi"))
    p.ir = out
    p._push_snapshot()
    assert p.ir["version"] == "d2p-3"


def test_to_v2_compatible() -> None:
    blackboard = ir.new_ir("img.png", 400, 300)
    blackboard["elements"].append(ir.element(
        "t1", "text", [0, 0, 100, 100],
        ir.provenance("TestAgent", "create"),
        text="hello",
    ))
    v2 = migrate.to_v2_compatible(blackboard)
    assert v2["version"] == "d2p-2"
    assert v2["image"]["width"] == 400
    assert len(v2["elements"]) == 1


def test_siliconflow_provider_config() -> None:
    from work.diagram2ppt.v3.providers.siliconflow import SiliconFlowProvider
    p = SiliconFlowProvider(api_key="test", model="deepseek-ai/DeepSeek-OCR")
    assert p.base_url == "https://api.siliconflow.cn/v1"
    assert p.model == "deepseek-ai/DeepSeek-OCR"


def test_openai_compat_provider_extract_json() -> None:
    from work.diagram2ppt.v3.providers.openai_compat import OpenAICompatProvider
    p = OpenAICompatProvider()
    assert p._extract_json('{"a": 1}')["a"] == 1
    assert p._extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert p._extract_json('some text {"a": 1} more')["a"] == 1


def test_pix2tex_model_loads_and_recognizes_formula() -> None:
    from PIL import Image
    from work.diagram2ppt.v3.models import get_local_model

    # Reuse the existing test crop if available; otherwise create a tiny one.
    crop_path = Path("/tmp/formula_test_crop3.png")
    if not crop_path.exists():
        source = Path("work/diagram2ppt/v2_out/fw_1.5x.png")
        if source.exists():
            im = Image.open(source)
            w, h = im.size
            crop = im.crop((int(w * 0.17), int(h * 0.09),
                            int(w * 0.32), int(h * 0.17)))
            crop.save(crop_path)

    if not crop_path.exists():
        pytest.skip("no formula test image available")

    model = get_local_model("pix2tex")
    latex = model(Image.open(crop_path).convert("RGB"))
    assert latex and isinstance(latex, str)
    assert "=" in latex


def test_formula_agent_repairs_formula_element() -> None:
    from PIL import Image
    from work.diagram2ppt.v3.agents.formula import FormulaAgent

    img_path = Path("work/diagram2ppt/v2_out/fw_1.5x.png")
    if not img_path.exists():
        pytest.skip("fw_1.5x.png not available")

    im = Image.open(img_path).convert("RGB")
    blackboard = ir.new_ir(str(img_path), im.width, im.height)
    bbox = [im.width * 0.16, im.height * 0.07,
            im.width * 0.33, im.height * 0.19]
    el = ir.element(
        id="f1",
        type="formula",
        bbox=bbox,
        provenance=ir.provenance("MigrateAgent", "from_v2_entities"),
        text="Bll",
    )
    blackboard["elements"].append(el)

    agent = FormulaAgent()
    changed = agent.run(blackboard, im)
    assert changed == ["f1"]
    assert el.get("text") != "Bll"
    assert el.get("ext", {}).get("latex") == el.get("text")
    assert any(r["agent"] == "FormulaAgent" for r in el["repair_history"])


def test_clean_formula_latex_strips_boxed_wrapper() -> None:
    from work.diagram2ppt.v3.models import _clean_formula_latex

    raw = r"\boxed { A = \frac{1}{2} }"
    cleaned = _clean_formula_latex(raw)
    assert cleaned == r"A = \frac{1}{2}"


def test_connector_agent_repairs_bad_connector() -> None:
    from PIL import Image, ImageDraw
    from work.diagram2ppt.v3.agents.connector import ConnectorAgent
    from work.diagram2ppt.v3 import ir as IR

    img = Image.new("RGB", (400, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 60, 100, 140], outline="black", width=2)
    draw.rectangle([280, 60, 360, 140], outline="black", width=2)
    draw.line([(100, 100), (280, 100)], fill="blue", width=3)

    blackboard = ir.new_ir("img.png", 400, 200)
    blackboard["elements"].append(ir.element(
        "s1", "rect", [20, 60, 100, 140],
        ir.provenance("Test", "create")))
    blackboard["elements"].append(ir.element(
        "s2", "rect", [280, 60, 360, 140],
        ir.provenance("Test", "create")))
    blackboard["elements"].append(ir.element(
        "a1", "arrow", [90, 80, 290, 120],
        ir.provenance("Test", "create"),
        points=[0.0, 0.0, 10.0, 10.0],
        color="#333333", thickness=2))

    agent = ConnectorAgent()
    changed = agent.run(blackboard, img)
    assert changed == ["a1"]
    el = ir.get_element(blackboard, "a1")
    assert el is not None
    assert el.get("from_id") == "s1"
    assert el.get("to_id") == "s2"
    pts = el.get("points", [])
    assert pts[0] > 90 and pts[2] < 290


def test_connector_agent_removes_inkless_connector() -> None:
    from PIL import Image
    from work.diagram2ppt.v3.agents.connector import ConnectorAgent
    from work.diagram2ppt.v3 import ir as IR

    img = Image.new("RGB", (200, 100), "white")
    blackboard = ir.new_ir("img.png", 200, 100)
    blackboard["elements"].append(ir.element(
        "a1", "arrow", [0, 0, 10, 10],
        ir.provenance("Test", "create"),
        points=[0.0, 0.0, 5.0, 5.0]))
    agent = ConnectorAgent()
    changed = agent.run(blackboard, img)
    assert changed == []
    assert ir.get_element(blackboard, "a1") is None


def test_migrate_preserves_icon_payload() -> None:
    entities = [
        {"id": "i1", "type": "icon", "bbox": [10, 10, 50, 50],
         "icon": {"kind": "gear", "color": "#6b7a8d", "glyph": "⚙"}},
    ]
    out = migrate.from_v2_entities(entities, "img.png", 200, 200)
    el = out["elements"][0]
    assert el["type"] == "icon"
    assert el.get("icon") == {"kind": "gear", "color": "#6b7a8d", "glyph": "⚙"}
    assert el.get("ext", {}).get("icon") == el["icon"]


def test_migrate_preserves_chart_payload() -> None:
    entities = [
        {"id": "c1", "type": "chart", "bbox": [10, 10, 100, 100],
         "chart": {"kind": "bar", "categories": ["A", "B"],
                   "series": [{"name": "s1", "color": "#4472c4",
                               "values": [1, 2]}]}},
    ]
    out = migrate.from_v2_entities(entities, "img.png", 200, 200)
    el = out["elements"][0]
    assert el["type"] == "chart"
    assert el.get("chart", {}).get("kind") == "bar"
    assert el.get("ext", {}).get("chart") == el["chart"]


def test_icon_agent_reclassifies_icon() -> None:
    from PIL import Image
    from work.diagram2ppt.v3.agents.icon import IconAgent
    from work.diagram2ppt.v3 import ir as IR

    class FakeProvider:
        name = "fake"
        def ask(self, image, prompt, temperature=0.0, max_tokens=4096):
            return '{"kind": "database", "color": "#4472c4", "glyph": "🗄"}'

    img = Image.new("RGB", (100, 100), "white")
    blackboard = ir.new_ir("img.png", 100, 100)
    blackboard["elements"].append(ir.element(
        "i1", "icon", [10, 10, 50, 50],
        ir.provenance("Test", "create"),
        icon={"kind": "other", "color": "#555555", "glyph": "◆"}))

    agent = IconAgent()
    agent.provider = FakeProvider()
    changed = agent.run(blackboard, img)
    assert changed == ["i1"]
    el = ir.get_element(blackboard, "i1")
    assert el["icon"]["kind"] == "database"
    assert el["ext"]["icon"]["kind"] == "database"


def test_chart_agent_repairs_chart_spec() -> None:
    from PIL import Image
    from work.diagram2ppt.v3.agents.chart import ChartAgent
    from work.diagram2ppt.v3 import ir as IR

    class FakeProvider:
        name = "fake"
        def ask(self, image, prompt, temperature=0.0, max_tokens=4096):
            return (
                '{"type": "bar", "categories": ["A", "B"], '
                '"series": [{"name": "s1", "color": "#4472c4", "values": [1, 2]}]}'
            )

    img = Image.new("RGB", (200, 200), "white")
    blackboard = ir.new_ir("img.png", 200, 200)
    blackboard["elements"].append(ir.element(
        "c1", "chart", [10, 10, 100, 100],
        ir.provenance("Test", "create")))

    agent = ChartAgent()
    agent.provider = FakeProvider()
    changed = agent.run(blackboard, img)
    assert changed == ["c1"]
    el = ir.get_element(blackboard, "c1")
    assert el["chart"]["kind"] == "bar"
    assert el["chart"]["categories"] == ["A", "B"]
    assert el["ext"]["chart"]["kind"] == "bar"


def test_provider_registry_uses_per_capability_model(monkeypatch) -> None:
    """Chart and icon can use different models than the default VLM."""
    import os
    from work.diagram2ppt.v3.providers import get_provider

    monkeypatch.setenv("I2E_V3_VLM_PROVIDER", "siliconflow")
    monkeypatch.setenv("I2E_VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("I2E_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setenv("I2E_CHART_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    monkeypatch.setenv("I2E_ICON_MODEL", "Qwen/Qwen2.5-VL-32B-Instruct")

    vlm = get_provider("vlm")
    chart = get_provider("chart")
    icon = get_provider("icon")

    assert vlm.model == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert chart.model == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert icon.model == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert chart.api_key == "vlm-key"
    assert icon.api_key == "vlm-key"


def test_provider_registry_falls_back_to_vlm_for_chart(monkeypatch) -> None:
    from work.diagram2ppt.v3.providers import get_provider

    monkeypatch.setenv("I2E_V3_VLM_PROVIDER", "siliconflow")
    monkeypatch.setenv("I2E_VLM_API_KEY", "vlm-key")
    monkeypatch.setenv("I2E_VLM_MODEL", "Qwen/Qwen2.5-VL-72B-Instruct")
    monkeypatch.delenv("I2E_CHART_MODEL", raising=False)

    chart = get_provider("chart")
    assert chart.model == "Qwen/Qwen2.5-VL-72B-Instruct"



def test_semantic_validation_rejects_formula_collapsing_to_zero() -> None:
    """A formula patch that collapses content to '0' must be flagged as degraded."""
    from work.diagram2ppt.v3 import semantic

    old_el = ir.element(
        id="e47",
        type="formula",
        bbox=[100, 100, 200, 140],
        provenance=ir.provenance("MigrateAgent", "create"),
        text=r"\gamma = \nabla \tau ( x )",
        ext={"latex": r"\gamma = \nabla \tau ( x )"},
    )
    new_el = ir.element(
        id="e47",
        type="formula",
        bbox=[100, 100, 200, 140],
        provenance=ir.provenance("FormulaAgent", "repair"),
        text="0",
        ext={"latex": "0"},
    )
    ir_before = ir.new_ir("img.png", 400, 300)
    ir_before["elements"].append(old_el)
    ir_after = ir.new_ir("img.png", 400, 300)
    ir_after["elements"].append(new_el)

    patch = {
        "patch_id": "p1",
        "round": 1,
        "agent": "FormulaAgent",
        "changed": ["e47"],
        "expected_fixes": ["defect_residual_e47"],
        "metrics_before": {},
        "metrics_after": {},
    }
    result = semantic.validate_patch(ir_after, patch, ir_before)
    assert result["ok"] is False
    assert "e47" in result["degraded"]


def test_semantic_validation_accepts_text_correction() -> None:
    """A text patch that fixes a typo without catastrophic shortening is OK."""
    from work.diagram2ppt.v3 import semantic

    old_el = ir.element(
        id="t1",
        type="text",
        bbox=[100, 100, 300, 140],
        provenance=ir.provenance("MigrateAgent", "create"),
        text="Helo world",
    )
    new_el = ir.element(
        id="t1",
        type="text",
        bbox=[100, 100, 300, 140],
        provenance=ir.provenance("TextAgent", "repair"),
        text="Hello world",
    )
    ir_before = ir.new_ir("img.png", 400, 300)
    ir_before["elements"].append(old_el)
    ir_after = ir.new_ir("img.png", 400, 300)
    ir_after["elements"].append(new_el)
    ir_after["defects"] = []  # target defect resolved

    patch = {
        "patch_id": "p1",
        "round": 1,
        "agent": "TextAgent",
        "changed": ["t1"],
        "expected_fixes": ["defect_residual_t1"],
        "metrics_before": {},
        "metrics_after": {},
    }
    result = semantic.validate_patch(ir_after, patch, ir_before)
    assert result["ok"] is True
    assert result["degraded"] == []


def test_semantic_validation_requires_content_agent_to_fix_target_defect() -> None:
    """A TextAgent patch whose target residual defect is still present is rejected."""
    from work.diagram2ppt.v3 import semantic

    old_el = ir.element(
        id="t1",
        type="text",
        bbox=[100, 100, 300, 140],
        provenance=ir.provenance("MigrateAgent", "create"),
        text="old",
    )
    new_el = ir.element(
        id="t1",
        type="text",
        bbox=[100, 100, 300, 140],
        provenance=ir.provenance("TextAgent", "repair"),
        text="new",
    )
    ir_before = ir.new_ir("img.png", 400, 300)
    ir_before["elements"].append(old_el)
    ir_after = ir.new_ir("img.png", 400, 300)
    ir_after["elements"].append(new_el)
    ir_after["defects"] = [
        {
            "id": "defect_residual_t1",
            "type": "high_residual",
            "element_id": "t1",
            "severity": 0.8,
        }
    ]

    patch = {
        "patch_id": "p1",
        "round": 1,
        "agent": "TextAgent",
        "changed": ["t1"],
        "expected_fixes": ["defect_residual_t1"],
        "metrics_before": {},
        "metrics_after": {},
    }
    result = semantic.validate_patch(ir_after, patch, ir_before)
    assert result["ok"] is False
    assert "defect_residual_t1" in result["target_defects_remaining"]


def test_style_agent_infers_shape_fill_and_border() -> None:
    """StyleAgent should sample a synthetic rectangle and set fill/border colors."""
    from PIL import Image, ImageDraw
    from work.diagram2ppt.v3.agents.style import StyleAgent

    img = Image.new("RGB", (200, 200), (255, 255, 255))
    # Draw a red rectangle with blue 3px border.
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 160, 120], fill=(255, 0, 0), outline=(0, 0, 255), width=3)

    blackboard = ir.new_ir("img.png", 200, 200)
    el = ir.element(
        id="s1",
        type="rect",
        bbox=[40, 40, 160, 120],
        provenance=ir.provenance("MigrateAgent", "create"),
        fill="#ffffff",
        border_color="#000000",
    )
    blackboard["elements"].append(el)

    agent = StyleAgent()
    changed = agent.run(blackboard, img)
    assert changed == ["s1"]

    updated = next(e for e in blackboard["elements"] if e["id"] == "s1")
    assert updated["fill"].lower() != "#ffffff"
    assert updated["border_color"].lower() != "#000000"


def test_style_agent_routes_through_high_residual_defect() -> None:
    """A high_residual defect on a shape should be routed to StyleAgent."""
    from work.diagram2ppt.v3 import verifier

    assert verifier._agent_for_type("rect", "high_residual") == "StyleAgent"
    assert verifier._agent_for_type("rounded_rect", "high_residual") == "StyleAgent"
    assert verifier._agent_for_type("rect", "missing_element") == "ShapeAgent"

    # Text with existing content is a style mismatch; empty text needs OCR.
    assert verifier._agent_for_type("text", "high_residual",
                                     {"text": "hello"}) == "StyleAgent"
    assert verifier._agent_for_type("text", "high_residual",
                                     {"text": ""}) == "TextAgent"
