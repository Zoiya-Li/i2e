import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw

from work.diagram2ppt.v2.decompose import _parse_entities
from work.diagram2ppt.v2.vlm import VLMClient
from work.diagram2ppt.v3 import content_orchestrator
from work.diagram2ppt.v3 import method_registry
from work.diagram2ppt.v3 import perception_orchestrator
from work.diagram2ppt.v3 import proposal_orchestrator
from work.diagram2ppt.v3 import regression_suite
from work.diagram2ppt.v3 import renderer
from work.diagram2ppt.v3 import strategy
from work.diagram2ppt.v3 import representation_plan
from work.diagram2ppt.v3 import typography
from work.diagram2ppt.v3 import visual_review
from work.diagram2ppt.v3.audit_agent_system import AuditAgentSystem
from work.diagram2ppt.v3.agents.pipeline_context import PipelineContextAgent
from work.diagram2ppt.v3.agents.chart import ChartAgent
from work.diagram2ppt.v3.agents.procedural_surface import ProceduralSurfaceAgent
from work.diagram2ppt.v3.planner import _candidate_rank
from work.diagram2ppt.v3.providers.openai_compat import OpenAICompatProvider


class PerceptionContractsTest(unittest.TestCase):
    def test_parse_qwen_bbox_variants(self):
        cases = [
            (
                '{"entities":[{"type":"text","bbox":[0,0,0.1,0.1],'
                '"bbox_text":"Title"}]}',
                "text",
                [0, 0, 126, 70],
            ),
            (
                '```json\n[{"bbox_2d":[0,0,1000,500],"type":"container"}]\n```',
                "container",
                [0, 0, 1258, 348],
            ),
            (
                '```json\n[{"bbox_2d":[0,0,1000,1000],"type":"chart"}]\n```',
                "chart",
                [0, 0, 1258, 697],
            ),
        ]
        for raw, typ, bbox in cases:
            with self.subTest(typ=typ):
                parsed = _parse_entities(raw, 1258, 697, (1258, 697))
                self.assertEqual(len(parsed), 1)
                self.assertEqual(parsed[0]["type"], typ)
                self.assertEqual(parsed[0]["bbox"], bbox)

    def test_v2_client_routes_text_only_model_to_vision_model(self):
        old = os.environ.get("I2E_VISION_MODEL")
        os.environ["I2E_VISION_MODEL"] = "Qwen/Qwen3.6-35B-A3B"
        try:
            client = VLMClient.__new__(VLMClient)
            client.base_url = "https://api.siliconflow.cn/v1"
            payload = {"model": "Qwen/Qwen3.5-397B-A17B"}
            client._apply_provider_options(payload)
            self.assertEqual(payload["model"], "Qwen/Qwen3.6-35B-A3B")
            self.assertNotIn("enable_thinking", payload)
            self.assertNotIn("thinking", payload)
        finally:
            if old is None:
                os.environ.pop("I2E_VISION_MODEL", None)
            else:
                os.environ["I2E_VISION_MODEL"] = old

    def test_v3_provider_routes_text_only_model_to_vision_model(self):
        old = os.environ.get("I2E_VISION_MODEL")
        os.environ["I2E_VISION_MODEL"] = "Qwen/Qwen3.6-35B-A3B"
        try:
            provider = OpenAICompatProvider(
                base_url="https://api.siliconflow.cn/v1",
                api_key="unused",
                model="Qwen/Qwen3.5-397B-A17B",
                timeout=12,
            )
            self.assertEqual(provider._model_for_image(), "Qwen/Qwen3.6-35B-A3B")
            self.assertEqual(provider._provider_extra_body(), {})
            self.assertEqual(provider.timeout, 12)
        finally:
            if old is None:
                os.environ.pop("I2E_VISION_MODEL", None)
            else:
                os.environ["I2E_VISION_MODEL"] = old

    def test_perception_orchestrator_fuses_agent_evidence_without_vlm(self):
        old_disable = os.environ.get("I2E_DISABLE_VLM_PERCEPTION")
        os.environ["I2E_DISABLE_VLM_PERCEPTION"] = "1"
        try:
            with TemporaryDirectory() as td:
                p = Path(td) / "simple.png"
                im = Image.new("RGB", (320, 180), "white")
                draw = ImageDraw.Draw(im)
                draw.rectangle([20, 20, 150, 120], outline="black", width=2)
                draw.rectangle([175, 35, 300, 130], outline="black", width=2)
                im.save(p)

                entities, report = perception_orchestrator.run(
                    str(p),
                    im,
                    vlm=None,
                    out_dir=Path(td),
                    log=lambda *_: None,
                    agents=[perception_orchestrator.CVGeometryAgent()],
                )
                self.assertGreaterEqual(len(report["evidence"]), 2)
                self.assertGreaterEqual(len(entities), 2)
                self.assertTrue((Path(td) / "perception_blackboard.json").exists())
        finally:
            if old_disable is None:
                os.environ.pop("I2E_DISABLE_VLM_PERCEPTION", None)
            else:
                os.environ["I2E_DISABLE_VLM_PERCEPTION"] = old_disable

    def test_candidate_rank_gates_visual_quality_before_representation(self):
        stable = {
            "visual_delta": 0.49,
            "coverage_explained": 1.0,
            "critical_defect_count": 20,
            "defect_count": 60,
            "text_accuracy": 0.5,
        }
        richer_but_worse = {
            "visual_delta": 0.57,
            "coverage_explained": 0.96,
            "critical_defect_count": 45,
            "defect_count": 98,
            "text_accuracy": 0.43,
        }
        blank_low_delta = {
            "visual_delta": 0.31,
            "coverage_explained": 0.05,
            "critical_defect_count": 4,
            "defect_count": 13,
            "text_accuracy": 0.5,
        }
        weak_rep = {"average": 0.14, "minimum": 0.0}
        strong_rep = {"average": 0.55, "minimum": 0.0}
        no_rep = {"average": 0.0, "minimum": 0.0}

        self.assertLess(
            _candidate_rank(stable, weak_rep),
            _candidate_rank(richer_but_worse, strong_rep),
        )
        self.assertLess(
            _candidate_rank(stable, weak_rep),
            _candidate_rank(blank_low_delta, no_rep),
        )

    def test_content_orchestrator_routes_shape_task_without_vlm(self):
        with TemporaryDirectory() as td:
            im = Image.new("RGB", (200, 120), "white")
            draw = ImageDraw.Draw(im)
            draw.rectangle([20, 20, 120, 80], fill="#d9ead3", outline="#333333", width=2)
            entities = [{
                "id": "shape_1",
                "type": "shape",
                "bbox": [20, 20, 120, 80],
                "content": None,
            }]
            out, report = content_orchestrator.run(
                entities,
                im,
                vlm=None,
                out_dir=Path(td),
                max_workers=1,
                log=lambda *_: None,
                agents=[content_orchestrator.ShapeContentAgent()],
            )
            self.assertEqual(report["summary"]["tasks"], 1)
            self.assertEqual(report["summary"]["ok"], 1)
            self.assertEqual(out[0]["content"], "done")
            self.assertTrue((Path(td) / "content_tasks.json").exists())

    def test_audit_agent_system_controls_top_level_loop(self):
        class DummyPlanner:
            def __init__(self, out_dir: Path):
                self.out_dir = out_dir
                self.image_path = out_dir / "source.png"
                self.original = Image.new("RGB", (120, 80), "white")
                self.max_rounds = 1
                self.ir = {"status": "new", "metrics": {}, "defects": []}
                self.agents = {}
                self._agent_attempts = {}

            def plan(self):
                self.ir = {
                    "status": "planned",
                    "metrics": {},
                    "defects": [],
                    "elements": [],
                }
                return self.ir

            def render_and_verify(self):
                self.ir["metrics"] = {
                    "visual_delta": 0.0,
                    "coverage_explained": 1.0,
                    "critical_defect_count": 0,
                }
                self.ir["defects"] = []
                return {
                    "passed": True,
                    "metrics": self.ir["metrics"],
                    "defects": [],
                }

            def _save_ir(self, name):
                (self.out_dir / name).write_text(
                    __import__("json").dumps(self.ir, indent=2))

        with TemporaryDirectory() as td:
            planner = DummyPlanner(Path(td))
            final = AuditAgentSystem(planner, log=lambda *_: None).run()
            self.assertEqual(final["status"], "accepted")
            self.assertEqual(
                final["audit_agent"]["version"],
                AuditAgentSystem.version,
            )
            self.assertTrue((Path(td) / "audit_trace.json").exists())
            self.assertTrue((Path(td) / "ir_final.json").exists())
            self.assertTrue((Path(td) / "audit_state_initial.json").exists())

    def test_proposal_scheduler_defers_defect_clusters_when_visual_tasks_exist(self):
        graph = {
            "tasks": [
                {"id": "visual_1", "kind": "visual_region_defect"},
                {"id": "region_1", "kind": "chart"},
                {"id": "cluster_1", "kind": "defect_cluster"},
                {"id": "visual_2", "kind": "visual_region_defect"},
            ]
        }
        scheduled = proposal_orchestrator._scheduled_tasks(graph, max_tasks=8)
        self.assertEqual([t["id"] for t in scheduled], ["visual_1", "visual_2"])

    def test_regional_transaction_excludes_out_of_region_side_effects(self):
        base = {
            "round": 0,
            "elements": [
                {"id": "inside_old", "type": "rect", "bbox": [10, 10, 90, 90]},
                {"id": "outside_text", "type": "text", "bbox": [210, 10, 290, 90], "text": "old"},
            ],
        }
        candidate = {
            "round": 0,
            "elements": [
                {"id": "inside_new", "type": "surface", "bbox": [12, 12, 88, 88]},
                {"id": "outside_text", "type": "text", "bbox": [210, 10, 290, 90], "text": "changed"},
                {"id": "outside_new", "type": "rect", "bbox": [220, 120, 290, 180]},
            ],
        }
        task = {
            "id": "surface_task",
            "kind": "visual_region_defect",
            "bbox": [0, 0, 100, 100],
            "locked_method": "procedural_surface",
        }
        merged, transaction = proposal_orchestrator._regional_transaction_ir(
            base,
            candidate,
            task,
            ["inside_old", "inside_new", "outside_text", "outside_new"],
        )
        by_id = {e["id"]: e for e in merged["elements"]}
        self.assertNotIn("inside_old", by_id)
        self.assertIn("inside_new", by_id)
        self.assertEqual(by_id["outside_text"]["text"], "old")
        self.assertNotIn("outside_new", by_id)
        self.assertEqual(transaction["mode"], "regional")
        self.assertIn("inside_new", transaction["committed"])
        self.assertIn("inside_old", transaction["removed"])
        self.assertIn("outside_text", transaction["ignored"])

    def test_visual_review_degraded_fallback_uses_strategy_regions(self):
        with TemporaryDirectory() as td:
            compare = Path(td) / "compare.png"
            Image.new("RGB", (400, 240), "white").save(compare)
            plan = {
                "regions": [
                    {
                        "id": "region_chart_0",
                        "kind": "chart",
                        "bbox": [220, 20, 380, 100],
                        "primary_method": "chart_parser",
                    },
                    {
                        "id": "region_flow_pipeline_0",
                        "kind": "pipeline_context_row",
                        "bbox": [20, 20, 180, 120],
                        "primary_method": "pipeline_context_layout",
                    },
                ]
            }
            review = visual_review.review(
                compare,
                vlm=None,
                canvas_width=400,
                canvas_height=120,
                strategy_plan=plan,
            )
        region_ids = {d["region_id"] for d in review["defects"]}
        self.assertEqual(region_ids, {"region_chart_0", "region_flow_pipeline_0"})
        self.assertNotIn("left_surface", region_ids)
        self.assertNotIn("auditor_cards", region_ids)

    def test_representation_plan_attaches_typography_contract(self):
        regions = [{
            "id": "region_flow",
            "kind": "pipeline_context_row",
            "bbox": [0, 0, 300, 100],
            "primary_method": "pipeline_context_layout",
        }]
        representation_plan.attach_to_regions(regions)
        rep = regions[0]["representation"]
        self.assertEqual(rep["typography_contract"]["method"], "pipeline_context_layout")
        task = {"id": "task_region_flow"}
        representation_plan.apply_to_task(task, regions[0])
        self.assertEqual(task["typography_contract"]["method"], "pipeline_context_layout")

    def test_typography_contract_scores_role_control(self):
        task = {
            "bbox": [0, 0, 300, 100],
            "locked_method": "pipeline_context_layout",
        }
        controlled = {
            "elements": [{
                "id": "pipeline_context_text_raw",
                "type": "text",
                "bbox": [10, 10, 140, 45],
                "text": "Raw Tables",
                "font_size": 14,
                "ext": {"typography": {"role": "process_card", "fit_width_factor": 0.45}},
            }]
        }
        uncontrolled = {
            "elements": [{
                "id": "ocr_1",
                "type": "text",
                "bbox": [10, 10, 140, 45],
                "text": "Raw Tables",
                "font_size": 14,
                "ext": {},
            }]
        }
        self.assertGreater(
            typography.score_contract(controlled, task)["score"],
            typography.score_contract(uncontrolled, task)["score"],
        )

    def test_proposal_candidate_render_falls_back_to_proxy(self):
        with TemporaryDirectory() as td:
            td_path = Path(td)
            old_build = proposal_orchestrator.builder.build_pptx
            old_available = proposal_orchestrator.renderer.is_available
            old_render = proposal_orchestrator.renderer.render_isolated
            old_proxy = proposal_orchestrator.v2_render.render
            try:
                proposal_orchestrator.builder.build_pptx = lambda ir, out: Path(out).write_bytes(b"pptx")
                proposal_orchestrator.renderer.is_available = lambda: True

                def fail_render(*_args, **_kwargs):
                    raise RuntimeError("powerpoint unavailable")

                proposal_orchestrator.renderer.render_isolated = fail_render
                proposal_orchestrator.v2_render.render = lambda ir, original: Image.new("RGB", original.size, "white")
                out = proposal_orchestrator._render_candidate_image(
                    {"elements": [], "canvas": {"width_px": 100, "height_px": 60}},
                    Image.new("RGB", (100, 60), "white"),
                    td_path / "source.png",
                    td_path,
                    "candidate",
                )
                self.assertTrue(out.endswith(".proxy.png"))
                self.assertTrue(Path(out).exists())
            finally:
                proposal_orchestrator.builder.build_pptx = old_build
                proposal_orchestrator.renderer.is_available = old_available
                proposal_orchestrator.renderer.render_isolated = old_render
                proposal_orchestrator.v2_render.render = old_proxy

    def test_isolated_renderer_timeout_is_reported(self):
        old_run = renderer.subprocess.run
        old_exists = Path.exists
        try:
            renderer.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd="render", timeout=3)
            )
            Path.exists = lambda self: True
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                renderer.render_isolated("/tmp/fake.pptx", "/tmp/fake.png")
        finally:
            renderer.subprocess.run = old_run
            Path.exists = old_exists

    def test_method_registry_infers_methods_from_generic_visual_contracts(self):
        cases = [
            (
                "The 3D manifold and coordinate space are flattened; rebuild as editable vector surface.",
                "editable generated surface with axes and vector arrows",
                "procedural_surface",
            ),
            (
                "The chart is missing axes, ticks, legend, and line series.",
                "editable native chart axes and data series",
                "chart_parser",
            ),
            (
                "The process flow row has broken arrows and disconnected labels.",
                "editable process cards and connectors",
                "pipeline_context_layout",
            ),
        ]
        for problem, expected_expression, method in cases:
            with self.subTest(method=method):
                self.assertEqual(
                    method_registry.infer_method(
                        kind="visual_region_defect",
                        objective=problem,
                        expected_native_expression=expected_expression,
                    ),
                    method,
                )

    def test_ownerless_defects_route_from_region_representation_contract(self):
        chart_rep = method_registry.contract_for_method("chart_parser")
        ir = {
            "elements": [],
            "strategy_plan": {
                "regions": [{
                    "id": "generic_chart_region",
                    "kind": "chart",
                    "bbox": [10, 10, 160, 100],
                    "primary_method": "chart_parser",
                    "representation": {
                        "method": chart_rep["method"],
                        "owner_agent": chart_rep["owner_agent"],
                        "component_template": chart_rep.get("component_template", ""),
                    },
                    "fallback_methods": ["native_trace"],
                }]
            },
            "defects": [{
                "id": "missing_chart_ink",
                "type": "missing_element",
                "bbox": [20, 20, 150, 90],
                "severity": 0.9,
            }],
        }
        strategy.apply_defect_strategy(ir)
        defect = ir["defects"][0]
        self.assertEqual(defect["suggested_agent"], "ChartAgent")
        self.assertEqual(defect["strategy"]["method"], "chart_parser")

    def test_strategy_infers_scientific_surface_without_surface_entity(self):
        entities = [
            {"id": "t_cov", "type": "text", "bbox": [60, 70, 230, 100], "text": "covariate space X"},
            {"id": "f_vec", "type": "formula", "bbox": [310, 120, 460, 160], "latex": "beta = nabla e(x)"},
            {"id": "t_het", "type": "text", "bbox": [420, 270, 560, 320], "text": "high heterogeneity"},
        ]
        plan = strategy.plan_from_entities(entities, 1000, 600)
        regions = plan["regions"]
        surface = [r for r in regions if r.get("kind") == "procedural_3d_surface"]
        self.assertEqual(len(surface), 1)
        self.assertEqual(surface[0]["primary_method"], "procedural_surface")
        self.assertEqual(
            surface[0]["representation"]["owner_agent"],
            "ProceduralSurfaceAgent",
        )

    def test_strategy_infers_generic_flow_pipeline_from_repeated_blocks(self):
        entities = [
            {"id": "panel", "type": "container", "bbox": [20, 20, 760, 240]},
            {"id": "step1", "type": "shape", "bbox": [70, 55, 130, 210]},
            {"id": "step2", "type": "shape", "bbox": [210, 55, 290, 210]},
            {"id": "step3", "type": "shape", "bbox": [380, 55, 460, 210]},
            {"id": "step4", "type": "shape", "bbox": [560, 55, 650, 210]},
            {"id": "label", "type": "text", "bbox": [30, 28, 220, 48], "text": "Overall Architecture"},
        ]
        plan = strategy.plan_from_entities(entities, 1000, 600)
        flows = [r for r in plan["regions"] if r.get("kind") == "pipeline_context_row"]
        self.assertEqual(len(flows), 1)
        self.assertEqual(flows[0]["primary_method"], "pipeline_context_layout")
        self.assertEqual(
            flows[0]["representation"]["owner_agent"],
            "PipelineContextAgent",
        )
        self.assertFalse(plan["candidate_policy"]["try_residual_replacement"])
        self.assertEqual(plan["candidate_policy"]["residual_candidate_limit"], 0)

    def test_specialist_agent_records_method_contract_result(self):
        rep = method_registry.contract_for_method("pipeline_context_layout")
        ir = {
            "canvas": {"width_px": 1000, "height_px": 500},
            "round": 0,
            "elements": [],
        }
        task = {
            "id": "generic_pipeline_task",
            "kind": "visual_region_defect",
            "region_id": "any_new_flow_region",
            "bbox": [500, 60, 960, 150],
            "locked_method": "pipeline_context_layout",
            "representation": {
                "method": rep["method"],
                "required_agents": rep["required_agents"],
                "forbid_agents": rep["forbid_agents"],
                "acceptance_policy": rep["acceptance_policy"],
            },
            "objective": "Rebuild an aligned process flow row with arrows.",
        }
        changed = PipelineContextAgent().run(
            ir,
            Image.new("RGB", (1000, 500), "white"),
            task=task,
        )
        self.assertGreater(len(changed), 0)
        result = ir["agent_contract_results"][-1]
        self.assertEqual(result["method"], "pipeline_context_layout")
        self.assertEqual(result["agent"], "PipelineContextAgent")
        self.assertTrue(result["satisfied"])

    def test_pipeline_context_agent_uses_generic_flow_for_architecture_region(self):
        rep = method_registry.contract_for_method("pipeline_context_layout")
        ir = {
            "canvas": {"width_px": 800, "height_px": 400},
            "round": 0,
            "elements": [
                {
                    "id": "shape_a",
                    "type": "rounded_rect",
                    "bbox": [80, 40, 130, 180],
                    "fill": "#e3d4e7",
                    "border_color": "#e3d4e7",
                },
                {
                    "id": "shape_b",
                    "type": "rounded_rect",
                    "bbox": [180, 40, 240, 180],
                    "fill": "#c9dde3",
                    "border_color": "#c9dde3",
                },
                {
                    "id": "title",
                    "type": "text",
                    "bbox": [20, 5, 180, 25],
                    "text": "A. Overall Architecture",
                },
            ],
        }
        task = {
            "id": "task_region_flow_pipeline_0",
            "kind": "pipeline_context_row",
            "region_id": "region_flow_pipeline_0",
            "bbox": [0, 0, 360, 220],
            "locked_method": "pipeline_context_layout",
            "objective": "Rebuild an overall architecture flow region.",
            "representation": {"method": rep["method"]},
        }
        changed = PipelineContextAgent().run(
            ir,
            Image.new("RGB", (800, 400), "white"),
            task=task,
        )
        ids = {e["id"] for e in ir["elements"]}
        self.assertTrue(any(x.startswith("generic_flow_card_") for x in changed))
        self.assertIn("shape_a", ids)
        self.assertIn("shape_b", ids)
        self.assertNotIn("shape_a", changed)
        scores = proposal_orchestrator._candidate_structure_scores(
            {"roles": ["PipelineContextAgent"], "ir": ir},
            task,
        )
        self.assertIn("pipeline_context", scores)
        labels = {
            str(e.get("text"))
            for e in ir["elements"]
            if e.get("type") == "text"
        }
        self.assertFalse({"Raw\nTables", "CATE\nEstimator", "CI\nEstimator"} & labels)

    def test_pipeline_context_agent_consumes_region_semantic_contract(self):
        class FakeProvider:
            name = "fake_vlm"

            def ask_json(self, *_args, **_kwargs):
                return {
                    "title": {"text": "System Flow", "bbox": [0.02, 0.02, 0.30, 0.10]},
                    "blocks": [
                        {"label": "Input", "bbox": [0.10, 0.25, 0.22, 0.75], "role": "process", "rotation": 90},
                        {"label": "Encoder", "bbox": [0.38, 0.25, 0.52, 0.75], "role": "process", "rotation": 90},
                        {"label": "Output", "bbox": [0.70, 0.25, 0.84, 0.75], "role": "process", "rotation": 90},
                    ],
                    "connectors": [
                        {"bbox": [0.23, 0.48, 0.37, 0.54], "direction": "right"},
                    ],
                }

        old_disable = os.environ.get("I2E_DISABLE_FLOW_REGION_VLM")
        os.environ["I2E_DISABLE_FLOW_REGION_VLM"] = "0"
        try:
            ir = {
                "canvas": {"width_px": 600, "height_px": 240},
                "round": 0,
                "elements": [
                    {"id": "shape_a", "type": "rounded_rect", "bbox": [60, 60, 120, 180]},
                    {"id": "shape_b", "type": "rounded_rect", "bbox": [230, 60, 310, 180]},
                    {"id": "shape_c", "type": "rounded_rect", "bbox": [420, 60, 500, 180]},
                    {"id": "ocr_noise", "type": "text", "bbox": [64, 86, 116, 100], "text": "— RE"},
                ],
            }
            task = {
                "id": "task_region_flow_pipeline_0",
                "kind": "pipeline_context_row",
                "region_id": "region_flow_pipeline_0",
                "bbox": [0, 0, 560, 220],
                "locked_method": "pipeline_context_layout",
                "objective": "Rebuild an overall architecture flow region.",
                "representation": {"method": "pipeline_context_layout"},
            }
            agent = PipelineContextAgent()
            agent.provider = FakeProvider()
            changed = agent.run(ir, Image.new("RGB", (600, 240), "white"), task=task)
        finally:
            if old_disable is None:
                os.environ.pop("I2E_DISABLE_FLOW_REGION_VLM", None)
            else:
                os.environ["I2E_DISABLE_FLOW_REGION_VLM"] = old_disable

        texts = {e.get("text") for e in ir["elements"] if e.get("type") == "text"}
        self.assertTrue({"Input", "Encoder", "Output"} <= texts)
        self.assertTrue(any(x.startswith("generic_flow_semantic_arrow_") for x in changed))
        self.assertEqual(ir["agent_tool_calls"][-1]["action"], "read_flow_region_semantics")

    def test_pipeline_context_agent_uses_ocr_fallback_when_vlm_semantics_fail(self):
        class FailingProvider:
            name = "fake_vlm"

            def ask_json(self, *_args, **_kwargs):
                raise TimeoutError("slow")

        class FakeOCRProvider:
            name = "fake_ocr"

            def ocr(self, *_args, **_kwargs):
                return "Input\nEncoder\nOutput"

        old_disable = os.environ.get("I2E_DISABLE_FLOW_REGION_VLM")
        old_ocr_disable = os.environ.get("I2E_DISABLE_FLOW_REGION_OCR")
        os.environ["I2E_DISABLE_FLOW_REGION_VLM"] = "0"
        os.environ["I2E_DISABLE_FLOW_REGION_OCR"] = "0"
        try:
            ir = {
                "canvas": {"width_px": 600, "height_px": 240},
                "round": 0,
                "elements": [
                    {"id": "shape_a", "type": "rounded_rect", "bbox": [60, 60, 120, 180]},
                    {"id": "shape_b", "type": "rounded_rect", "bbox": [230, 60, 310, 180]},
                    {"id": "shape_c", "type": "rounded_rect", "bbox": [420, 60, 500, 180]},
                ],
            }
            task = {
                "id": "task_region_flow_pipeline_0",
                "kind": "pipeline_context_row",
                "region_id": "region_flow_pipeline_0",
                "bbox": [0, 0, 560, 220],
                "locked_method": "pipeline_context_layout",
                "objective": "Rebuild an overall architecture flow region.",
                "representation": {"method": "pipeline_context_layout"},
            }
            agent = PipelineContextAgent()
            agent.provider = FailingProvider()
            agent.ocr_provider = FakeOCRProvider()
            agent.run(ir, Image.new("RGB", (600, 240), "white"), task=task)
        finally:
            if old_disable is None:
                os.environ.pop("I2E_DISABLE_FLOW_REGION_VLM", None)
            else:
                os.environ["I2E_DISABLE_FLOW_REGION_VLM"] = old_disable
            if old_ocr_disable is None:
                os.environ.pop("I2E_DISABLE_FLOW_REGION_OCR", None)
            else:
                os.environ["I2E_DISABLE_FLOW_REGION_OCR"] = old_ocr_disable

        texts = {e.get("text") for e in ir["elements"] if e.get("type") == "text"}
        self.assertTrue({"Input", "Encoder", "Output"} <= texts)
        self.assertEqual(ir["agent_tool_calls"][-1]["action"], "read_flow_region_ocr_labels")
        self.assertEqual(ir["agent_tool_failures"][-1]["action"], "read_flow_region_semantics")

    def test_chart_agent_builds_generic_native_chart_without_vlm(self):
        im = Image.new("RGB", (240, 140), "white")
        draw = ImageDraw.Draw(im)
        draw.line([30, 110, 210, 110], fill="#222222", width=2)
        draw.line([30, 20, 30, 110], fill="#222222", width=2)
        pts = [(30, 95), (60, 80), (95, 88), (130, 55), (170, 62), (210, 35)]
        draw.line(pts, fill="#1f4e79", width=3)
        ir = {
            "canvas": {"width_px": 240, "height_px": 140},
            "round": 0,
            "elements": [{
                "id": "chart_panel",
                "type": "rounded_rect",
                "bbox": [12, 6, 228, 128],
                "fill": "#ffffff",
                "border_color": "#dddddd",
            }, {
                "id": "adjacent_chart",
                "type": "chart",
                "bbox": [20, 90, 220, 150],
                "chart": {"kind": "line"},
            }],
        }
        task = {
            "id": "task_region_chart_0",
            "kind": "visual_region_defect",
            "region_id": "region_chart_0",
            "bbox": [20, 10, 220, 120],
            "locked_method": "chart_parser",
            "acceptance_policy": "semantic_chart",
        }
        changed = ChartAgent().run(ir, im, task=task)
        self.assertTrue(any(x.startswith("generic_chart_region_chart_0") for x in changed))
        self.assertIn("chart_panel", {e["id"] for e in ir["elements"]})
        self.assertNotIn("chart_panel", changed)
        self.assertIn("adjacent_chart", {e["id"] for e in ir["elements"]})
        self.assertNotIn("adjacent_chart", changed)
        score = proposal_orchestrator._generic_chart_structure_score(ir, task)
        self.assertGreaterEqual(score, 0.72)

    def test_chart_transaction_commits_generic_chart_primitives(self):
        base = {"round": 0, "elements": []}
        candidate = {
            "round": 0,
            "elements": [{
                "id": "generic_chart_region_chart_0_axis_x",
                "type": "line",
                "bbox": [20, 100, 220, 101],
                "points": [20, 100, 220, 100],
            }],
        }
        task = {
            "id": "task_region_chart_0",
            "kind": "visual_region_defect",
            "bbox": [10, 10, 230, 120],
            "locked_method": "chart_parser",
        }
        merged, transaction = proposal_orchestrator._regional_transaction_ir(
            base,
            candidate,
            task,
            ["generic_chart_region_chart_0_axis_x"],
        )
        self.assertEqual(transaction["committed"], ["generic_chart_region_chart_0_axis_x"])
        self.assertEqual(
            [e["id"] for e in merged["elements"]],
            ["generic_chart_region_chart_0_axis_x"],
        )

    def test_procedural_surface_agent_seeds_surface_from_contract(self):
        rep = method_registry.contract_for_method("procedural_surface")
        ir = {
            "canvas": {"width_px": 1200, "height_px": 700},
            "round": 0,
            "elements": [],
        }
        task = {
            "id": "generic_surface_task",
            "kind": "procedural_3d_surface",
            "region_id": "new_scientific_surface",
            "bbox": [20, 40, 720, 430],
            "locked_method": "procedural_surface",
            "representation": {
                "method": rep["method"],
                "required_agents": rep["required_agents"],
                "forbid_agents": rep["forbid_agents"],
                "acceptance_policy": rep["acceptance_policy"],
            },
        }
        changed = ProceduralSurfaceAgent().run(
            ir,
            Image.new("RGB", (1200, 700), "white"),
            task=task,
        )
        self.assertGreater(len(changed), 0)
        self.assertTrue(any(e.get("type") == "surface" for e in ir["elements"]))
        result = ir["agent_contract_results"][-1]
        self.assertEqual(result["method"], "procedural_surface")
        self.assertTrue(result["satisfied"])

    def test_regression_suite_case_naming_is_data_only(self):
        self.assertEqual(
            regression_suite._case_name(
                Path("/tmp/My Figure.png.ocr_upscale.png"), 2),
            "02_My_Figure.png",
        )
        with TemporaryDirectory() as td:
            artifacts = regression_suite._artifacts(Path(td))
            self.assertIn("diagram_v3.compare.png", artifacts)
            self.assertIsNone(artifacts["diagram_v3.compare.png"])


if __name__ == "__main__":
    unittest.main()
