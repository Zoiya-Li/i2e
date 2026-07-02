# diagram2ppt — 当前状态战报

> 更新：2026-06-27  
> 负责人：AI agent + 用户协作整理

---

## 1. 这是什么

`diagram2ppt` 是 i2e 项目在 2026-06  pivoted 后的**主战场**：把学术/技术框架图（如 `framework.png`）重建成**原生可编辑**的 PowerPoint / SVG。

核心产品契约：
- 文字是文本框
- 方框是 PowerPoint shape
- 箭头是 connector
- 公式是 OMML 对象
- 图表是原生 chart
- （v10+）流形/曲面是可编辑 SVG 矢量

与 Lovart/OmniPSD/Qwen-Image-Layered 的区别：它们拆 RGBA 图层，我们拆语义原生对象。

---

## 1.5 状态词汇表 + 最终目标（Target）

**状态词汇表（贯穿本文）**：**Current** = 仓库已有、可运行 · **Active** = 当前主开发线 · **Frozen** = 已完成仅维护 · **Target** = 最终架构方向，**尚未实现**。不要把 `Target` 当作已存在的文件/API/稳定行为。

**最终目标（Target）**：diagram2ppt 不止是「图片 → PPTX/SVG」，而是 i2e **Visual Design Decompiler** 的主实验场——把整页处理演进成可保存、可审计、可迭代的反编译系统：

```
Input → Preprocess → Evidence Extraction → Component Decomposition → Local Object Generation
      → Constraint Recovery → Editable Design IR → Global Assembly → SVG/PPTX Renderer
      → Preview Rendering → Visual Diff/Audit → Error Routing → Targeted Refinement → Export → Correction Capture
```

核心闭环（Target）：`rendered preview → compare with original → locate error → map error to component/element → create refinement task → update IR → re-render`。

原则：不要一次性让模型生成最终 SVG/PPTX；不要整页盲目重生成；不要把模型输出直接当真相；所有中间产物必须可保存、可 diff、可复现。

**v3 演进路线（Target，尚未全部落地）**：
- **P0** 让 v3 稳定跑完一张新图：换轻量模型 / 调 timeout、降 max-rounds、允许 `partial / rejected / fallback` 状态、每次运行输出 run manifest。验收标准是**失败可诊断**，不是质量完美。
- **P1** 运行态契约：`RunManifest / PipelineState / Task / TaskResult / AuditResult / Component / Element / Constraint / FallbackRecord / CorrectionRecord`，并落盘全部中间产物。
- **P2** 组件级局部闭环：`crop → local generate → local render → local diff → local refine → accepted`（先 text / rect / line-arrow / simple group / formula / raster fallback）。
- **P3** SVG canonical loop：`IR → SVG → PNG preview → diff → refine`，SVG 作为中间验证层（坐标接近像素，便于验证）。
- **P4** PPTX native lowering：text→textbox、rect→autoshape、line/arrow→connector、formula→OMML、table→native table、chart→native chart、fallback→picture；**必须过真 PowerPoint 验收**。
- **P5** audit-driven refinement：audit 产出可执行修复任务（映射 component_id / element_id / suggested_task），错误类型如 missing/extra/wrong_text/wrong_bbox/wrong_font_size/wrong_color/wrong_shape/wrong_connector/wrong_formula/wrong_z_order/low_editability。

**多维质量指标（Target）**：visual/text/layout/style score、object_coverage、editability_score、native_object_ratio、fallback_area_ratio、correction_count、convergence_rounds、true_ppt_render_score。注意：整页截图当背景可刷高视觉分但产品价值低，评分必须同时压 fallback 面积、抬 native 比例与可编辑性。

**IR 关系**：短期不破坏单层 `ir/ir-v1.schema.json`（承重墙）；多层 IR（Evidence / Component / Editable Design / Correction）在 v3 内**增量、非破坏**引入。

### 1.6 已交付基础设施（Current，2026-07-01）

面向 Target 的 P0–P4 优先级已落地为**可运行、离线可测**的基础设施（非 Target，均在仓库中）：

| 模块 / 产物 | 作用 | 优先级 |
|---|---|---|
| `v3/runtime/` + `state_log.json` | **P1 Runtime State Machine Kernel**：`RuntimeState` 是单一真相源，`PlannerKernel` 包装现有 Planner。Phase 3 已让 `AuditAgentSystem` 通过 `kernel.transition(...)` 调度 operator，把控制流从 Planner 移入 Kernel；operators 覆盖 perceive / compose / render_verify_audit / proposal_phase / repair / rollback_or_accept / derive_components / audit_tasks / svg_loop / accept / fail / legacy_planner_loop / finalize。每次 transition 落盘 `state_log.json`，支持 `kernel.replay(state_log.json)` 恢复现场 | P1 |
| `v3/run_manifest.py` + `v3/run.py` 包装 | 每次运行都写 `run_manifest.json`（含 timeout/SIGTERM/异常），outcome ∈ accepted/partial/rejected/error/interrupted——**失败可诊断**。P1 扩展：`renderer_mode`、`memory`（run 记忆是否复用）、`last_successful_stage`（stalled 在哪一阶段）、`acceptance_blockers`；**proxy 渲染或缺失 renderer_mode 永不判 accepted**（降级 partial，真 PowerPoint 才是生产验收）。可复现性：`I2E_USE_RUN_MEMORY=0` 关闭同源记忆复用（regression 默认关）。**run.py finally 现自动跑 post-run**：run 一旦 finalize 就 best-effort 生成 `components.json`/`audit_tasks.json`/`svg_loop.json` 并记入 manifest（`--no-postprocess` 可关） | P0/P1 |
| `v3/pptx_stats.py` | 确定性 PPTX 结构指纹（shape 直方图、pictures 数、OMML 数、native_object_ratio、sha256），无网络 | P1/P3 |
| `v3/baselines/v2_framework.json` | 冻结的 v2 回归基线（**实测**：97 shapes / 7 pictures / 9 OMML / native_ratio 0.9278），并记录与旧文档"0 图片=v3.3"说法的出入 | P1 |
| `regression_suite.py`：`load_v2_baseline` / `compare_to_baseline` | 每份回归报告内嵌 v2 基线；每个 v3 case 产物与基线对比可编辑性（pictures/native_ratio delta） | P3 |
| `v3/triage.py` + `v3_out_index.{json,md}` | 非破坏地扫描并分类 v3 输出目录（当前 29 个活跃 run，全部 partial，含 editability/fallback 列，可按 visual_delta 排序；最佳 `v3_out_testpng_default` 0.357/cov1.0） | P2 |
| `v3/metrics.py` | §8 多维指标（离线部分）：native_element_ratio、fallback_area_ratio、editability_score、object_coverage；visual/text 分数从 `ir['metrics']` 透传（需真实渲染） | §8 |
| `v3/fallback.py` | §9 fallback 审计：识别 raster_crop/editable=False，校验「局部+显式+可追踪」，标记 undocumented / full_page 违规（`ext.forced` 视为强制原生，不算 fallback） | §9 |
| `v3/components.py` + `components.json` | P2 Component IR：把 strategy 区域升级为一等 `Component`（生命周期 planned/generated/rendered/audited/accepted/fallback、component-local 指标、per-component crop + sub-IR、provenance）。CLI `python -m work.diagram2ppt.v3.components <run_dir>`。`local_visual_delta` 为 Target 钩子（待组件级 render/diff） | P2 |
| `v3/audit_tasks.py` + `audit_tasks.json` | P5 统一可执行审计任务：把 verifier defects + visual_review defects 合并为单一 `AuditTask`（type ∈ refine_geometry/refine_text/rebuild_component/apply_fallback、component_id/element_id、source_error、severity、acceptance gate），按严重度排序。CLI `python -m work.diagram2ppt.v3.audit_tasks <run_dir>` | P5 |
| `v3/builder.py` build profiles | P4 fallback 分层：`--profile all_native`（研究，零 raster）vs `product_delivery`（允许**有文档的局部** fallback，拒绝 undocumented / full_page）；经 `I2E_BUILD_PROFILE` 生效。`group` 两档都拒（尚不可渲染） | P4 |
| `v3/svg_loop.py` + `svg_loop.json` | P3 SVG canonical loop：v3 IR → SVG（复用 v2 `export_svg`，chart 占位兜底 + minimal-SVG fallback，永不崩）→ PNG（`rsvg-convert`，缺工具则跳过）→ 与原图 pixel diff。CLI `python -m work.diagram2ppt.v3.svg_loop <run_dir>`；实测 testpng `visual_delta_vs_source=0.087`。SVG 是 debug/preview 层，不取代 PPTX 交付 | P3 |
| `v3/visual_review.py` REGION_PRIORS | 明确标注为 **framework.png 专属 fixture、仅最后兜底**；通用 review 已从 `strategy_plan` 区域生成（`_semantic_regions_px`，兜底为通用 whole-slide 而非该 fixture） | 清理 |
| 测试 `test_run_manifest.py` `test_runtime_kernel.py` `test_v2_baseline.py` `test_triage.py` `test_metrics.py` `test_fallback.py` `test_components.py` `test_audit_tasks.py` `test_builder_profiles.py` `test_svg_loop.py` `tests/test_capture_corrections.py` | 上述契约 + Correction schema + RuntimeState / Kernel / replay / legacy loop 的离线回归（全套件 **191 passed**；重跑 `pytest tests/ work/diagram2ppt/tests/ -q`） | — |

> **实测洞见（由新指标暴露）**：全部 29 个 v3 run 的 editability=1.0 / fallback=0（全原生政策生效），但最佳 `visual_delta` 仅 0.357——即 **v3 在可编辑性上已胜过 v2（1.0 vs 0.735），输在视觉保真**。v2 hybrid 交付含 26.5% raster fallback 面积且 7 处均无 §9 文档。瓶颈是收敛/保真，不是可编辑性。

> 说明：`regression_suite` 默认模型已从超时的 `Qwen3.6-35B-A3B` 对齐到 `run.py` 的 `Qwen3-VL-32B-Instruct`。仍未解决的是**让 v3 端到端产出 `status: accepted`**——那需要真实 provider/GPU 与方法收敛，属 Target 研究，不是基础设施。

---

## 2. 两条代码线

### 2.1 v2 — 稳定基线（已基本完成）

**位置**：`work/diagram2ppt/v2/`

**架构**：全局 VLM 理解 → 类型路由（text/OCR、formula/LaTeX-OCR、box/arrow、chart、3D/illustration）→ 组装渲染 → 迭代修复环（render-diff → refine/demote/identify）。

**最佳交付物**：
- `work/diagram2ppt/v22_out/diagram_final.pptx` —— **实测为 hybrid 交付**（原生对象 + 忠实裁切兜底），**不是**零图片全原生。
- 实测结构（`v3/pptx_stats.py`）：97 shapes = 43 text + 38 autoshape + 6 connector + 2 chart + 1 freeform + **7 pictures**；9 OMML 公式；native_object_ratio **0.9278**；约 **26% 面积为 raster fallback**。
- v2 自有指标：native fraction **0.70 / 0.696**（count / area），coverage **0.972**。
- 策略：顽固元素降级为忠实裁切（faithful crop），保证 100% 保真。
- 注：早期文档"0 图片 + 402 dots + 12 icons"描述对应的是**另一个** all-native 实验产物 `diagram_v33.pptx`（~60KB），与磁盘上的 `diagram_final.pptx` 不是同一文件。冻结基线见 `v3/baselines/v2_framework.json`。

**v2 后续演进**：
- `v2_out/`：早期输出
- `v22_out/`：加入 OCR 专家、公式/图表专家后的最终交付
- `v24_out/`：后续微调
- `v2_out_1920/`：1920 宽度版本测试

**状态**：✅ **可用**。`pytest tests/test_diagram_v2.py` 19/19 通过（刚修复 2 处断言漂移）。

---

### v2 与 v3 的关系（必读）

| | v2 | v3 |
|---|---|---|
| **角色** | 生产基线（production baseline） | 下一代研究管线（research successor） |
| **何时用** | 今天要结果 | 探索新架构 / 做消融 |
| **fidelity 策略** | 原生对象 + 忠实裁切 fallback | 强制全原生，零图片 |
| **稳定度** | 已交付可用 PPTX | 还没有 accepted 输出 |
| **代码关系** | v3 大量导入 v2：decompose、handlers、build_pptx、render、diff、snapshot | 在 v2 上加 orchestration / audit / contract 层 |

**工作约定**：
1. **v2 只修 bug，不加新大功能。** 它是当前唯一能交付的 pipeline。
2. **v3 做新架构。** 等 v3 稳定并显著超过 v2 后，再切换默认入口。
3. **不要同时在 v2 和 v3 做同一类改动。** 先在 v3 验证，再决定是否 backport 到 v2。

更多细节：
- `work/diagram2ppt/v2/README.md`
- `work/diagram2ppt/v3/README.md`

---

### 2.2 v3 — 当前活跃开发线（agent/audit 重构中）

**位置**：`work/diagram2ppt/v3/`

**目标**：从 v2 的 "extract-and-demote" 转向 **"search-and-audit"**——用 blackboard IR、语义区域计划、多 agent 竞争提案、真实 PowerPoint 渲染验证，强制输出 100% native 对象。

**新增核心模块**（2026-06-22 ~ 06-27）：

| 模块 | 职责 |
|---|---|
| `perception_orchestrator.py` | 多源感知融合：CV 几何 + VLM 结构 + VLM 文本 + OCR |
| `content_orchestrator.py` | 把 v2 handlers 包装成 typed ContentAgent，线程池执行 |
| `strategy.py` | 将实体映射成语义区域（surface、pipeline、auditor_cards、action_cards 等） |
| `method_registry.py` | 重建方法契约注册表（required agents、acceptance policy、native expression） |
| `representation_plan.py` | 把方法契约附加到策略区域 |
| `proposal_orchestrator.py` | 区域级多 agent 提案、渲染、验证、提交 |
| `audit_agent_system.py` | 外层审计循环：决定下一步用 proposal_phase、single_repair 还是 stop |
| `quality_gate.py` | 预构建清洗：OCR 去噪、公式规范化、卡片文本拆分 |
| `typography.py` | 字阶系统、模板 slot、排版约束 |
| `regression_suite.py` | 回归测试套件 |
| `renderer.py` | 真实 PowerPoint 渲染（macOS AppleScript → PDF → PNG） |
| `verifier.py` | PPTX 与原图对比，输出缺陷和指标 |
| `semantic.py` | 语义补丁校验：防止内容退化 |

**入口**：
```bash
python -m work.diagram2ppt.v3.run <image> -o <out_dir> --max-rounds N
```

**默认模型**（`run.py`）：SiliconFlow `Qwen/Qwen3-VL-32B-Instruct`（VLM/vision）、`Qwen/Qwen3.5-397B-A17B`（planner）。provider 级 fallback（未设 `I2E_VLM_MODEL` 时）仍为旧的 `Qwen/Qwen3.6-35B-A3B`。

**当前问题**：
- 默认模型在 `test.png` 上 timeout（第一次跑 90s 超时，第二次加 timeout 后被 kill）
- 未产生过 `status: accepted` 的完整输出
- 源码在快速重构，接口和默认值还在变

**状态**：🚧 **活跃开发中，未稳定**。

---

## 3. 输出目录整理

### 3.1 当前布局（2026-06-27 整理后）

```
work/diagram2ppt/
  v2/                      # v2 稳定源码
  v3/                      # v3 活跃源码
  tests/                   # v3 perception contract tests
  DEFECTS.md               # framework.png 缺陷台账（v1-v10.2）
  STATUS.md                # 本文件
  archive_202506/          # 归档的旧 v3 实验输出
    v3_outputs_2026-06-19/
    v3_outputs_2026-06-20/
    v3_outputs_2026-06-21/
  v3_out_*                 # 保留的近期 v3 输出（59 个，Jun 22-24）
  v2_out/                  # v2 早期输出
  v22_out/                 # v2 最终交付输出
  v24_out/                 # v2 后续微调输出
  v2_out_1920/             # v2 1920 宽度测试
```

### 3.2 归档规则

- **已归档**：2026-06-19 ~ 06-21 的 140 个 `v3_out_*` 目录（均为早期 fix/run/planner/visual_planner 迭代）
- **保留原地**：2026-06-22 ~ 06-24 的 59 个目录，供当前开发参考
- **未删除任何文件**，仅做目录移动

### 3.3 值得关注的目录

| 目录 | 说明 |
|---|---|
| `v22_out/` | v2 最终可用交付 |
| `v3_out_test_root2/` | 我们最近一次跑 `test.png` 的输出（被 kill，但最完整）|
| `v3_out_fw_1.5x_repr194_surface_cluster/` | 最新一次 repr 实验（Jun 23）|

---

## 4. 缺陷与已知阻塞

### 4.1 framework.png 缺陷台账（来自 `DEFECTS.md`）

截至 v10.2：
- 早期 15 个缺陷（D1-D15）大部分已 closed 或 closed-partial
- D16（流形立体感）通过 SVG 渐变矢量曲面解决
- v9.0 范式修正：从“全部重画”转向**选择性保真**
- v10.0：SVG 渐变矢量曲面实现可编辑 + 保真
- v10.2：彻底禁止原图截图，所有内容走矢量

### 4.2 当前技术阻塞

| 阻塞 | 影响 | 状态 |
|---|---|---|
| v3 默认模型 timeout | 无法稳定跑通新图 | 需要换模型或调 timeout |
| v3 未产生 accepted 输出 | 无法宣称 v3 可用 | 需要收敛 acceptance criteria |
| v2/v3 并行维护 | 精力分散 | 需要明确 v2 冻结、v3 主攻 |
| 大量近期 v3_out 目录仍乱 | 找结果困难 | 已归档旧目录，新目录待进一步分类 |

---

## 5. 测试状态

### 5.1 全局 pytest

```bash
python -m pytest tests/ work/diagram2ppt/tests/ -q
# 结果：191 passed, 0 failed
```

### 5.2 各测试文件

| 文件 | 结果 | 说明 |
|---|---|---|
| `tests/test_diagram_v2.py` | 19/19 passed | 刚修复 `_fit_font_px` 和 formula OMML 断言漂移 |
| `tests/test_diagram_v3.py` | 全绿 | v3 IR、builder、agents、providers |
| `work/diagram2ppt/tests/test_perception_contracts.py` | 26/26 passed | v3 契约层 |
| 其他 tests/* | 全绿 | omnimatte、plate、remote、gemini provider 等 |

---

## 6. 下一步建议（按优先级）

### P0：让 v3 能稳定跑完一张新图
- 换更轻量/可靠的模型，或增加 timeout
- 先只跑 `Planner.plan()` 阶段，确认感知 + 策略链路通
- 记录一条“官方推荐命令”

### P1：明确 v2/v3 分工
- v2 冻结为 production baseline
- v3 作为下一代 agentic pipeline 继续开发
- v2 的优质输出作为 v3 的对比基准

### P2：进一步整理 v3_out 目录
- 把保留的 59 个目录按类型分组（fix / repr / test / arch / walkthrough）
- 或者为每次重要实验建立带 README 的结果目录

### P3：更新顶层 STATUS.md
- 项目整体状态（core i2e + omnimatte + diagram2ppt）需要更新
- `STATUS.md`（根目录）仍停留在 2026-05-27，未反映 diagram2ppt pivot

---

## 7. 关键文档索引

| 文档 | 内容 |
|---|---|
| `docs/positioning-20260610.md` | 战略 pivot：从拆像素到语义原生结构 |
| `docs/diagram2ppt-progress.md` | v1→v3 演进史、实验结果、工程坑 |
| `work/diagram2ppt/DEFECTS.md` | framework.png 逐轮缺陷与修复证据 |
| `work/diagram2ppt/v3/docs/execution-semantics-spec.md` | v3 运行语义审计：当前是 TSI，与真 state-machine kernel 的差距及迁移路径 |

---

## 8. 一句话总结

> **v2 已经能交付可用 PPTX；v3 正在从“能跑”升级为“可审计、可复现、全原生”，但还没稳定跑通新图。当前最急的是让 v3 的端到端命令先不 timeout。**
