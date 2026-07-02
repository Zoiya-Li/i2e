# i2e — AGENTS.md

> 本文件写给 AI coding agent。读者应被视作对项目一无所知；所有信息均来自仓库实际内容，不做假设。

> **状态词汇表（贯穿全文，务必按字面理解）**：本文用四类标签区分「已有」与「想要」。**绝不要把 `Target` 当作已存在的文件、API 或稳定行为**——需要用到时先在代码里确认。
>
> - **Current** — 仓库已有、当前可运行的能力。
> - **Active** — 当前主开发线（正在改，接口/默认值会变）。
> - **Frozen** — 已完成，仅 bug 维护，不加新功能。
> - **Target** — 最终架构方向，**尚未实现**；描述的是意图，不是代码。

## 1. 项目概览

**i2e**（image → editable）当前（**Current**）是把 AI 生成的平面视觉图（营销海报、技术框架图等）转换成可编辑、品牌合规的设计资产。

当前主战场：

- **(Active)** **`work/diagram2ppt/`** —— 活跃开发线。把学术/技术框架图重建成原生可编辑的 PowerPoint / SVG。其中：
  - **v2** 是稳定基线（Current），已能产出可用 PPTX（`v22_out/diagram_final.pptx`）。
  - **v3** 是下一代 agent/audit 研究管线（Active），尚未稳定，默认模型在复杂图上可能超时。
- **(Frozen)** **`extractor/`、`editor/`、`render/`、`capture/` 等** —— 核心 poster 管线，已完成并冻结维护。
- **(Frozen, paused)** **`work/poster/`、`work/omnimatte.py` 等** —— editable omnimatte 概念验证，已完成并暂停。

核心产品契约：**AI 只在“拆解”那一刻用一次，之后的编辑是原生、确定、毫秒级的。** 与 Lovart/OmniPSD/Qwen-Image-Layered 的像素 RGBA 分层不同，i2e 输出语义原生结构：文字是文本框、方框是形状、箭头是 connector、公式是 OMML、图表是 chart。

### 1.1 最终目标（Target，尚未实现）

i2e 的长期形态是 **Visual Design Decompiler / 视觉设计反编译系统**：把任意平面视觉资产恢复成可编辑、可审计、可迭代、可跨格式导出的**设计源结构**，而不仅是一张 SVG 或 PPTX。

目标链路（Target）：

```
Visual Artifact → Evidence → Components → Editable Design IR → SVG / PPTX / (future) Figma-like JSON / HTML
```

长期产品契约：AI 可参与反编译过程（转换期多轮 VLM / agent / audit / refine），但**导出后的用户编辑必须是 native / deterministic / fast / auditable / not model-dependent**——改字、改色、移动形状、调 connector、编辑公式或图表时不再依赖 AI 重新生成图片。

**防幻觉清单**：以下均为 **Target**，当前仓库中**不存在**对应的稳定文件或接口，除非你在代码里亲自确认后再引用：

- 多层 IR：`Evidence IR` / `Component IR` / `Editable Design IR` / `Correction IR`。当前承重墙仍是**单层** `ir/ir-v1.schema.json`；短期不破坏 IR v1，多层概念只在 `work/diagram2ppt/v3/` 内逐步引入。
- 运行态契约对象：`RunManifest` / `PipelineState` / `Task` / `TaskResult` / `AuditResult` / `Component` / `Element` / `Constraint` / `FallbackRecord` / `CorrectionRecord`。
- 组件级局部闭环：`crop → local generate → local render → local diff → local refine → component accepted`。
- 审计驱动修复：audit 产出**可执行修复任务**（映射到 component_id / element_id / suggested_task），而非只打分。
- 多维质量指标：`editability_score` / `native_object_ratio` / `fallback_area_ratio` / `true_ppt_render_score` 等。
- 显式 fallback 记录：`editable=false` + `reason` + `source bbox` + `confidence` + `future replacement target`；禁止无标记的整页 fallback。

演进路线（Target，详见 `work/diagram2ppt/STATUS.md`）：P0 让 v3 稳定跑完（不 timeout、允许 partial/rejected/fallback、输出 run manifest）→ P1 运行态契约 → P2 组件级局部闭环 → P3 SVG canonical loop → P4 PPTX native lowering → P5 audit-driven refinement。

## 2. 仓库地图

```
ir/                      IR v1 schema + 样例（承重墙）
extractor/               Node ② — 图片 → IR（VLM 抽取 + 组装校验）
ocr/                     文字几何检测（RapidOCR / Paddle / Tesseract）
segment/                 前景切图（rembg / SAM-2 / MobileSAM）
inpaint/                 背景重建（flat / OpenCV / LaMa）
editor/                  Node ③ — 浏览器编辑器 + 后端 HTTP 服务
render/                  Node ④ — 编辑后 IR → 输出 PNG
capture/                 Node ⑤ — 修正捕获（predicted → corrected）
bench/                   飞轮北极星指标：平均每图修正数
fonts/                   字体/颜色匹配
label/                   VLM 标注 / 漏检发现
verify/                  基于证据的 needs_review 校验
tests/                   pytest + smoke 测试
work/diagram2ppt/        diagram → PPTX/SVG 活跃管线（v2 + v3）
work/poster/             omnimatte poster 概念验证
work/lib/                共享数学/几何帮助函数
archive/                 旧实验与废弃代码
external/                第三方子模块（SAM3、Crafter）
```

## 3. 技术栈与依赖

- **语言**：Python 3.11（仓库中 `__pycache__` 显示 cpython-311）。
- **无根级构建配置**：没有顶层 `pyproject.toml`、`setup.py`、`setup.cfg`、`Makefile`、`tox.ini` 或 `.pre-commit-config.yaml`。依赖直接由 `requirements.txt` 管理。
- **核心依赖**（见 `requirements.txt`）：
  - `jsonschema>=4.20`
  - `pillow>=10`
  - `httpx>=0.27`
  - `anthropic>=0.96`（仅 `--provider anthropic`）
  - `rapidocr-onnxruntime>=1.4`（`--ocr rapid`）
  - `rembg>=2.0`（`--assets rembg`）
  - `opencv-python>=4.8`（`--inpaint opencv`）
- **重型/可选依赖**（按需装在隔离环境或单独进程）：
  - `simple-lama-inpainting`（ photographic 背景重建，torch 重）
  - `ultralytics`（SAM-2 / MobileSAM 分割）
  - `diffusers` + `torch`（SD-1.5 amodal completion）
  - `python-pptx`（v2/v3 PPTX 导出，已装）
- **系统依赖**：macOS 上通常有 `tesseract`（OCR 回退）。

## 4. 构建与安装

```bash
# 安装核心依赖
pip install -r requirements.txt
```

重型模型依赖（LaMa、SAM、SD）按需安装，建议单独 env，避免与 rembg 等包的 pillow/numpy 版本冲突。

## 5. 运行架构

### 5.1 核心 poster 管线（Node 模型）

```
图片
  │
  ▼
Node ② extractor/      VLM/Claude/vision provider → 原始元素 → assemble_ir() → 合法 IR v1
  │
  ▼
Node ③ editor/         浏览器编辑（http://127.0.0.1:8765）
  │
  ▼
Node ④ render/         IR → PNG（fallback / layered）
  │
  ▼
Node ⑤ capture/        每次编辑生成 predicted→corrected Correction
  │
  ▼
Node ⑥/⑦（未来）       corrections 回流重训子模型
```

- IR 是 **load-bearing wall**：`ir/ir-v1.schema.json` + `example-fengyoujing-poster.ir.json`。
- 任何持久化的 IR 必须通过 `extractor.assemble.validate_ir()` 校验。
- Correction 必须含 `field_path` + `kind`，用于把训练信号路由到具体子模型。

### 5.2 diagram2ppt 管线

```bash
# v2 稳定可用（默认输出 SVG，零图片）
python -m work.diagram2ppt.v2.run framework.png -o work/diagram2ppt/v2_out

# v2 legacy 输出 PPTX（native + faithful crop 兜底）
python -m work.diagram2ppt.v2.run framework.png --legacy -o work/diagram2ppt/v2_out

# v3 研究管线（默认 SiliconFlow Qwen3.6-35B-A3B，可能超时）
python -m work.diagram2ppt.v3.run <image.png> -o work/diagram2ppt/v3_out --max-rounds 5
```

**约定**：v2 只修 bug，不加新大功能；v3 做新架构。不要同时在两边做同类改动。

## 6. 测试说明

### 6.1 全量测试

```bash
python -m pytest tests/ work/diagram2ppt/tests/ -q
# 当前：104 passed, 0 failed
```

### 6.2 单条 smoke 测试（均离线，无需 API key）

```bash
python tests/smoke.py                  # 端到端飞轮：extract → edit → corrections → metric
python tests/editor_smoke.py           # Node ③→⑤ save/capture
python tests/export_smoke.py           # Node ④ render
python tests/ocr_smoke.py              # OCR 几何融合
python tests/segment_smoke.py          # 资源实现
python tests/inpaint_smoke.py          # 背景重建 + layered render
python tests/verify_smoke.py           # needs_review 校验
```

### 6.3 测试策略

- 离线优先：使用 `MockProvider`、stub segmenter、合成图片覆盖核心路径。
- v2 测试用 `MockVLM` 回放脚本响应，验证解析、残差、coverage、收敛/降级/识别回退、PPTX 构建。
- v3 有 `work/diagram2ppt/tests/test_perception_contracts.py` 验证契约层。
- 新增功能应补对应 smoke 或 pytest；保持“无网络也能跑核心环”。

## 7. 代码风格与约定

- 每文件开头通常写 `from __future__ import annotations`。
- 使用类型注解；复杂接口用 `typing.Protocol`（如 `Provider`、`Segmenter`、`Inpainter`）。
- 模块职责按 **Node** 划分：extractor/ocr/segment/inpaint/editor/render/capture。
- **provider-agnostic**：中间表示不绑定具体模型；`mock` provider 必须能离线跑通全流程。
- IR 相关：
  - 任何产出 IR 的代码必须调用 `validate_ir()`。
  - `extraction` 必须记录 `confidence`、`model`、`model_version`、`method`。
  - 修正字段走 `corrections[]`，每条有 `field_path`/`kind`/`predicted`/`corrected`。
- 资源路径：cutout / mask / background 等先写 placeholder，后续阶段替换为真实文件，并在 `extraction.method` 中追加 `+<stage>`。
- 不提交生成的 `*.assets/` 目录、`.env`、缓存（见 `.gitignore`）。

## 8. 主要入口命令

```bash
# Node ②：抽取 IR（mock 离线）
python -m extractor.extract poster.png -o out.ir.json --provider mock

# 带 OCR + 切图 + 背景重建
python -m extractor.extract poster.png -o out.ir.json \
  --provider mock --ocr rapid --assets rembg --inpaint flat

# Node ③：启动编辑器
python -m editor.server out.ir.json

# Node ④：把编辑后的 IR 渲染为 PNG
python -m render.export out.edited.ir.json --out out.png

# Node ⑤/指标：计算目录下 IR 的平均修正数
python bench/flywheel.py <dir-of-*.ir.json>

# 验证 IR schema
python -c "import json,jsonschema; jsonschema.Draft202012Validator(json.load(open('ir/ir-v1.schema.json'))).validate(json.load(open('ir/example-fengyoujing-poster.ir.json'))); print('IR OK')"
```

## 9. 安全与运维

- **密钥**：API key（Anthropic / OpenAI-compat / SiliconFlow）通过 `.env` 或环境变量注入。`.env` 已在 `.gitignore` 中，**绝对不要提交**。
- **远程 GPU 环境**：详见 `CLAUDE.md`。
  - 只通过指定 SSH + `docker exec -it 29e8e3afb73f` 进入容器。
  - 工作区限制在 `/home/lzy/AAAI_2026/` 子目录。
  - **禁止**删除任何文件、缓存、checkpoint、日志或临时目录；磁盘满时停止并报告。
  - 使用 GPU 前必须 `nvidia-smi`，不抢占他人任务。
  - 国内 HuggingFace 下载慢时，设置 `HF_ENDPOINT=https://hf-mirror.com`。
- **编辑器安全**：`/asset` 只返回 IR 引用的白名单路径；不要放宽该白名单。
- **数据保护**：`corrections[]` 含用户修正数据，视为训练资产，不要随意删除或外传。

## 10. 关键文档索引

| 文件 | 内容 |
|---|---|
| `README.md` | 项目总览与快速开始 |
| `STATUS.md` | 项目当前状态（poster 冻结、diagram2ppt 活跃、omnimatte 暂停） |
| `BUILD-PLAN.md` | 原始 MVP 切片计划（poster 飞轮） |
| `CLAUDE.md` | 远程计算环境、不可协商操作规则 |
| `ir/README.md` | IR v1 设计契约与校验命令 |
| `docs/positioning-20260610.md` | 战略 pivot：从 RGBA 图层到语义原生结构 |
| `docs/diagram2ppt-progress.md` | diagram2ppt v1→v3 演进史 |
| `work/diagram2ppt/STATUS.md` | v2/v3 当前状态与分工 |
| `work/diagram2ppt/DEFECTS.md` | `framework.png` 缺陷台账 |
| `work/diagram2ppt/v2/README.md` | v2 稳定基线说明 |
| `work/diagram2ppt/v3/README.md` | v3 agentic 管线说明 |

---

**一句话**：i2e 当前（Current）是一个以 IR v1 为承重墙、以“修正飞轮”为护城河的语义原生结构管线；当前最值得保护的资产是 `work/diagram2ppt/v2/` 的稳定交付能力，最前沿的不确定地带是 `v3` 的端到端收敛。**最终目标（Target）**：把它演进成 Visual Design Decompiler——不是把图片变成文件，而是把视觉结果重新变成可编辑、可审计、可跨格式导出的设计源结构。
