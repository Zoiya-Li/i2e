# i2e 项目思考过程与进度

## 2026-06-09 ~ 06-10

---

## 1. 问题起源：9 种方法都不满意

### 用户的核心诉求

将 AI 生成的视觉内容（海报、框架图等）分解为**可编辑的图层/元素**。

### 尝试过的 9 种方法

| # | 方法 | 文件 | 做了什么 | 问题 |
|---|------|------|----------|------|
| 1 | SAM3 全分割 | `regen_sam3.py` | SAM3 单次概念分割 + SD-1.5 amodal completion | 需要远程 GPU；mask 边界减法产生薄 artifact |
| 2 | Omnimatte 迭代剥离 | `omnimatte.py` | 分离物体+效果（烟雾/阴影），12 层 | 远程 GPU；clean plate 有鬼影残留 |
| 3 | Clean Plate 生成 | `plate.py` | Best-of-N SD fill + GroundingDINO critic 过滤 confabulation | 仍有 confabulated 对象；需要 6 seeds |
| 4 | Faithful 零退化提取 | `faithful_layers.py` | 原始像素保留，22 文本 + 9 图形层 | 文字是光栅裁切不是矢量；背景用 cv2 Telea 不够好 |
| 5 | VLM 叠加检测 | `detect_overlay.py` | VLM 枚举 + SAM3 精确 mask | 依赖 SAM3 远程 GPU |
| 6 | 字体匹配 | `font_match.py` | 系统 font IoU 匹配 | 匹配不准（Comic Sans for PEPPERMINT） |
| 7 | 编辑演示 | `edit_demo.py` | 4 种编辑操作证明可编辑 | 只是演示 |
| 8 | SVG 导出 | `export_faithful.py` | 结构化 SVG + data attributes | 9.9MB；字体匹配仍不准 |
| 9 | Gemini 生成式分解 | `gen_decompose/` | Gemini 识别实体 → 逐个生成 → rembg 去背景 → 组装 | 生成的不忠实原图；风格漂移 |

### 三个核心痛点（用户确认）

1. **生成式不够忠实** — Gemini 画的跟原图不一样
2. **提取式不够干净** — SAM 抠图边缘脏、inpaint confabulate
3. **编辑体验不够好** — 文字无法重排、字体不准、图层太碎

### 根本矛盾

```
提取式（SAM 从原图抠）= 忠实原图但边缘脏、背景难修
生成式（Gemini 重新画）= 干净但不忠实、风格漂移
```

两种方法都在"拆像素为光栅图层"——但用户真正需要的是**结构化的可编辑元素**。

---

## 2. 关键转折：框架图 → PPT 的启示

### 用户的灵魂之问

> "最简单的一个例子，我拿到了一个 gpt 画的框架图，我想把他变成 ppt 形式让我可编辑，这个项目能不能解决？"

答案是 **不能**。因为：

| 用户需要的 | 现有方法给的 |
|-----------|------------|
| 方框是可编辑矩形（改颜色/大小/文字） | SAM 抠出来的是光栅 PNG |
| 箭头是可编辑连线 | Gemini 重新画的是一张图片 |
| 文字是可编辑文本框 | OCR 裁切的是像素裁片 |

**核心范式错误：所有方法输出的都是"光栅图层"，而不是"结构化的矢量元素"。**

### 提出的三种新模式

#### 方案 A：残差自修复（Residual-Guided Repair）
- 原图 → SAM3 抠图 → 初步重组 → 残差图 → 针对性修复
- 本质还是提取式，对严重遮挡帮助有限

#### 方案 B：类型感知分解（Type-Aware Decomposition）⭐ 推荐
- 不同类型元素用不同最优方法：照片用 alpha matte、文字用 OCR+矢量、背景用生成、效果用参数化
- 可行性最高，每个子模块有现成工具

#### 方案 C：设计意图逆向（Design Intent Reverse Engineering）
- 不拆像素，还原设计决策
- 输出最接近"可编辑设计文件"但落地难度最大

### 最终方向：先用框架图验证"VLM 理解 → 结构化输出 → 原生格式生成"范式

框架图是最简 MVP：
- 比 poster 简单（没有照片、没有 alpha、没有 inpaint）
- 输出格式明确（PPT）
- 成功可衡量（PPT 是否还原了原图结构）

如果这个范式对框架图有效，同样的架构可以扩展到海报。

---

## 3. 实现：diagram2ppt 模块

### 架构

```
image → Gemini VLM 分析 → 结构化 JSON → python-pptx → .pptx
```

三个文件：
- `analyze.py` — VLM prompt + 多格式 JSON 解析
- `build_ppt.py` — JSON → 原生 PPT 形状
- `run.py` — CLI 入口

### 技术决策

1. **单轮分析**（不用两轮 follow-up）—— prompt 缩短到 ~350 字符，`send_keys` 能在几秒内打完
2. **复用 GeminiWebDriver** —— 连接已有 Chrome CDP 9222
3. **原生 PPT 形状** —— 每个元素都是 PowerPoint Shape/Connector，不是光栅
4. **鲁棒解析器** —— 处理 Gemini 的各种输出格式（标准 JSON、带前缀的 JSON、Gemini 自定义 schema）

### 遇到的技术问题与解决

#### 问题 1：Gemini 回显 prompt 示例数据
- **现象**：第一次 DIAGRAM_PROMPT 包含示例 `{"elements": [{"text": "Data Processing"...}]}`，Gemini 直接把示例原样返回
- **解决**：去掉 prompt 中的具体示例数据，改为抽象描述格式

#### 问题 2：Gemini 使用自己的 schema
- **现象**：Gemini 用了 `element_type`, `shape`, `color` 等字段，而不是我们要求的 `type`, `fill`, `border_color`
- **解决**：`_normalize_gemini_elements()` 函数做 schema 映射（shape name → type、color name → hex 等）

#### 问题 3：JSON 前缀 "JSON\n"
- **现象**：Gemini 在 JSON 前加了 `JSON\n` 标记，导致 `json.loads()` 失败
- **解决**：用 `re.search(r'\{\s*"elements"', raw)` 替代 `raw.find('{"elements"')` 处理空格/换行

#### 问题 4：两轮 analyze 的 send_keys 长文本超时
- **现象**：第一轮 ack 后，第二轮 ~600 字符的 DIAGRAM_PROMPT 通过 `send_keys` 发送超时（180s）
- **尝试**：
  - `execCommand('insertText')` —— 不触发 Gemini React 状态更新，消息没发出去
  - `followup_text` 改用 execCommand —— 同样失败
- **最终解决**：缩短 prompt 到 ~350 字符，用单轮模式（图片 + prompt 一起发）

#### 问题 5：多轮 _wait_for_text 返回第一轮回复
- **现象**：两轮模式下，`rfind("Gemini said")` 找到的是第一轮的标记，第二轮还没回复就返回了
- **解决**：`_wait_for_text` 新增 `min_markers` 参数，要求页面上至少有 N 个 "Gemini said" 标记才开始检查稳定性

#### 问题 6：Gemini 免费额度耗尽
- **现象**：经过 poster 管线的大量测试后，Gemini 返回 "You're running low on usage"
- **解决**：等待额度恢复

---

## 4. 当前状态（2026-06-10）

### 已完成

- ✅ `diagram2ppt` 模块完整实现（analyze + build_ppt + run）
- ✅ 端到端管线验证通过
  - 输入：test_diagram_20260609_230407.png (1024×559)
  - Gemini 分析：7 shapes + 6 connectors
  - 输出：test_diagram_v2.pptx (13.3" × 7.3")
- ✅ 输出为原生 PPT 形状（矩形、文本框、连接器），全部可编辑
- ✅ `driver.py` 改进：
  - `analyze()` 支持 `single_turn` 模式
  - `_wait_for_text()` 支持 `min_markers` 参数防止多轮提前返回
  - `followup_text()` 实现优化（虽然最终没用上）

### 已知局限

1. **位置精度**：Gemini 估算的位置是近似值（fraction），不保证像素精确
2. **颜色精度**：Gemini 估算的 hex 颜色可能有偏差
3. **字体大小**：Gemini 返回的 font_size 是估算值
4. **复杂图表**：只测试了简单的流程图，复杂图表（嵌套、曲线箭头、渐变填充）未验证
5. **Gemini 依赖**：需要 Chrome CDP 连接 + Gemini 免费额度

### 下一步方向

#### 短期（diagram2ppt 完善）
1. 用户拿自己的框架图测试，验证泛化性
2. 位置精度优化（可能需要第二轮 VLM 精确定位）
3. 支持更多 shape 类型（三角形、星形等）
4. 支持背景颜色/渐变

#### 中期（范式扩展到海报）
用同样的 "VLM 理解 → 类型感知分解 → 原生格式生成" 范式处理海报：
- 产品照片 → 从原图提取（alpha matte，不是 binary mask）
- 文字 → OCR + 矢量重排
- 背景 → 风格分析 + 重新生成
- 效果（烟雾/阴影）→ 参数化
- 装饰图形 → 矢量化

#### 长期（工具化）
- 支持更多输入（截图、照片、手绘草图）
- 支持更多输出（PSD、HTML+CSS、Figma）
- 批量处理
- 本地 VLM 替代 Gemini（减少依赖）

---

## 7. v2：迭代式类型感知管线（2026-06-10 实现并跑通）

战略背景见 `docs/positioning-20260610.md`（Lovart Edit Elements 已商品化像素拆解；
新轴 = 语义原生结构；用户实测 Lovart 在 framework.png 上拆不了）。

### 架构（work/diagram2ppt/v2/）

```
vlm.py        SiliconFlow Qwen3.5-397B API client（替代 Gemini 网页自动化，撑得起迭代）
parsing.py    鲁棒 JSON 提取 + max_tokens 截断挽救
ir.py         diagram IR：px bbox + status(native/demoted) + tries/residual
render.py     PIL 渲染代理（diff 用，非交付物）
diff.py       per-element 残差(1-SSIM) + 未覆盖墨水聚类(coverage)
loop.py       global → render-diff → refine(crop)/identify(missing)/demote → checkpoint
build_pptx.py 原生 shape/textbox/connector + raster_crop 贴图（忠实兜底）
run.py        CLI: python -m work.diagram2ppt.v2.run framework.png -o out/
```

核心保证：**收敛不了的元素降级为原图忠实裁切**——保真有底，迭代只负责抬
native fraction（北极星）。

### framework.png 首跑（138 次 VLM 调用，3 轮）

round1: 132 el, native 0.71/0.50, cov 0.752 → round2: cov 0.932 →
final: 180 el, **native 0.48/0.31, cov 0.967**；PPTX = 55 文本框 + 22 形状 +
10 连接线 + 93 裁切图。对照图 `v2_out/compare.png`：结构可辨识。

### 测试

`tests/test_diagram_v2.py` 9 项（解析变体+截断挽救、残差信号、coverage、
收敛/refine/降级/识别回退、PPTX 构建），全离线 MockVLM；全套件 40/40。

### 下一步（按 native fraction 提升量排序）

1. OCR 专家接管文本几何（~70% 降级是 text；本机 rapidocr 装不上 → 远程或换源）
2. 文本渲染代理与 PPT 字体对齐（虚增残差）
3. 公式/统计图专家（LaTeX-OCR、轴反提取）
4. 嵌套面板两级分解（panel 内局部坐标系）

---

## 5. 文件清单

```
work/diagram2ppt/
├── __init__.py              # 模块描述
├── analyze.py               # Gemini VLM 分析 → 结构化 JSON
├── build_ppt.py             # JSON → python-pptx 原生 PPT
├── run.py                   # CLI: --image 或 --json
├── test_diagram_20260609_230407.png   # 测试用 Gemini 生成的框架图
├── test_diagram.diagram.json          # 解析后的结构化 JSON
├── test_diagram_v2.pptx               # 端到端生成的 PPT（v2，成功）
└── _raw_response.txt                  # Gemini 原始响应（调试用）

work/gen_decompose/          # 之前的 Gemini 生成式分解（poster）
├── driver.py                # ★ 改进：single_turn、min_markers
├── identify.py              # 海报实体识别
├── generate_layers.py       # 实体生成
├── assemble.py              # 图层组装
└── output/                  # 海报分解结果
```

---

## 6. 核心认知总结

1. **范式比实现重要**：9 种方法都困在"拆像素"范式里。正确的范式是"理解视觉结构 → 重建为原生可编辑元素"。

2. **从最简 case 开始验证**：框架图 → PPT 是比海报 → PSD 简单 10 倍的问题。先验证范式可行，再扩展复杂度。

3. **VLM 的局限性**：
   - 会回显 prompt 中的示例
   - 不一定遵循指定的 JSON schema
   - 位置/颜色估算有误差
   - 免费额度有限
   → 需要鲁棒的解析和 schema 归一化

4. **Selenium 自动化的脆弱性**：
   - `send_keys` 长文本慢且不可靠
   - `execCommand` 不触发 React 状态
   - 多轮对话的 "Gemini said" 计数需要 `min_markers`
   - Gemini UI 的 usage limit 不可控
   → 更好的路径：直接用 Gemini API（但需要 API key）
