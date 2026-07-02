# i2e 定位修正 —— 2026-06-10

**触发**：联网核查发现 Lovart 已把"任意图片一键拆层"产品化（Edit Elements，背后是
OmniPSD，arXiv 2512.09247，NUS Show Lab × Lovart 合作）；Qwen-Image-Layered（Apache 2.0）
开源了可变层数 RGBA 分解。**"我押了图片拆解"作为差异化已死** —— 像素级拆解正在商品化，
应验了 STATUS.md 自己的预言。

## 新定位（用户确认）

> 不是"我押了拆解"，而是"我押了拆解的目标格式是**带语义的原生结构**，不是 RGBA 图层"。
> 文字是文本框、方框是形状、箭头是关系、logo 绑定品牌库源文件、每个元素带出身和修正
> 记录——输出物离开工具后在 PPT/Figma/品牌工作流里仍然是活的。

## 支撑论点

### 1. 编辑成本结构（用户一手验证，2026-06-10）

实测：我们的风油精海报在 Lovart 网站上很难编辑——**每次编辑都要动用 AI**。
这是 OmniPSD 范式的结构性必然（图层是扩散产物，编辑要回推理）：

| | Lovart（AI 中介编辑） | i2e（原生结构编辑） |
|---|---|---|
| 延迟 | 秒~分钟/次 | 毫秒 |
| 成本 | 按 credits 计费 | 免费 |
| 确定性 | 非确定，可能漂移 | 完全确定 |
| 离线 | 不行 | 可以 |
| 撤销 | 受限 | 无限 undo |

产品论点：**AI 只在拆解那一刻用一次，之后的编辑永远不需要 AI。**
设计师一张稿子几十上百次微调，按次付费的非确定性编辑在该频率下不可用。

### 2. 确定性 = 品牌合规硬保证

移动原生文本框不可能"顺便"改字体/logo 颜色；AI 重渲染每次都有 off-brand 漂移风险。

### 3. 飞轮只属于结构化范式

直接操作产生干净 (predicted→corrected) 训练对；AI 中介编辑产生的是 prompt+新像素，
Lovart 收集不到结构化修正信号。

### 4. Counter-positioning

Lovart 的激励是把用户锁在 canvas 里继续消耗 credits；"拆完导出原生格式然后离开"
对它是反战略的。这比"先发优势"硬。

## 必须钉死的反例（OPEN）

Lovart 宣称可导出分层 PSD/SVG。**若导出 PSD 中文字是真 TypeLayer（带字体、可直接改字）**，
则"免 AI 编辑"论点失效，差异化只剩语义深度（形状/关系/品牌绑定/出身）。
→ 实验：海报过 Lovart Edit Elements → 导出 PSD → `psd-tools` 检查图层类型与字体元数据。

## 已知的同轴竞品（浅层，但存在）

- CopySlides（图片→原生 PPT 对象，多 OCR racing，1000+ 字体识别）
- ImageToDrawio（flowchart→PPT connector 重建）

它们缺：语义类型系统、品牌绑定、修正飞轮、跨格式 IR。跟 Lovart 比它们是 feature，
跟 i2e 比它们缺的恰好是 IR 契约里最重的部分。

## 验证结果（2026-06-10，用户实测）

**Lovart 在论文级技术框架图上拆解失败。** 测试图 = `i2e/framework.png`（CATE-CI Auditor，
含流程框+箭头、数学公式、坐标曲线图、柱状图、3D 散点面、图例卡片）。用户在 Lovart canvas
上未能获得可用的拆分（"我都没找到他能拆分的点"→ 实测做不了）。低分辨率现场截图存
`work/diagram2ppt/lovart_test_screenshot_20260610.png`。

→ **moat 假设成立的方向**：把这类图做成真正可编辑 = 我们的护城河。
用户原则："我们不是冲着能做去的，我们要的是做好。"

### "做好"的操作化定义（三条硬标准）

1. **重组保真**：所有层/元素叠回去 ≈ 原图（faithful-layers 的教训：不丢东西是底线，
   任何收敛不了的元素降级为忠实像素裁切，保真永远 100%）。
2. **原生化比例**：尽可能多的元素是原生可编辑对象（shape/connector/文本框/公式/chart），
   而非光栅。**北极星指标 = 固定保真下的 native fraction，随版本单调上升。**
3. **编辑存活**：拖动框→箭头跟随；改文字→不破坏布局；改数据→图表重绘。

### v2 架构：类型感知 + 迭代收敛（用户判断：需要大 pipeline + 迭代，正确）

```
Phase 0  全局理解   VLM → panel/group 层级 + 元素枚举 + 类型标注
Phase 1  类型路由   每类元素 → 专属专家：
   文本   → OCR(几何) + VLM(内容/角色) → 文本框
   公式   → 公式检测 → LaTeX-OCR → OMML/公式对象
   框/箭头 → 轮廓/线条拟合(cv2) + VLM 关系标注 → shape + connector(from→to)
   统计图 → 轴 OCR + 曲线追踪 → 原生 chart；失败则忠实裁切
   3D/插画 → alpha matte 忠实裁切（不强行原生化）
Phase 2  组装渲染   结构化 IR(扩展 formula/chart/connector 类型) → python-pptx
Phase 3  迭代修复环  render(IR) vs 原图 → 分区域残差(SSIM) → 定位高残差元素 →
   VLM critic 针对性重提取 → 收敛 or 降级忠实裁切 → needs_review 进编辑器 → 修正飞轮
```

注：Phase 3 = 残差自修复（旧方案 A）的复活——当年用在"拆像素"上是错位，
用在"结构化输出 render-diff"上是正确位置。verify/check.py、ocr/、faithful_layers、
IR schema、editor 修正捕获全部可复用。Gemini 网页自动化撑不起迭代环（额度/脆弱），
需切 API（Gemini API 或 Qwen-VL）。

## v2 骨架首跑结果（2026-06-10，framework.png 真实跑通）

实现：`work/diagram2ppt/v2/`（vlm/parsing/ir/render/diff/loop/build_pptx/run，
SiliconFlow Qwen3.5-397B API），测试 `tests/test_diagram_v2.py` 9 项全离线，
全套件 40/40 绿。输出 `work/diagram2ppt/v2_out/`。

| round | elements | native(count/area) | coverage | 备注 |
|---|---|---|---|---|
| 1 | 132 | 0.71 / 0.50 | 0.752 | 全局 78 + identify 补 20 |
| 2 | 142 | 0.72 / 0.51 | 0.932 | refine 收敛部分文本 |
| 3→final | 180 | **0.48 / 0.31** | **0.967** | 顽固元素降级忠实裁切 |

138 次 VLM 调用。PPTX：180 个对象 = 55 文本框 + 22 原生形状 + 10 连接线 + 93 忠实裁切图片。
重组对照 `v2_out/compare.png`：整体结构可辨识，公式/统计图/3D 面按设计走裁切。

**诚实读数**：native fraction 最终 0.48 是因为 fidelity gate 把残差 >0.45 的全部踢成裁切
——指标如实暴露当前 VLM-only 提取的精度上限，这正是它的作用。主要误差源（按量排序）：
文本几何/字号（≈70% 的降级是 text）→ 需要 OCR 专家；嵌套面板内小元素 bbox 漂移；
PIL 渲染代理与 PPT 渲染的字体差异虚增文本残差。改进路径与类型路由计划一致：
OCR 接管文本几何、公式/图表专家接管对应裁切区，native fraction 应单调上升。

修过的工程坑（都有测试/兜底）：max_tokens 截断 JSON→挽救解析器；Qwen-VL 拒收 <28px
crop→最短边放大到 64；SSIM 7×7 窗口在细条 bbox 崩溃→均值差分回退；VLM 返回整数 id；
每轮 checkpoint 落盘防崩溃丢进度。

### v2.1 评分修正（同日，用户反馈"很多东西是截图根本不能编辑"）

降级分析定位三个评分错误：30 个 text 因 PIL 字体差异被 SSIM 冤杀（面积 7.7%）、
11 个容器框因嵌套子元素墨水被算进残差（面积 14.3%）、降级容器与原生子元素双重渲染。
修复：① text 改 edge-F1 评分（k=5/T=0.62，实测平反 24/30、误放 2/55；**已知盲区**：
位移进繁忙区域的文本 ~0.4 过门，靠上游 refine+coverage 兜）；② 容器壳评分
（children_of 排除子元素区域，naive 0.33→shell 0.0003）；③ faithful_crop 给覆盖
原生子元素的裁切打环形中值色补丁。对已有 IR 零调用重过门：
**native 0.48→0.61（count）/ 0.31→0.37（area），可编辑对象 87→110，截图 93→70**。
产物 `v2_out/diagram_v21.pptx`、`compare_v21.png`。测试 11 项，全套件 42/42。
剩余 70 张截图：40 按设计（3D/统计图/公式，等专家）+ 26 真几何错误 + 4 未识别。

### v2.2：OCR 专家上线（同日，远程 A800 GPU2 环境）

类型路由的第一个专家落地：RapidOCR 跑在 A800 docker（`/home/lzy/AAAI_2026/i2e/ocr/`，
pylibs 双层 overlay 隔离，headless opencv 遮蔽 Qt 版），`work/remote.py` base64 传输。
`v2/ocr_snap.py`：内容相似度匹配（VLM 内容为准、OCR 几何接管，支持多行 union），
高置信未匹配行补成新文本元素（raster 区域内跳过）。framework.png 实测：89 行 OCR，
snap 36/41 文本 + 补回 23 行漏提取。

**北极星序列（native fraction count/area）**：
v2 `0.48/0.31` → v2.1 `0.61/0.37` → **v2.2 `0.72/0.65`**；coverage 0.972；
截图 93→70→**38**（其中 25 是 3D/图表/公式按设计）；VLM 调用 138→46（OCR 砍掉一半
refine 工作量）。产物 `work/diagram2ppt/v22_out/`（PPTX = 90 shapes + 8 connectors
+ 38 pictures）。测试 12 项，全套件 43/43。

v3 已知问题：① OCR 补入行与形状自带文字重复（dedup 只挡了 raster 区域，没挡
shape.text）；② 个别胶囊框 refine 后漂移、connector 端点错；③ 9 个 oval/rounded_rect
仍降级；④ 渲染代理字体差异仍虚增部分文本残差。

### v2.3：真实 PPT 验收暴露三病并修复（2026-06-12，用户实机截图反馈）

用户在 PowerPoint 里发现渲染代理掩盖的三类问题，全部定位修复：
① **巨型文字**（"Feat ure Engi neeri ng"、竖排 R E T A I）—— `_fit_font_px` 字号
回退只看高度不看宽度，窄高框算出巨字后 PPT 强制换行；修复 = 宽高双约束
（0.55 字宽系数）+ 单行文本关 word_wrap + AUTO_SIZE.NONE，显式 font_size 只封顶
不抬高；② **重复文字** —— `dedupe_text()` 通用去重（形状自带标签 > 独立文本 >
OCR 补行），回溯删了 21 个；snap_text 增 `_claimed_by_element` 防再犯；
③ **幻觉连接线** —— `connector_ink_fraction` 沿线采样原图墨水，<0.35 删除
（实测：真箭头偏移≈0.45，幻觉放射线≈0.10），剪 2 条。
v2.3 = `v22_out/diagram_v23.pptx`：69 shapes + 6 connectors + 38 pics，
native 面积 0.645 持平（count 0.72→0.66 是删掉 21 个垃圾重复文本的结果，不是退步）。
测试 15 项，全套件 46/46。教训：**渲染代理验收 ≠ 真实格式验收，PPT 排版引擎
（autofit/wrap/字体）必须单独过目**。
遗留（v3）：胶囊 3/4 标签 bbox 漂移（edge-F1 盲区实例）、Feature Engineering 等
框文字错位、25 张图表/公式/3D 截图等专家模块。

### v2.4：公式 + 图表专家上线（2026-06-12，回应"难点全是贴图"）

`v2/experts.py`，13 次 VLM 调用：
- **公式专家**：math-regex 筛候选（乱码文本也行，VLM 从像素重新转写）→ LaTeX →
  远程 latex2mathml+mathml2omml（pylibs3）→ `<a14:m>` OMML 注入 textbox。
  **10/10 公式成为可双击编辑的 PowerPoint 原生公式**（主公式、β/γ 梯度定义、
  胶囊内 T~X→p̂ 等）；无 OMML 时回退 LaTeX 文本。
- **图表专家**：VLM 抽 categories/series/颜色 → python-pptx 原生 chart（数据
  VLM 读取=近似值，ext.approx 标记）。2 个转换成功，1 个 spec 解析失败留作裁切。
- **sanitize**：清掉 2 个退化细条裁切（coverage blob 残渣）。

v2.4 = `v22_out/diagram_v24.pptx`：70 shapes + **10 公式** + **2 charts** +
6 connectors + 33 pics；native 0.70/0.696（**面积新高**，序列 0.31→0.37→0.65→0.70）。
诚实缺陷：theta_text 转写丢失（"θ≈0°"→"0"）；chart_q0 的 kind 可能被误读为 bar；
chart 数值是 VLM 目测。剩余 33 张图：3D 渲染×3（格式上限）、图标 pictogram×12、
胶囊缩略散点×5 等。测试 17 项，全套件 48/48。

### v3：禁止截图模式（2026-06-12，用户政策决定"把所有东西都画出来"）

**政策转折**：编辑性压倒像素保真——deck 里零图片，全部是对象。这有意识地
**反转了海报时代的 faithful-pixels 教义**（设计师要原始像素；PPT 用户要能改的对象。
两种产品契约，不是矛盾）。`v2/vectorize.py`：
- 失败形状/文本 → 强制原生（ext.forced，接受视觉误差）×11
- 图标 → VLM 分类 → MSO autoshape（database→CAN/gear→GEAR_6/warning→三角/
  document→FOLDED_CORNER）或字形 textbox ×18
- 散点缩略图/3D → **cv2 确定性提取**：blob→逐点小圆（位置/半径/颜色还原），
  大轮廓→freeform 填充面 ×4
结果：`diagram_v3.pptx` = 81 shapes + 284 dots + 18 icons + 2 charts + 6 connectors
+ 10 公式，**pictures=0**，native 1.0/1.0（by construction），18 次 VLM 调用。
测试 19 项，全套件 50/50。

**诚实代价**（compare_v3.png 可见）：3D 曲面成了淡色多边形面（轮廓色提取偏白）、
大 3D 图的点没提出来（密集散点融进轮廓 mask，dotclouds=0）、部分图标字形呆板。
v3.1 打磨点：密集散点的分水岭分离、轮廓主色采样修正、图标 shape 库扩充。

### v3.1→v3.3：渲染自检闭环（2026-06-12，用户："自己截图看，迭代改"）

**新工作循环（固化为 `v2/snapshot.py`）**：构建 → AppleScript 驱动**真 PowerPoint**
导 PDF（其 PNG 导出坏/沙箱）→ PyMuPDF 转 PNG → 自己看 → 改 → 重渲染。
QuickLook/PIL 代理都会撒谎（不渲染 OMML/custGeom、autofit 不同）；本机装了
MS PowerPoint = ground truth。沙箱路径 `~/Library/Containers/com.microsoft.Powerpoint/
Data/Documents/`。

4 轮渲染→看→改的修复（每轮有真渲染截图证据，snapshots/）：
1. 巨型 β/∑ = 胶囊缩略散点被面积阈值误判成 icon → **dots-first 路由**（≥6 点=dotcloud）
2. 巨型公式 = OMML 注入未设字号 → 每个 m:r 注入 a:rPr sz（schema 序 m:rPr→a:rPr→m:t）
3. 图表轴标签 18pt 巨字 → tick/legend 7pt
4. 3D 轮廓尖刺 = bbox 内其他原生元素污染掩膜 → **提取前抠掉子元素区域**（同壳评分思想）
   + 模糊大核闭运算 + 均匀重采样（非粗化 eps）
5. 深色团 = strong-pixel 取色过深 → 回归整体中位色（描边供对比度）；
   小图(<20000px²)不做轮廓只留散点
另：真 PowerPoint 验证 **OMML 公式确实渲染为真公式**、freeform 几何正确、
XML 中无巨型字号（QuickLook 之前的"巨字"半数是它自己的渲染缺陷）。
**最终交付 `v22_out/diagram_final.pptx`（=v3.3）**：0 图片，81 shapes+402 dots+
12 icons+2 charts+10 公式+6 connectors。真渲染快照 `snapshots/ppt_v33_true.png`。
遗留：3D 轮廓形状仍粗糙（风格化示意级）、胶囊 3 公式 bbox 横跨错误、
Alignment Score 标签漂移、RETAIN 列文字挤压。

## 生死验证实验（升级版三方对比）

同两张图（风油精海报 + work/diagram2ppt/test_diagram）分别过：
Lovart Edit Elements / Qwen-Image-Layered / i2e 管线，回答：

1. **海报上我们输多少？**（大概率输；输的幅度决定是否彻底放弃像素侧）
2. **框架图/结构化文档上他们是否真的给不出原生可编辑对象？**（若是，全部赌注压这里）
3. **Lovart 导出的 PSD 文字层是 type layer 还是光栅？**（决定编辑成本论点存废）

## 来源

- OmniPSD: https://arxiv.org/abs/2512.09247 · https://github.com/showlab/OmniPSD
- Lovart Edit Elements: https://news.aibase.com/news/22743 · https://www.agiyes.com/ainews/lovart-edit-elements/
- Qwen-Image-Layered: https://github.com/QwenLM/Qwen-Image-Layered · https://arxiv.org/html/2512.15603v1
- CopySlides: https://copyslides.com/convert-image-to-editable-powerpoint
- ImageToDrawio: https://imagetodrawio.com/image-to-ppt
