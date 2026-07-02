# i2e IR v1 — 设计契约

> IR 是整个飞轮的**承重墙**。Node ②(拆层→IR)写它,编辑器读它,Node ⑤(修正捕获)往它的 `corrections` 里追加。
> 文件:`ir-v1.schema.json`(规范) · `example-fengyoujing-poster.ir.json`(基于真实海报的样例)

## 1. 它是什么

一张"死图"被拆解后的**结构化场景图**:有哪些元素、各自什么属性、什么层级、模型有多大把握、用户改了什么。任意来源的图 → 同一个 IR → 任意输出格式。换输入只改前端抽取器,换输出只改后端渲染器,**中间这堵墙不动**。

## 2. 五条设计原则(为什么长这样)

1. **模型无关 / 格式中立**:`source.generator` 只是备注,IR 不绑定任何生成器或导出格式。
2. **可序列化、第三方可读**:纯 JSON,有 `$id` 和 `ir_version`,任何人能读能写能校验。这是它未来沉淀成"标准"的前提。
3. **可版本演进**:`ir_version` 锁死;破坏性变更必须升版本,迁移器据此分流。
4. **每个元素自带"出身"**:`extraction`(confidence + model_version + method)。没有出身,后面的修正就无法归因到具体模型——飞轮断电。
5. **前向兼容**:每个对象留 `ext` 扩展位;但顶层和已定义对象 `additionalProperties:false`,保证标准本体收紧不腐化。

## 3. 三个节点怎么挂在墙上

```
Node ②  ──写──▶  elements[]            (每个元素带 extraction.confidence、needs_review)
编辑器  ──读/改─▶  elements[].text.content 等(只暴露 editable[] 列出的字段)
Node ⑤  ──追加─▶  corrections[]         (每改一处 = 一条 predicted→corrected)
```

## 4. 修正契约(Node ⑤ —— 护城河的本体,最重要)

**铁律:用户的修正不是额外标注,是他为了交活本来就要做的编辑;我们零成本把它捕获成训练数据。**

每条 `Correction` = 一个字段从"模型预测值"到"用户最终值"的一次改动:

| 字段 | 作用 |
|---|---|
| `field_path` + `kind` | 把这条训练样本**路由到对应的子模型**(`text.content`→OCR、`font`→字体识别、`type`→分类、`mask_refine`→分割、`inpaint_redo`→背景重建…) |
| `predicted` / `corrected` | 监督对:(模型错的, 正确的) |
| `confidence_at_prediction` | 校准检查:被改的,是不是当初低置信的那些? |
| `model_version` | 哪个模型错了,信号才能精确回流 |

样例里 4 条修正演示了全部要点:OCR 改字(`沁谅→沁凉`)、字体纠正、logo 匹配到品牌库、徽标被重分类(vector→logo)。**这就是喂养 Node ⑥/⑦ 的原料。**

> 北极星指标由此而来:在固定测试图集上,**平均每张图的 `corrections` 条数必须随时间下降**。降 = 飞轮在转;不降 = 这只是个工具。

## 5. v1 范围 & 故意不做的事

**做**:元素类型 `text / raster / background / logo / vector / group`;三类高价值编辑挂在 `editable[]` 上 = 改文案(`text.content`)、本地化(`text.content`+`lang`)、改尺寸(靠 `bbox`+`nbox`)。

**故意不做**(守住"最小"):
- 不做约束式自动重排——`nbox` 先只作为 resize 的基准,真正的 layout solver 推后。
- 不做复杂元素替换、不做实时重训——修正先批量离线回流。
- 多端样式、物理单位、动画——全部推后。

## 6. 怎么验证

```bash
# 任选其一
npx ajv-cli validate -s ir-v1.schema.json -d example-fengyoujing-poster.ir.json --spec=draft2020
python -c "import json,jsonschema; jsonschema.Draft202012Validator(json.load(open('ir-v1.schema.json'))).validate(json.load(open('example-fengyoujing-poster.ir.json'))); print('IR OK')"
```
