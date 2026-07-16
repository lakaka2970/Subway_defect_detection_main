这两个问题非常关键，而且你已经进入**“模型设计 → 系统设计 → 标注体系设计”**的阶段了，这一步如果处理不好，后面 Transformer + Cascade 都会“越做越乱”。

我给你**实事求是 + 工程可落地 + 不会破坏约束**的完整回答。

---

# 🧠 问题1：改 label system 会不会影响工程硬约束？

## ❗结论先说：

> ❌ 不能直接改（会破坏工程闭环）
> ✅ 但可以“分层改”，这是正确做法

---

# 📌 1.1 你的硬约束本质是什么？

你的系统硬约束是：

```text
必须输出 7类接触网缺陷识别结果
（VHBNM / VHBNL / SVHBNM / SVHBNL / SVHTNL / CBHPM / CBVPM）
```

👉 这是：

> ✔ “交付层 label system（不可变）”

---

# 📌 1.2 为什么不能直接改 label system？

如果你直接改成：

```text
normal / loose / missing / anomaly
```

会导致：

### ❌ 工程后果：

| 问题       | 影响       |
| -------- | -------- |
| 输出类别不一致  | 无法对接现有系统 |
| 评估体系失效   | mAP无法对比  |
| 数据无法复用   | 旧标注全部失效  |
| 训练-部署不一致 | 工程崩溃     |

---

# 📌 1.3 正确做法（关键）

## ✔ 双层 label system（工业标准做法）

```text
Level 1: 7-class detection（工程输出）
Level 2: state reasoning（Transformer内部）
```

---

## 🧠 映射关系（核心设计）

```text
YOLO输出（必须保持）：

VHBNM
VHBNL
SVHBNM
SVHBNL
SVHTNL
CBHPM
CBVPM
```

↓

Transformer内部：

```text
normal
loose_state
missing_state
structural_anomaly
```

---

## ✔ 关键结论：

> ❗你不是“改 label system”，而是“增加一个 reasoning layer”

---

# 🧠 问题2：数据集要不要重新标注？

## ❗结论先说：

> ❌ 不需要全量重标
> ⚠️ 但必须做“增量标注 + 结构修正”

---

# 📌 2.1 当前标注体系的问题

你当前规则是：

> “框内包含缺陷 + 物理支撑结构特征”

## ❗问题：

这会导致：

### 1️⃣ label语义混合

```text
框 = defect + structure context
```

👉 对 YOLO 是好的
👉 对 Transformer 是干扰

---

### 2️⃣ 状态不可分

例如：

* loose vs normal
* missing vs shadow

👉 模型无法学习“状态差异”

---

### 3️⃣ FP Level 2 的根源

你现在最大误检：

> ❗模型在学“结构”，不是“异常状态”

---

# 📌 2.2 是否需要重标？

## ✔ 分三种情况：

### ✔（1）必须重标（少量）

```text
SVHBNM
SVHBNL
VHBNL
CBVPM
```

👉 这些类建议做：

> ✔ 标注 audit（不是重标，是修正）

---

### ✔（2）无需重标（大部分）

```text
VHBNM
CBHPM
SVHTNL
```

👉 保留即可

---

### ✔（3）新增标注（关键）

你现在必须新增：

## ⭐ Normal / Background / Negative samples

```text
normal bolts
normal pins
normal base structures
reflection areas
shadow structures
blur regions
```

---

# 📌 2.3 标注标准（升级版）

你现在标准：

> ✔ “框内包含缺陷 + 结构”

需要升级为：

---

## ✔ YOLO检测标注标准（保持不变）

```text
规则：
- 框必须覆盖缺陷主体
- 可包含少量结构context
- 不要过度扩展到整结构
```

---

## ❗新增标准（关键）

### ✔ Transformer训练用 crop 标注标准

```text
目标：状态判断，不是定位
```

### 标准如下：

| 类型      | 标注原则       |
| ------- | ---------- |
| normal  | 完整结构，无异常   |
| missing | 应存在但不存在    |
| loose   | 存在但位置/角度异常 |
| anomaly | 不确定结构异常    |

---

## ❗关键变化：

> YOLO标注 = 空间问题
> Transformer标注 = 状态问题

---

# 🧠 核心认知升级（非常重要）

你现在系统其实是：

```text
错误认知：
YOLO = 完整解决方案

正确认知：
YOLO = proposal generator
Transformer = reasoning module
```

---

# 🚀 下一步 To Do List（工程级拆解）

我帮你拆成：

* 👨‍🔧 人工必须做
* 🤖 vibe coding可自动化做

---

# 🧑‍🔧 A. 人工必须做（关键路径）

---

## A1. FP数据体系构建（最优先）

### 做什么：

* 从 YOLO inference 收集：

  * FP
  * FN
  * borderline samples

### 分类：

```text
FP_LEVEL_1
FP_LEVEL_2
FP_LEVEL_3
```

---

## A2. SVHBNM / SVHBNL 标注审计

### 做什么：

* 随机抽 200–300 张

检查：

```text
✔ 是否误标
✔ 框是否过大
✔ 是否混入normal
✔ 是否类别边界不清
```

---

## A3. Normal dataset 构建（必须新增）

### 收集：

```text
正常螺母
正常销钉
完整结构
阴影结构
反光结构
模糊结构
```

---

## A4. Boundary dataset 构建（最重要）

```text
normal ↔ loose transition
normal ↔ missing transition
```

---

## A5. label体系确认（最终冻结）

确认：

```text
7-class detection label（不变）
state label（新增）
```

---

# 🤖 B. Vibe Coding可开发任务（工程核心）

---

## B1. FP Mining自动系统

### 功能：

```text
输入：YOLO inference结果
输出：FP分类数据集
```

### 模块：

```text
fp_detector.py
fp_classifier.py
fp_exporter.py
```

---

## B2. YOLO → Transformer dataset builder

### 功能：

自动生成：

```text
224x224 crop
+ state label
+ normal/defect balance
```

---

## B3. Cascade inference pipeline

```text
YOLO → candidate boxes
→ crop generator
→ transformer inference
→ fusion output
```

---

## B4. score fusion engine

```python
final_score = 0.6 * yolo + 0.4 * transformer
```

---

## B5. hard negative miner

```text
自动筛选：
- high confidence FP
- near threshold samples
```

---

## B6. evaluation split system（必须）

避免数据泄漏：

```text
split by source image (NOT by crop)
```

---

# 🚀 最终开发路线图（推荐执行顺序）

---

## Phase 1（数据优先）

```text
✔ FP mining系统
✔ normal dataset构建
✔ SVHBNM audit
```

---

## Phase 2（YOLO优化）

```text
✔ YOLO high recall retrain
✔ HN加入
```

---

## Phase 3（Transformer引入）

```text
✔ state classifier训练
✔ FP level 2/3重点学习
```

---

## Phase 4（系统融合）

```text
✔ cascade inference
✔ score fusion
✔ threshold calibration
```

---

# 🧠 最终总结（一句话）

> ❗你不是在“改模型”，而是在从 detection system 升级到 reasoning system

---