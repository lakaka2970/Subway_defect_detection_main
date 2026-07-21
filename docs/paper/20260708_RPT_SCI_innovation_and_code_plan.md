# 地铁接触网超高清缺陷检测项目：SCI 创新主线、RPT 深化与代码改造方案

> 项目对象：1.27 亿像素级地铁接触网图像，原图约 `13000 × 9800`。  
> 工程约束：车载端与地面端均要求单张图像推理时间 `≤ 10 s`，并达到 `Recall ≥ 90%`、`Precision ≥ 90%`。  
> 源码基础：当前项目已有 YOLO11、EMA/SimAM、两阶段推理、SmartSlicer、WBF 融合、原生分辨率 crop、合成缺陷、难负样本挖掘、阈值校准和 FastAPI 服务。  
> 推荐论文路线：不再主打“改进 YOLO11 注意力模块”，而是主打“超高清工业图像中的分辨率保持训练、预算约束推理与风险校准”。

---

## 1. 结论先行

### 1.1 删除原“创新点 3：部件状态一致性联合学习”

原建议中的“缺陷检测 + 部件状态一致性联合学习”不建议作为当前论文核心创新点，主要原因如下：

1. **标注成本高**：需要新增部件状态、拓扑关系或正常/异常状态标签，而当前项目主要已有的是缺陷框标注。
2. **短期风险大**：若状态分支效果不稳定，容易拖累主任务；论文审稿中也可能被质疑任务定义复杂但证据不足。
3. **与当前代码耦合弱**：现有源码更成熟的部分是切片、原生 crop、难负样本、阈值校准和两阶段部署，而不是部件状态建模。

因此，应将论文核心转向更契合现有工程基础、可实验验证、可落地交付的方向：

> **分辨率保持训练 RPT + 10 秒预算约束自适应推理 + 类别风险校准。**

---

## 2. 推荐 SCI 论文主线

### 2.1 中文主线

**面向 1.27 亿像素地铁接触网图像的分辨率保持训练与预算约束小缺陷检测方法。**

### 2.2 英文主线

**Resolution-Preserving Training with Budget-Constrained Adaptive Inference for Ultra-High-Resolution Metro Catenary Defect Detection.**

### 2.3 核心问题定义

地铁接触网图像具有三个特点：

1. **分辨率极高**：单图约 `13000 × 9800`，约 `127 MP`。
2. **缺陷极小**：螺母缺失、螺母松动、销钉缺口等缺陷往往只占原图极小区域。
3. **部署约束强**：车载端和地面端都要求单张图像 `≤ 10 s`，且 `Recall/Precision ≥ 90%`。

如果将原图直接 resize 到 `1024`，横向缩放比例约为：

```text
1024 / 13000 ≈ 0.0788
```

即原图中：

```text
20 px 缺陷 → 约 1.6 px
40 px 缺陷 → 约 3.2 px
60 px 缺陷 → 约 4.7 px
```

经过 YOLO 的 stride-8、stride-16 特征下采样后，许多微小缺陷会退化为几乎不可学习的弱响应点。这是当前项目最值得写成论文的核心矛盾：

> **整图 resize 保证速度，但破坏缺陷像素尺度；全图原生切片保留尺度，但计算量过大；项目需要在 10 秒预算内保留微小缺陷可学习性。**

---

## 3. 近三年研究背景与差异化定位

### 3.1 相关研究趋势

近三年小目标检测和高分辨率检测主要有以下趋势：

| 方向 | 代表工作/趋势 | 对本项目的启发 |
|---|---|---|
| 切片辅助推理 | SAHI、ASAHI、GOIS | 切片已不是新概念，必须提出更明确的问题约束和评价协议 |
| 自适应切片 | ASAHI、GOIS | 固定切片会导致冗余计算，应向预算约束、自适应 tile 调度发展 |
| 小目标综述 | 2024–2025 小目标检测综述 | 小目标面临低像素占比、上下文弱、类别不平衡、边缘部署等问题 |
| 接触网缺陷检测 | 2025 接触网综述、MSIM-YOLOv11m 等 | 单纯改 YOLO 模块已拥挤，工程可落地的系统级方法更有区分度 |
| 合成缺陷 | DefectFill、TF-IDG 等 | 合成缺陷可作为辅助，但不宜作为主创新，除非有强质量筛选机制 |

### 3.2 与已有切片方法的差异

如果论文只写“把大图切成 1024/1280 crop 再检测”，会被认为类似 SAHI/ASAHI/GOIS，创新性不足。

本项目应强调差异：

| 已有方法 | 主要目标 | 本项目应强调的差异 |
|---|---|---|
| SAHI | 通用高分辨率小目标切片推理与微调 | 本项目是超高清工业图像，目标是原图缺陷像素尺度守恒和 10 秒工程预算 |
| ASAHI | 自适应切片数量，减少冗余计算 | 本项目不仅控制切片数量，还要保证 Recall/Precision ≥90% |
| GOIS | 动态两阶段切片，提升 tiny object AP/AR | 本项目要结合接触网 ROI、风险优先级、车载/地面双模式部署 |
| 普通 YOLO 改进 | 模块替换、注意力、多尺度融合 | 本项目主线不是“换模块”，而是“训练—推理—校准一致的工程检测协议” |

### 3.3 推荐论文定位

论文不应定位为：

```text
基于 YOLO11-EMA-SimAM 的地铁接触网缺陷检测方法
```

而应定位为：

```text
一种面向 127MP 超高清地铁接触网图像的分辨率保持训练、预算约束推理与风险校准检测框架。
```

这样更容易避开“YOLO 加模块”同质化问题。

---

## 4. 最终建议的创新点体系

建议最终保留 3 个主创新点 + 1 个辅助创新点。

---

## 4.1 创新点一：PSA-RPT 像素尺度对齐的分辨率保持训练协议

### 4.1.1 创新点名称

**PSA-RPT：Pixel-Scale Aligned Resolution-Preserving Training**  
中文：**像素尺度对齐的分辨率保持训练协议**

### 4.1.2 核心思想

不是简单“切图训练”，而是让训练样本中的缺陷像素尺度、缺陷位置分布、上下文范围与推理阶段保持一致。

RPT 应包含三个层次：

1. **像素尺度守恒**  
   缺陷在训练 crop 中保持原图像素尺寸，不被整图 resize 压缩。

2. **训练—推理一致**  
   训练 crop 的尺寸、重叠、缺陷位置、边界截断比例、负样本类型，应与推理 tile 分布一致。

3. **可学习性约束**  
   保证微小缺陷在输入图像与浅层特征图上仍具有足够像素表达。

### 4.1.3 与当前代码的关系

当前源码中 `scripts/generate_native_crops.py` 已经具备 RPT 雏形：

- 支持 `1024/1280` 原生分辨率 crop；
- 正样本围绕缺陷 bbox 生成；
- 支持中心、偏中心、近边缘、角落等位置去偏采样；
- 负样本来自无缺陷但视觉相似区域；
- 按 source image 做 train/val 划分，避免同源 crop 泄漏。

但还需要补强为论文级方法。

### 4.1.4 建议新增指标：PSDD

建议提出一个简单但有说服力的指标：

**PSDD：Pixel-Scale Distribution Divergence**  
中文：**像素尺度分布偏差**

用于衡量训练 crop 中缺陷 bbox 像素尺寸分布与原图/推理 tile 中真实缺陷尺寸分布的一致性。

可定义为：

```text
PSDD = D( S_train || S_infer )
```

其中：

```text
S_train = 训练 crop 中缺陷 bbox 像素宽高/面积分布
S_infer = 推理 tile 中缺陷 bbox 像素宽高/面积分布
D       = Wasserstein distance / KL divergence / JS divergence
```

实践中推荐用 Wasserstein distance，原因是对连续尺寸分布更直观。

### 4.1.5 建议新增实验

| 实验组 | 说明 | 预期结论 |
|---|---|---|
| Full resize 1024 | 原图直接缩放到 1024 训练 | 小缺陷 Recall 明显低 |
| Random crop | 随机原生 crop | 负样本多但正样本不足，Recall 不稳定 |
| Center crop | 缺陷中心 crop | Recall 提升，但模型学习到中心偏置 |
| RPT-debiased crop | 缺陷去中心化 crop | 边缘/角落缺陷召回更好 |
| RPT + hard negative | 加难负样本 | Precision 明显提升 |
| RPT + calibration | 加每类阈值 | 达到 Recall/Precision ≥90% 的 operating point |

---

## 4.2 创新点二：BCAT 预算约束自适应切片推理

### 4.2.1 创新点名称

**BCAT：Budget-Constrained Adaptive Tiling**  
中文：**预算约束自适应切片推理**

### 4.2.2 为什么需要这个创新点

以原图 `13000 × 9800` 为例，当前 `SmartSlicer` 默认参数为：

```text
slice_size = 1024
overlap = 0.15
stride = 870
```

估算切片数：

```text
cols = 15
rows = 12
total = 180 tiles
```

如果使用 `1280`：

```text
slice_size = 1280
overlap = 0.15
stride = 1088
cols = 12
rows = 9
total = 108 tiles
```

全图固定切片虽然保留了原生像素，但会带来两个问题：

1. **大量 tile 是背景或低风险结构区域**，浪费计算。
2. **车载端/地面端均要求 ≤10 秒**，不能无限增加 tile 数。

因此应从“固定切片”升级为“预算约束的 tile 调度”。

### 4.2.3 核心思想

每张图不是处理全部 tile，而是在时间预算内优先处理高风险 tile。

可定义：

```text
tile_score = α * roi_score
           + β * structure_density
           + γ * boundary_risk
           + δ * historical_fn_risk
           - λ * compute_cost
```

其中：

| 项 | 含义 |
|---|---|
| `roi_score` | Stage 1 ROI 模型输出的结构区域置信度 |
| `structure_density` | tile 内接触网结构/金属部件密度 |
| `boundary_risk` | 目标可能跨 tile 边界的风险 |
| `historical_fn_risk` | 历史漏检高发区域或类别权重 |
| `compute_cost` | tile 尺寸、模型大小、设备耗时估计 |

然后在 `≤10s` 或 `≤K tiles` 的预算下选择 top-K tiles。

### 4.2.4 车载端与地面端参数建议

| 模式 | 推荐模型 | 推荐策略 | 目标 |
|---|---|---|---|
| 车载端 onboard | YOLO11n/s + TensorRT FP16 | 少 tile，高 Recall，快速筛查 | `≤10s`，Recall≥90% |
| 地面端 ground | YOLO11s/m + WBF/校准 | 复查疑似区域，高 Precision | `≤10s`，Precision≥90% |

### 4.2.5 与当前代码的关系

当前代码：

- `subway_defect/pipeline/slicer.py`：固定规则切片；
- `subway_defect/pipeline/two_stage.py`：ROI 相交即处理；
- `config/model/inference.yaml`：固定 `slice_size`、`overlap`、`downsample_ratio`。

建议改造：

- 新增 `BudgetTileScheduler`；
- 新增 tile 评分机制；
- 新增 `max_tiles`、`latency_budget_ms` 参数；
- `TwoStagePipeline` 不再直接 `list(self.slicer.roi_tiles(...))`，而是请求 scheduler 返回排序后的 tile 列表。

---

## 4.3 创新点三：EGRC 车载—地面双模式风险校准机制

### 4.3.1 创新点名称

**EGRC：Edge-Ground Risk Calibration**  
中文：**车载—地面双模式风险校准机制**

### 4.3.2 核心思想

车载端和地面端不应使用完全相同的 operating point。

车载端更重视 Recall：

```text
宁可多报疑似缺陷，也不能漏掉严重缺陷。
```

地面端更重视 Precision：

```text
对车载端疑似缺陷进行复核，降低误报和人工复核压力。
```

因此每类缺陷应有不同的阈值策略：

```text
thresholds_onboard[class]  = 满足 Recall ≥ 90% 的最低阈值
thresholds_ground[class]   = 满足 Precision ≥ 90% 的较高阈值
```

### 4.3.3 与当前代码的关系

当前源码已有 `scripts/calibrate_thresholds.py`，并且 `stage5_hard_negative.yaml` 已把“阈值校准”写进训练流程。问题是：

1. FastAPI 推理流程目前只接收一个全局 `confidenceThreshold`；
2. `TwoStagePipeline` 也只使用全局 `defect_conf`；
3. 每类阈值没有真正接入在线推理；
4. onboard / ground 没有分离阈值文件。

### 4.3.4 建议输出文件

建议校准脚本输出：

```text
data/calibrated_thresholds/
├── thresholds_onboard.json
├── thresholds_ground.json
├── pr_curves_per_class.json
├── calibration_summary.md
└── calibration_report.csv
```

示例结构：

```json
{
  "mode": "onboard",
  "target": {"recall": 0.90, "precision": 0.80},
  "thresholds": {
    "VHBNM": 0.31,
    "VHBNL": 0.36,
    "SVHBNM": 0.29,
    "SVHBNL": 0.34,
    "SVHTNL": 0.42,
    "CBHPM": 0.38,
    "CBVPM": 0.40
  }
}
```

```json
{
  "mode": "ground",
  "target": {"precision": 0.90, "recall": 0.90},
  "thresholds": {
    "VHBNM": 0.46,
    "VHBNL": 0.51,
    "SVHBNM": 0.44,
    "SVHBNL": 0.50,
    "SVHTNL": 0.55,
    "CBHPM": 0.49,
    "CBVPM": 0.52
  }
}
```

---

## 4.4 辅助创新点：RPT 兼容的稀有缺陷增强与难负样本闭环

### 4.4.1 创新点定位

不建议把“合成缺陷”作为第一创新点，因为扩散式缺陷生成和工业缺陷合成已经是近年热点，容易同质化。

但可以作为辅助创新：

> **所有合成缺陷必须服务于 RPT：生成后仍保持原生缺陷像素尺度，并通过质量筛选进入训练。**

### 4.4.2 当前代码基础

当前项目已有：

- `scripts/generate_synthetic_defects.py`
- `scripts/generate_defect_copy_paste.py`
- `subway_defect/synthetic/defect_synthesis.py`
- `subway_defect/augmentations/defect_copy_paste.py`
- `scripts/collect_hard_negatives.py`

这些可形成一个闭环：

```text
少样本类别 → 合成缺陷 → 质量筛选 → RPT crop → 训练 → 误报收集 → 难负样本再训练 → 阈值校准
```

### 4.4.3 建议新增质量筛选

合成样本进入训练前，应增加过滤：

| 过滤项 | 目的 |
|---|---|
| bbox 像素尺寸范围 | 防止生成缺陷太大/太小，破坏 RPT 分布 |
| 边界伪影检测 | 防止 Copy-Paste/Inpainting 边界不自然 |
| 局部亮度/颜色连续性 | 防止缺陷与背景融合差 |
| 结构位置合法性 | 防止螺母、销钉出现在不合理位置 |
| 模型自检置信度 | 使用已有模型检查生成样本是否可识别 |

---

## 5. 项目代码具体修改方向

---

## 5.1 P0：先修复会影响运行和复现的问题

### 5.1.1 修复 `SmartSlicer.stride` 只读属性问题

当前 `fastapi_server.py` 中存在：

```python
state.pipeline.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))
```

但 `SmartSlicer.stride` 是只读 `@property`，没有 setter，会导致请求时报错。

#### 推荐修改方式 A：增加 `update()` 方法

修改文件：

```text
subway_defect/pipeline/slicer.py
```

新增：

```python
def update(self, slice_size: int | None = None, overlap: float | None = None):
    if slice_size is not None:
        self.slice_size = int(slice_size)
    if overlap is not None:
        self.overlap = float(overlap)
    self._stride = int(self.slice_size * (1 - self.overlap))
```

然后修改：

```text
subway_defect/deployment/fastapi_server.py
```

将：

```python
state.pipeline.slicer.slice_size = req.slice_size
state.pipeline.slicer.overlap = req.slice_overlap
state.pipeline.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))
```

改为：

```python
state.pipeline.slicer.update(req.slice_size, req.slice_overlap)
```

### 5.1.2 统一类别数

当前存在类别数混乱：

| 位置 | 当前情况 |
|---|---|
| `classes.py` | 16 类完整字典，7 类训练子集 |
| `stage2-5 yaml` | `nc: 7` |
| 部分 model yaml | `nc: 18` |
| `SPECIFICATION.md` | 提到 18 类 |

建议统一为：

```text
论文实验：7 类已标注缺陷
工程字典：16 类完整缺陷体系
未来扩展：18 类不要写入当前论文，除非有完整标注和训练配置
```

#### 建议代码修改

新增：

```text
subway_defect/classes.py
```

中的显式函数：

```python
def get_training_class_names() -> list[str]:
    return TRAIN_CLASSES


def get_training_nc() -> int:
    return TRAIN_NC


def get_full_class_names() -> list[str]:
    return DEFECT_CLASSES
```

并修改所有脚本中硬编码的 7 类列表：

```text
scripts/collect_hard_negatives.py
scripts/generate_native_crops.py
scripts/create_defect_data_yaml.py
scripts/fix_classes_txt.py
```

统一从 `subway_defect.classes` 导入。

### 5.1.3 修正 model YAML 的 `nc`

建议新增两个模型配置目录：

```text
subway_defect/models/train7/
subway_defect/models/full16/
```

其中：

```text
train7/yolo11s-EMA-SimAM.yaml     nc: 7
train7/yolo11m-EMA-SimAM.yaml     nc: 7
full16/yolo11s-EMA-SimAM.yaml     nc: 16
full16/yolo11m-EMA-SimAM.yaml     nc: 16
```

当前论文和训练默认使用 `train7/`。

---

## 5.2 P1：把 RPT 从脚本升级为正式模块

### 5.2.1 新增模块目录

建议新增：

```text
subway_defect/rpt/
├── __init__.py
├── crop_generator.py
├── pixel_scale.py
├── sampling_policy.py
├── split_policy.py
└── report.py
```

### 5.2.2 功能拆分

| 文件 | 功能 |
|---|---|
| `crop_generator.py` | 从原图和 YOLO 标签生成原生分辨率 crop |
| `pixel_scale.py` | 统计 bbox 像素尺寸分布，计算 PSDD |
| `sampling_policy.py` | 中心/偏中心/边缘/角落采样策略 |
| `split_policy.py` | 按 source image / line / date 划分数据集 |
| `report.py` | 输出 RPT 数据集报告 |

### 5.2.3 RPT 元数据文件

建议每个 crop 生成一条元数据，保存为：

```text
data/subway_crops/rpt_metadata.jsonl
```

每行示例：

```json
{
  "crop_id": "IMG_001_x3200_y4800_s1280",
  "source_image": "IMG_001.jpg",
  "source_w": 13000,
  "source_h": 9800,
  "crop_x0": 3200,
  "crop_y0": 4800,
  "crop_size": 1280,
  "sampling_zone": "near_edge",
  "has_defect": true,
  "bbox_pixel_w": 36,
  "bbox_pixel_h": 28,
  "bbox_visibility": 0.92,
  "is_boundary_case": false,
  "split": "train"
}
```

这个文件对论文非常重要，因为可以支撑：

- 像素尺度统计；
- 位置分布统计；
- 边界样本统计；
- 同源图像泄漏检查；
- 消融实验复现。

### 5.2.4 `generate_native_crops.py` 修改方向

当前脚本已经较好，但建议补充：

1. 输出 `rpt_metadata.jsonl`；
2. 输出 `pixel_scale_report.csv`；
3. 输出 `position_zone_report.csv`；
4. 增加 `--target-pixel-min` 和 `--target-pixel-max`；
5. 增加 `--boundary-case-ratio`；
6. 增加 `--split-by source|line|date|scene`；
7. 增加 `--export-report data/subway_crops/rpt_report.md`。

建议命令：

```bash
python scripts/generate_native_crops.py \
  --src data/raw/images \
  --labels data/raw/labels \
  --output data/subway_crops_rpt1280 \
  --crop-size 1280 \
  --negatives-per-image 20 \
  --boundary-case-ratio 0.25 \
  --split-by source \
  --export-metadata \
  --export-report
```

---

## 5.3 P1：新增像素尺度分析脚本

### 5.3.1 新增脚本

```text
scripts/analyze_pixel_scale.py
```

### 5.3.2 功能

输入：

```text
原图目录 + 标签目录 + crop metadata
```

输出：

```text
outputs/pixel_scale/
├── bbox_pixel_distribution.csv
├── psdd_summary.json
├── psdd_report.md
├── bbox_area_hist.png
├── bbox_min_side_hist.png
└── train_infer_scale_alignment.png
```

### 5.3.3 关键统计

应统计：

```text
bbox_w_px
bbox_h_px
bbox_area_px
bbox_min_side_px
bbox_max_side_px
bbox_area_ratio
feature_p3_size = bbox_min_side_px / 8
feature_p4_size = bbox_min_side_px / 16
```

论文中非常有用的表格：

| 训练方式 | 缺陷最小边中位数 | P3 特征尺寸 | P4 特征尺寸 | Recall |
|---|---:|---:|---:|---:|
| full resize 1024 | 2.1 px | 0.26 cell | 0.13 cell | 低 |
| RPT 1024 | 28.5 px | 3.56 cells | 1.78 cells | 高 |
| RPT 1280 | 28.5 px | 3.56 cells | 1.78 cells | 高 |

注意：RPT 1024 与 RPT 1280 中缺陷像素尺寸不变，差异主要在上下文大小和 batch/速度。

---

## 5.4 P1：将 SmartSlicer 升级为可调度切片系统

### 5.4.1 新增文件

```text
subway_defect/pipeline/tile_scheduler.py
```

### 5.4.2 推荐类设计

```python
from dataclasses import dataclass

@dataclass
class TileCandidate:
    tile: object
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    score_parts: dict


class BudgetTileScheduler:
    def __init__(
        self,
        max_tiles: int = 80,
        latency_budget_ms: int = 10000,
        min_roi_overlap: float = 0.05,
        boundary_boost: float = 0.2,
    ):
        ...

    def rank_tiles(self, image, all_tiles, roi_boxes=None, roi_scores=None):
        ...

    def select(self, candidates):
        return candidates[:self.max_tiles]
```

### 5.4.3 修改 `SmartSlicer`

当前 `roi_tiles()` 只返回与 ROI 相交的 tile。建议新增：

```python
def candidate_tiles(self, img, roi_boxes=None):
    """Yield all candidate tiles with coordinates and ROI overlap stats."""
```

返回结构包含：

```text
row, col, x0, y0, x1, y1, roi_overlap, boundary_flag
```

这样 scheduler 才能评分，而不是简单过滤。

### 5.4.4 修改 `TwoStagePipeline`

当前逻辑：

```python
if roi_boxes is not None and len(roi_boxes) > 0:
    tiles = list(self.slicer.roi_tiles(image, roi_boxes))
else:
    tiles = list(self.slicer.iter_tiles(image))
```

建议改为：

```python
candidates = self.slicer.candidate_tiles(image, roi_boxes)
tiles = self.tile_scheduler.select(
    self.tile_scheduler.rank_tiles(
        image=image,
        candidates=candidates,
        roi_boxes=roi_boxes,
        roi_scores=roi_scores,
    )
)
```

同时返回：

```text
candidate_slices
selected_slices
tile_budget_used
tile_score_summary
```

用于论文实验与接口调试。

---

## 5.5 P2：把每类阈值接入推理服务

### 5.5.1 新增模块

```text
subway_defect/pipeline/thresholding.py
```

### 5.5.2 推荐类设计

```python
class PerClassThresholds:
    def __init__(self, path=None, default=0.4):
        self.default = default
        self.thresholds = {}
        if path:
            self.load(path)

    def load(self, path):
        ...

    def get(self, class_name: str, class_id: int | None = None) -> float:
        ...

    def keep(self, detection: dict) -> bool:
        class_name = detection.get("class_name")
        conf = detection.get("confidence", 0.0)
        return conf >= self.get(class_name)
```

### 5.5.3 修改 `TwoStagePipeline`

新增参数：

```python
thresholds: Optional[PerClassThresholds] = None
```

在 `_detect_defects()` 中，模型推理时可以使用较低全局阈值，例如：

```python
model_conf = min(per_class_thresholds.values()) if thresholds else self.defect_conf
```

然后对每个 detection 做二次过滤：

```python
if self.thresholds is not None and not self.thresholds.keep(det):
    continue
```

### 5.5.4 修改 FastAPI

新增配置项：

```yaml
threshold_file_onboard: data/calibrated_thresholds/thresholds_onboard.json
threshold_file_ground: data/calibrated_thresholds/thresholds_ground.json
```

`load_models()` 中根据 mode 加载不同阈值。

接口返回中建议增加：

```json
{
  "thresholdMode": "onboard",
  "thresholdSource": "data/calibrated_thresholds/thresholds_onboard.json"
}
```

---

## 5.6 P2：完善 10 秒预算评测工具

### 5.6.1 新增脚本

```text
scripts/benchmark_latency_budget.py
```

### 5.6.2 功能

对不同策略评测：

```text
full resize 1024
fixed tiling 1024
fixed tiling 1280
ROI-only tiling
BCAT max_tiles=40
BCAT max_tiles=60
BCAT max_tiles=80
BCAT max_tiles=100
ground WBF
TensorRT FP16
```

输出：

```text
outputs/benchmark/
├── latency_budget_report.md
├── latency_budget.csv
├── recall_precision_latency.csv
├── tile_count_distribution.csv
└── pr_latency_curve.png
```

### 5.6.3 论文重点指标

| 指标 | 说明 |
|---|---|
| `total_time_ms` | 单张总耗时 |
| `stage1_time_ms` | ROI 阶段耗时 |
| `stage2_time_ms` | 缺陷检测耗时 |
| `candidate_slices` | 候选 tile 数 |
| `selected_slices` | 实际推理 tile 数 |
| `Recall` | 总召回率 |
| `Precision` | 总精确率 |
| `Recall@Precision≥90%` | 工程约束下召回 |
| `Precision@Recall≥90%` | 工程约束下精度 |
| `FP/image` | 每张图平均误报数 |
| `FN/image` | 每张图平均漏检数 |

---

## 5.7 P2：完善 WBF 与地面端复核机制

当前 `WBFFusion` 可用于双模型融合，但建议从“全图双模型重复推理”改成“车载候选 + 地面复核”。

### 5.7.1 推荐流程

```text
1. onboard 模式：轻模型快速推理，输出疑似缺陷与不确定 tile；
2. ground 模式：只复查 onboard 疑似区域附近 tile；
3. 对重叠结果进行 WBF；
4. 使用 ground 阈值进行最终输出。
```

### 5.7.2 新增接口字段

请求：

```json
{
  "extraParams": {
    "candidateRegions": [...],
    "mode": "ground_review"
  }
}
```

响应：

```json
{
  "reviewedRegions": 12,
  "fusionMode": "wbf",
  "thresholdMode": "ground"
}
```

---

## 6. 建议实验章节设计

### 6.1 数据集设置

建议按真实应用场景划分数据，而不是随机 crop 划分：

| 划分方式 | 说明 |
|---|---|
| by source image | 最基本要求，避免同源 crop 泄漏 |
| by line/section | 更接近跨线路泛化 |
| by date/lighting | 检验不同光照、隧道环境、成像条件 |
| by device | 若有不同采集设备，可检验设备域泛化 |

### 6.2 主实验

| 方法 | 输入策略 | 推理策略 | 校准 | 目标 |
|---|---|---|---|---|
| YOLO11 baseline | full resize 1024 | 单图 | 无 | 基线 |
| YOLO11 + fixed tiling | RPT crop | 固定切片 | 无 | 验证分辨率保持 |
| YOLO11 + RPT | PSA-RPT | 固定切片 | 无 | 验证 RPT |
| YOLO11 + RPT + BCAT | PSA-RPT | 预算切片 | 无 | 验证 10 秒预算 |
| YOLO11 + RPT + BCAT + EGRC | PSA-RPT | 预算切片 | 每类阈值 | 完整方法 |
| Ground review | PSA-RPT | 车载候选复核 | ground 阈值 + WBF | 地面端性能 |

### 6.3 消融实验

| 消融项 | 预期说明 |
|---|---|
| 去掉 RPT，使用 full resize | 小缺陷 Recall 大幅下降 |
| RPT 只中心 crop | 边界/角落目标召回下降 |
| 去掉 hard negative | Precision 下降，误报增加 |
| 去掉 BCAT，固定全图切片 | 速度超预算或冗余过高 |
| 去掉 per-class threshold | 难以同时满足 Recall/Precision ≥90% |
| 只用 mAP 选模型 | 工程指标不稳定，PR operating point 不可控 |

### 6.4 推荐论文表格

#### 表 1：不同输入策略下缺陷像素尺度

| 输入策略 | 缺陷最小边中位数 | P3 cell 数 | P4 cell 数 | Recall |
|---|---:|---:|---:|---:|
| Resize-1024 | 待统计 | 待统计 | 待统计 | 待统计 |
| RPT-1024 | 待统计 | 待统计 | 待统计 | 待统计 |
| RPT-1280 | 待统计 | 待统计 | 待统计 | 待统计 |

#### 表 2：不同切片策略下速度与性能

| 策略 | 平均 tile 数 | 单图耗时 | Recall | Precision | FP/image |
|---|---:|---:|---:|---:|---:|
| Fixed 1024 | 180 | 待测 | 待测 | 待测 | 待测 |
| Fixed 1280 | 108 | 待测 | 待测 | 待测 | 待测 |
| ROI-only | 待测 | 待测 | 待测 | 待测 | 待测 |
| BCAT-60 | ≤60 | 待测 | 待测 | 待测 | 待测 |
| BCAT-80 | ≤80 | 待测 | 待测 | 待测 | 待测 |

#### 表 3：每类阈值校准结果

| 类别 | 默认阈值 | 校准阈值 | Recall | Precision | FP 减少比例 |
|---|---:|---:|---:|---:|---:|
| VHBNM | 0.40 | 待测 | 待测 | 待测 | 待测 |
| VHBNL | 0.40 | 待测 | 待测 | 待测 | 待测 |
| SVHBNM | 0.40 | 待测 | 待测 | 待测 | 待测 |
| SVHBNL | 0.40 | 待测 | 待测 | 待测 | 待测 |
| SVHTNL | 0.40 | 待测 | 待测 | 待测 | 待测 |
| CBHPM | 0.40 | 待测 | 待测 | 待测 | 待测 |
| CBVPM | 0.40 | 待测 | 待测 | 待测 | 待测 |

---

## 7. 推荐代码改造路线图

### 7.1 第一阶段：工程稳定性修复，1–2 天

| 优先级 | 任务 | 文件 |
|---|---|---|
| P0 | 修复 `SmartSlicer.stride` 只读 bug | `slicer.py`, `fastapi_server.py` |
| P0 | 统一 7/16/18 类配置 | `classes.py`, `models/*.yaml`, `config/train/*.yaml` |
| P0 | 补充 `train-pipeline` 命令入口 | `pyproject.toml` |
| P0 | 检查 README 中不存在的 tests/docs 描述 | `README.md` |
| P0 | FastAPI 加载 per-class threshold 的占位配置 | `fastapi_server.py`, `inference.yaml` |

### 7.2 第二阶段：RPT 论文核心模块，3–5 天

| 优先级 | 任务 | 文件 |
|---|---|---|
| P1 | 抽象 RPT 模块 | `subway_defect/rpt/*` |
| P1 | 输出 crop 元数据 | `generate_native_crops.py` |
| P1 | 像素尺度统计与 PSDD | `scripts/analyze_pixel_scale.py` |
| P1 | RPT 数据集报告 | `subway_defect/rpt/report.py` |
| P1 | 增加边界样本比例控制 | `sampling_policy.py` |

### 7.3 第三阶段：预算约束推理，4–7 天

| 优先级 | 任务 | 文件 |
|---|---|---|
| P1 | 新增 `BudgetTileScheduler` | `pipeline/tile_scheduler.py` |
| P1 | 修改 `TwoStagePipeline` 使用 scheduler | `pipeline/two_stage.py` |
| P1 | 在接口返回 tile 统计 | `fastapi_server.py` |
| P2 | 增加 latency benchmark | `scripts/benchmark_latency_budget.py` |
| P2 | 增加 TensorRT FP16 benchmark | `deployment/export_tensorrt.py` |

### 7.4 第四阶段：风险校准与地面复核，3–5 天

| 优先级 | 任务 | 文件 |
|---|---|---|
| P2 | per-class threshold 模块 | `pipeline/thresholding.py` |
| P2 | onboard/ground 双阈值文件 | `calibrate_thresholds.py` |
| P2 | 接入 FastAPI 推理 | `fastapi_server.py` |
| P2 | ground review 模式 | `two_stage.py`, `fastapi_server.py` |
| P2 | WBF 只作用于候选复核区域 | `wbf_fusion.py` |

---

## 8. 推荐新增/修改配置

### 8.1 `config/model/inference.yaml`

建议新增：

```yaml
# ── 预算约束切片 ─────────────────────────────────────────────────
tiling_mode: budget_adaptive   # fixed | roi_only | budget_adaptive
max_tiles_onboard: 60
max_tiles_ground: 90
latency_budget_ms: 10000
min_roi_overlap: 0.05
boundary_boost: 0.2

# ── 阈值校准 ─────────────────────────────────────────────────────
threshold_mode: per_class      # global | per_class
threshold_file_onboard: data/calibrated_thresholds/thresholds_onboard.json
threshold_file_ground: data/calibrated_thresholds/thresholds_ground.json

# ── RPT 一致性 ───────────────────────────────────────────────────
rpt_train_crop_size: 1280
rpt_infer_slice_size: 1280
rpt_overlap: 0.15
```

### 8.2 `config/train/pretrain/stage3_main_training.yaml`

建议补充注释，不一定改参数：

```yaml
# RPT 主训练：保持原生缺陷像素尺度，不使用 YOLO 默认大范围 multi-scale
# 注意：multi_scale 不建议开启 ±50%，否则会破坏 RPT 像素尺度对齐假设
imgsz: 1280
multi_scale: false
```

当前 `stage3_main_training.yaml` 没有开启 `multi_scale`，这是合理的。旧版 `full.yaml` 中有 `multi_scale: 0.5`，建议论文实验不要使用旧版配置。

---

## 9. 论文摘要可用版本

### 中文摘要草稿

针对地铁接触网巡检中 1.27 亿像素级超高清图像的小缺陷检测问题，直接缩放整图会导致螺母缺失、螺母松动和销钉缺口等微小缺陷退化为少数像素，难以被深度检测器有效学习；而全图原生分辨率密集切片又难以满足车载端和地面端单张图像 10 秒内处理的工程约束。为此，本文提出一种分辨率保持训练与预算约束自适应推理框架。首先，设计像素尺度对齐的分辨率保持训练协议，在原生分辨率下构造正样本、难负样本和边界样本，使训练样本中的缺陷像素尺度与推理阶段保持一致。其次，提出预算约束自适应切片策略，根据结构 ROI、边界风险和计算预算对候选 tile 进行排序，在有限推理预算内优先处理高风险区域。最后，设计车载—地面双模式类别风险校准机制，通过每类阈值搜索在不同部署模式下平衡召回率与精确率。实验将从像素尺度分布、检测精度、误报率、漏检率和单图推理时间等方面验证所提方法在超高清地铁接触网缺陷检测中的有效性。

### 英文摘要草稿

Ultra-high-resolution metro catenary inspection images, typically reaching 127 megapixels, pose a unique challenge for small defect detection. Directly resizing such images to conventional detector inputs severely compresses tiny defects such as missing nuts, loose nuts, and pin gaps into only a few pixels, making them difficult to learn. In contrast, exhaustive native-resolution tiling preserves defect details but violates the strict latency requirement of onboard and ground-side inspection systems. To address this problem, we propose a resolution-preserving training and budget-constrained adaptive inference framework for ultra-high-resolution metro catenary defect detection. First, a pixel-scale aligned resolution-preserving training protocol is developed to construct positive, hard-negative, and boundary-aware crops at native resolution, aligning the defect pixel-scale distribution between training and inference. Second, a budget-constrained adaptive tiling strategy ranks candidate tiles according to structural ROI confidence, boundary risk, and computational cost, enabling high-risk regions to be processed within a fixed latency budget. Finally, an edge-ground risk calibration mechanism is introduced to optimize class-specific operating points for onboard high-recall screening and ground-side high-precision verification. The proposed framework is evaluated in terms of pixel-scale preservation, recall, precision, false reports per image, and end-to-end inference latency.

---

## 10. 推荐论文标题

### 中文标题

1. **面向超高清地铁接触网图像的分辨率保持训练与预算约束小缺陷检测方法**
2. **面向 1.27 亿像素接触网巡检图像的分辨率保持小目标缺陷检测框架**
3. **融合像素尺度对齐训练与自适应切片推理的地铁接触网缺陷检测方法**

### 英文标题

1. **Resolution-Preserving Training with Budget-Constrained Adaptive Inference for Ultra-High-Resolution Metro Catenary Defect Detection**
2. **Pixel-Scale Aligned Training and Adaptive Tiling for Tiny Defect Detection in 127-Megapixel Metro Catenary Images**
3. **A Latency-Aware Resolution-Preserving Framework for Ultra-High-Resolution Catenary Defect Detection**

---

## 11. 最终建议采用的论文贡献写法

建议论文贡献写成三条，简洁有力：

1. **提出 PSA-RPT 分辨率保持训练协议。**  
   针对超高清接触网图像中微小缺陷在整图缩放后退化为亚像素目标的问题，构建像素尺度对齐的原生分辨率训练样本，并通过去中心化采样和边界样本生成提高训练—推理一致性。

2. **提出 BCAT 预算约束自适应切片推理策略。**  
   针对 127MP 图像全图固定切片计算冗余、难以满足 10 秒部署约束的问题，基于 ROI 置信度、结构密度、边界风险和计算成本对 tile 进行排序，在有限 tile 预算内优先处理高风险区域。

3. **提出 EGRC 车载—地面双模式风险校准机制。**  
   针对车载端高召回和地面端高精度的不同需求，通过每类阈值校准和候选区域复核，在 Recall/Precision ≥90% 的约束下减少误报与漏检。

辅助贡献：

4. **构建 RPT 兼容的稀有缺陷增强与难负样本闭环。**  
   将合成缺陷、难负样本挖掘和阈值校准纳入统一训练流程，提升少样本类别检测稳定性和复杂背景下的 Precision。

---

## 12. 最小可行实施版本

如果时间有限，建议先做最小可行版本：

```text
V1 必做：
1. 修复 slicer update bug；
2. 统一 7 类训练配置；
3. generate_native_crops.py 输出 metadata；
4. analyze_pixel_scale.py 输出 PSDD 和尺度报告；
5. TwoStagePipeline 增加 max_tiles；
6. calibrate_thresholds.py 输出每类阈值；
7. FastAPI 接入每类阈值。
```

论文实验先完成：

```text
1. resize 1024 vs RPT 1280；
2. fixed tiling vs max_tiles budget tiling；
3. global threshold vs per-class threshold；
4. no hard negative vs hard negative；
5. onboard vs ground operating point。
```

只要这 5 组实验结果清晰，就能支撑一篇完整的 SCI 应用型论文。

---

## 13. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| RPT 被认为只是切片训练 | 审稿人说类似 SAHI | 强调 PSDD、训练—推理一致性、10 秒预算约束 |
| BCAT 导致漏检 | 低分 tile 中有缺陷 | 设置兜底策略：低密度区域随机抽样 + 高风险类扩大 ROI |
| 每类阈值牺牲 Recall | Precision 提升但漏检变多 | 分 onboard/ground 两套阈值，车载端阈值更低 |
| 合成缺陷伪影 | Precision 下降 | 增加质量过滤，只作为辅助增强 |
| 数据划分泄漏 | crop 同源图进入 train/val | 强制 source image / line / date 分组划分 |
| 10 秒难达成 | tile 数过多 | TensorRT FP16 + max_tiles + ROI-only + early exit |

---

## 14. 一句话总结

本项目最值得发表的创新不是“改一个 YOLO11 模块”，而是：

> **针对 1.27 亿像素地铁接触网图像，在不压缩微小缺陷像素尺度的前提下，通过 RPT 保证可学习性，通过预算约束切片保证 10 秒部署，通过类别风险校准确保 Recall/Precision ≥90%，形成一个训练—推理—部署一致的超高清工业缺陷检测框架。**
