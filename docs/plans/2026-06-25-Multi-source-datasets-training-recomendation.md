可以采用，而且**建议采用**。更准确地说，不是“拿公开数据集替代自制数据”，而是做：

```text
COCO 通用检测预训练
→ 公开工业缺陷/小目标数据集二次预训练
→ 自制接触网 ROI/crop 数据集领域适配
→ 自制验证集阈值校准与部署验证
```

这样做能缓解你当前自制数据少、类别覆盖不足、P2 小目标头初始化不充分的问题。项目当前自制数据规模大约是原始 399 张训练图，加增强后约 1880 张，验证集约 101 张，且当前只覆盖 7 类缺陷；这类规模直接训练复杂小目标模型确实偏少。 项目又有单 RTX 4090、单张 ≤10 秒、Recall/Precision ≥90% 的硬约束，所以更适合“迁移学习 + 小目标结构改造 + 强验证闭环”，而不是盲目堆大模型。

---

# 1. 是否推荐公开数据集预训练？

**推荐，但要分层使用。**

公开数据集和接触网缺陷之间通常存在域差异：钢板缺陷、PCB 缺陷、绝缘子缺陷、交通标志小目标，都只能学到一部分能力。它们分别适合提供：

| 数据来源         | 能学到什么               | 对本项目的价值      |
| ------------ | ------------------- | ------------ |
| COCO         | 通用边缘、纹理、目标检测能力      | 基础初始化        |
| 工业表面缺陷数据     | 裂纹、孔洞、斑点、擦伤、异常纹理    | 缺陷纹理表征       |
| PCB 缺陷数据     | 规则结构中的微小异常、缺失、短路、孔洞 | 最接近“规则结构中异常” |
| 绝缘子/电力设施缺陷数据 | 供电设施、绝缘子破损、闪络等      | 语义域更接近       |
| 小目标高分辨率数据    | 小目标定位、P2/P3 头训练     | 解决微小缺陷可见性    |
| 自制接触网 crop   | 真实目标、真实结构、真实相机分布    | 最终决定效果       |

因此我建议使用 **“公开多源预训练 + 自制数据精调”**，而不是只找一个公开数据集。

---

# 2. 推荐数据集优先级

## 第一优先级：工业缺陷检测数据集

### 1）NEU-DET / NEU Surface Defect

推荐程度：**高**

NEU 表面缺陷数据集包含 1800 张灰度图，每类 300 张，覆盖 rolled-in scale、patches、crazing、pitted surface、inclusion、scratches 等 6 类钢材表面缺陷。([faculty.neu.edu.cn][1]) 它和接触网不是同一语义域，但都是金属/工业纹理异常，适合让 backbone 和 neck 学“缺陷纹理”。

用法：

```text
COCO → NEU-DET → 自制接触网 crop
```

建议不要把 NEU 类别硬映射成你的 7 类接触网缺陷，而是作为中间域预训练。

---

### 2）GC10-DET

推荐程度：**高**

GC10-DET 是金属表面缺陷检测数据集，原论文将其作为 metallic surface defect detection benchmark，数据集中包含多个金属缺陷类别；Dataset Ninja 统计版本显示其包含约 2300 张图、3563 个标注目标、10 类缺陷。([MDPI][2]) ([Dataset Ninja][3])

它比 NEU-DET 类别更多，适合作为第二个工业缺陷预训练集。

用法：

```text
COCO → NEU-DET + GC10-DET 混合预训练 → 自制接触网
```

---

### 3）DeepPCB

推荐程度：**很高**

DeepPCB 包含 1500 对图像，每对由无缺陷模板图和对齐后的测试图组成，并标注 6 类 PCB 缺陷：open、short、mousebite、spur、pin hole、spurious copper。([GitHub][4])

这个数据集对你项目特别有价值，因为 PCB 的结构是规则重复的，缺陷通常也是规则结构中的局部缺失、断裂、孔洞或异常点，这一点和“螺栓缺失、开口销缺失、局部松动”非常接近。

推荐用法：

```text
COCO → DeepPCB → 自制接触网 crop
```

或者作为混合预训练主力：

```text
COCO → DeepPCB + NEU-DET + GC10-DET → 自制接触网
```

---

## 第二优先级：绝缘子/电力设施缺陷数据集

### 4）Insulator Defect Detection / Roboflow Insulator 系列

推荐程度：**高，但需注意许可**

Roboflow Universe 上有公开的 insulator defect detection 数据集，例如一个版本包含约 1600 张 Fault 图像，类别包括 insulator shell、broken insulator shell、flashover damaged insulator shell、good insulator shell 等。([Roboflow][5]) 另一个 insulator_models 数据集有 205 张开源绝缘子缺陷图像。([Roboflow][6])

它们和接触网绝缘子破损、供电设施巡检场景更接近，但数据来源和标注质量参差不齐。建议只作为“近域补充”，不要作为主预训练集。

推荐用法：

```text
COCO → 工业缺陷预训练 → 绝缘子缺陷近域微调 → 自制接触网
```

---

### 5）公开铁路接触网绝缘子数据

推荐程度：**中等，主要用于参考**

铁路接触网公开缺陷数据集非常少。2023 年一篇铁路接触网绝缘子缺陷检测论文明确提到，适合深度学习训练的铁路接触网绝缘子公开数据集较缺乏；该文使用的是郑徐高铁巡检车拍摄的 1300 张图像数据集。([PMC][7])

这说明：**真正同域公开数据很难依赖**。如果能找到接触网/绝缘子公开数据，可以加入；如果找不到，不要把训练方案建立在它一定可得的前提上。

---

## 第三优先级：小目标高分辨率数据集

### 6）TT100K

推荐程度：**中高**

TT100K 是交通标志检测数据集，包含 100000 张图像、30000 多个交通标志实例，图像分辨率高、目标通常较小。Ultralytics 文档也给出了 TT100K 检测数据集说明。([Ultralytics Docs][8]) 清华官方页面说明其图像覆盖光照和天气变化，且标注了类别、框和像素 mask。([清华大学计算机系图形与几何计算组][9])

它不是工业缺陷数据，但非常适合训练 **P2 小目标检测头、SAHI/crop 训练策略、多尺度定位能力**。

推荐用法：

```text
COCO → TT100K 小目标预训练 → 工业缺陷预训练 → 自制接触网
```

如果时间有限，可以不使用 TT100K；如果你新增了 P2 检测层，我建议加上它做一轮短预训练。

---

## 第四优先级：异常检测/分割数据集

### 7）MVTec AD

推荐程度：**中等，适合自监督或辅助预训练，不适合直接 YOLO 检测预训练**

MVTec AD 是工业异常检测基准，官方介绍其包含 15 个物体/纹理类别、5000 多张高分辨率图像，训练集为无缺陷图像，测试集包含正常和异常图像。([MVTec Software][10])

它的问题是：MVTec AD 主要用于异常检测/分割，不是标准 YOLO 检测框数据。可以把 mask 转 bbox，但类别和任务形态不完全一致。

推荐用法：

```text
用于 backbone 自监督预训练
或 mask → bbox 后做 generic_defect 单类检测预训练
```

---

### 8）VisA

推荐程度：**中等偏高**

VisA 包含 12 个子集、10821 张图像，其中 9621 张正常、1200 张异常，并提供图像级和像素级标注；其中有多个 PCB 子集，结构复杂，和规则结构异常较接近。([AWS开放数据注册表][11])

如果你愿意做 mask-to-bbox 转换，VisA 比 MVTec AD 更适合加入“规则结构异常”预训练。

---

### 9）KolektorSDD2

推荐程度：**中等**

KolektorSDD2 是工业表面缺陷数据集，官方页面说明其有 356 张可见缺陷图像、2979 张无缺陷图像，图像大小约 230×630，缺陷包括划痕、小斑点、表面瑕疵等。([ViCoS Lab][12])

它适合训练“缺陷/非缺陷”判别能力和 hard negative，但目标太少，不适合单独预训练主模型。

---

## AutoDL 平台优先建议

AutoDL 官方文档说明其提供部分常用开源数据，可在控制台“公开数据”菜单搜索数据集名称，找到后复制实例路径并解压到 `/root/autodl-tmp`。([AutoDL][13])

我无法直接看到你账号里的 AutoDL 公开数据列表，所以建议你按这个顺序搜索：

```text
1. COCO2017 / COCO
2. ImageNet
3. TT100K
4. MVTec AD
5. NEU-DET / NEU Surface Defect
6. GC10-DET
7. DeepPCB
8. Severstal
9. VisA
10. KolektorSDD2
```

现实上，AutoDL 更可能有 COCO、ImageNet、VOC、MVTec 这类常用数据；NEU-DET、GC10-DET、DeepPCB、VisA 可能需要你从 Kaggle、GitHub、Roboflow 或论文主页下载后上传到 AutoDL 数据盘。

---

# 3. 推荐最终数据组合

我建议你采用下面这个组合，不必一次性全上：

## 最推荐组合

```text
基础权重：
COCO YOLO11s / YOLO11m

公开预训练数据：
DeepPCB
NEU-DET
GC10-DET
TT100K 可选
Insulator Defect 可选

自制数据：
5120 原图 → 1024/1280 原生 ROI/crop
正样本 crop + 难负样本 crop + 边界 crop
```

如果算力和时间有限，优先级是：

```text
DeepPCB > GC10-DET > NEU-DET > Insulator Defect > TT100K > MVTec/VisA
```

如果你要最大化小目标能力：

```text
TT100K + DeepPCB + 自制 crop
```

如果你要最大化工业缺陷纹理能力：

```text
NEU-DET + GC10-DET + Severstal/KolektorSDD2
```

如果你要最大化接触网语义相似性：

```text
Insulator Defect + 自制接触网 ROI/crop
```

---

# 4. 数据标签处理策略

## 方案 A：保留公开数据原类别

适合做完整检测预训练：

```text
NEU: 6 类
GC10: 10 类
DeepPCB: 6 类
TT100K: 多类
```

优点：模型学到更丰富的类别区分能力。
缺点：最终切到自制 7 类时，YOLO 检测头分类层会重新初始化，分类头不能完全继承。

## 方案 B：全部公开缺陷合并为 `generic_defect`

适合本项目，我更推荐这个。

```text
NEU 所有缺陷 → generic_defect
GC10 所有缺陷 → generic_defect
DeepPCB 所有缺陷 → generic_defect
MVTec/VisA mask 转 bbox → generic_defect
Insulator broken/flashover → generic_defect
```

TT100K 不合并进 generic_defect，可以单独训练为 `tiny_object` 或只做短期 P2 小目标预训练。

优点：

```text
1. 让模型重点学“异常区域在哪里”
2. 避免公开类别和接触网类别语义冲突
3. 更利于 box 分支、neck、P2/P3 分支迁移
4. 对小数据集更稳
```

最终进入自制数据时，再把检测类别切成你的 7 类。

---

# 5. 详细训练方案

下面是假设你已经采纳前面建议后的完整方案：
**模型采用 YOLO11s-P2-EMA-SimAM-Lite；训练数据采用原生分辨率 ROI/crop；优化器以 AdamW 为主；增强降低 Mosaic/Erasing；最后做 hard negative 和阈值校准。**

---

## Phase 0：环境与数据准备

### 0.1 AutoDL 数据目录

建议目录结构：

```text
/root/autodl-tmp/
├── datasets/
│   ├── public/
│   │   ├── coco/
│   │   ├── neu_det/
│   │   ├── gc10_det/
│   │   ├── deeppcb/
│   │   ├── tt100k/
│   │   ├── insulator_defect/
│   │   └── mvtec_or_visa/
│   ├── subway_raw/
│   │   ├── images_5120/
│   │   └── labels/
│   ├── subway_crops/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── mixed_pretrain/
│       ├── images/
│       ├── labels/
│       └── data.yaml
└── Subway_defect_detection_main/
```

### 0.2 数据格式统一

所有检测数据统一转 YOLO 格式：

```text
class_id x_center y_center width height
```

对于 MVTec/VisA/Kolektor 这类 mask 数据：

```text
mask → connected components → bbox → YOLO label
```

过滤规则：

```text
bbox 面积 < 4 px 的丢弃
bbox 宽/高 < 2 px 的丢弃
异常区域太碎时合并相邻组件
```

---

## Phase 1：自制数据重新生成 crop

你现在最该优先做的是这个。

### 1.1 正样本 crop

从 5120 原图生成：

```text
crop_size: 1024 和 1280 两套
stride: 512 或 640
保留规则: crop 中至少包含 1 个缺陷框
```

同时对每个缺陷框做随机偏移 crop：

```text
缺陷中心 ± 100/200/300 px 随机偏移
保证缺陷不总是在 crop 中心
```

analysis 文件也建议 SAHI 风格原生分辨率 crop 训练，理由是推理流程不变、架构可不改、缺陷在特征图中信息量显著增加。

### 1.2 难负样本 crop

负样本非常关键，建议每张图采：

```text
10–30 个无缺陷但结构相似 crop
包括：完整螺栓、完整开口销、正常绝缘子、正常线夹、强反光区域、阴影区域
```

训练集比例建议：

```text
正样本 crop : 难负样本 crop = 1 : 1 到 2 : 1
```

### 1.3 验证集必须按源图划分

不要把同一张 5120 原图切出来的 crop 同时放进 train 和 val。项目原本就强调按源图分组防泄漏。

---

## Phase 2：公开数据中间域预训练

### 2.1 公开数据混合策略

建议建立 `public_defect_mix.yaml`：

```yaml
train:
  - /root/autodl-tmp/datasets/public/deeppcb/train/images
  - /root/autodl-tmp/datasets/public/gc10_det/train/images
  - /root/autodl-tmp/datasets/public/neu_det/train/images
  - /root/autodl-tmp/datasets/public/insulator_defect/train/images

val:
  - /root/autodl-tmp/datasets/public/deeppcb/val/images
  - /root/autodl-tmp/datasets/public/gc10_det/val/images
  - /root/autodl-tmp/datasets/public/neu_det/val/images

nc: 1
names: ["generic_defect"]
```

如果加入 TT100K，建议单独先跑一轮，不要和 defect 混在一起：

```yaml
nc: 1
names: ["tiny_object"]
```

---

## Phase 3：P2 小目标头预训练，可选但推荐

目标：让新增 P2 检测分支先学会小目标定位。

```text
数据：TT100K 或 DeepPCB
模型：YOLO11s-P2-EMA-SimAM-Lite
输入：1024 或 1280
训练：50–80 epochs
```

推荐配置：

```yaml
epochs: 80
imgsz: 1024
batch: 16
optimizer: AdamW
lr0: 0.001
lrf: 0.02
warmup_epochs: 5
cos_lr: true

mosaic: 0.2
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.4
degrees: 3.0
translate: 0.1
scale: 0.5
close_mosaic: 20
```

输出：

```text
weights/p2_tiny_pretrain.pt
```

如果没有 TT100K，直接用 DeepPCB 也可以。

---

## Phase 4：工业缺陷公开数据预训练

目标：让 backbone、neck、P2/P3 分支学习工业异常纹理。

```text
初始化：p2_tiny_pretrain.pt 或 COCO yolo11s.pt
数据：DeepPCB + NEU-DET + GC10-DET + 可选 Insulator
类别：generic_defect
```

推荐配置：

```yaml
epochs: 120
imgsz: 1024
batch: 16
optimizer: AdamW
lr0: 0.001
lrf: 0.02
warmup_epochs: 10
weight_decay: 0.0005
cos_lr: true

mosaic: 0.2
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
degrees: 5.0
translate: 0.1
scale: 0.5
shear: 1.0
perspective: 0.0002
close_mosaic: 30
```

训练策略：

```text
前 10 epoch：冻结 backbone 前半部分，只训 neck + head
第 11 epoch 后：解冻全部，但 backbone 使用较低学习率
```

如果当前训练脚本不支持分层学习率，先用全局 AdamW 即可。

输出：

```text
weights/public_defect_pretrain.pt
```

验收标准：

```text
公开数据 mAP50 不必追求极致
重点看训练是否稳定、P2/P3 是否有效、loss 是否正常下降
```

---

## Phase 5：自制接触网 Head/Neck 适配

目标：从公开缺陷域切换到真实接触网 7 类。

初始化：

```text
public_defect_pretrain.pt
```

数据：

```text
subway_crops/train
subway_crops/val
```

类别：

```text
7 类：VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM
```

训练配置：

```yaml
epochs: 50
imgsz: 1024
batch: 16
optimizer: AdamW
lr0: 0.001
lrf: 1.0
warmup_epochs: 3
cos_lr: false

freeze: backbone early/middle
mosaic: 0.1
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
hsv_h: 0.015
hsv_s: 0.5
hsv_v: 0.5
degrees: 3.0
translate: 0.1
scale: 0.4
```

建议冻结：

```text
冻结 backbone 前 60% 层
训练 neck + P2/P3/P4/P5 detect head + 注意力模块
```

验收标准：

```text
mAP50 > 0.35
mAP50-95 > 0.25
Recall 明显高于当前 baseline
各类别不能出现 AP=0
```

这一步类似你们原 C1，但建议不要只训练 head，而是训练 neck + head，因为 P2 新分支和接触网小目标分布需要 neck 适配。

---

## Phase 6：自制数据小目标尺度适应训练

这是主训练阶段。

```text
输入：1024/1280 原生 crop
模型：YOLO11s-P2-EMA-SimAM-Lite
初始化：Phase 5 best.pt
```

推荐配置：

```yaml
epochs: 120
imgsz: 1280
batch: 12 或 16
optimizer: AdamW
lr0: 0.0008
lrf: 0.02
warmup_epochs: 8
warmup_momentum: 0.5
weight_decay: 0.0005
cos_lr: true
patience: 40

mosaic: 0.2
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
hsv_h: 0.015
hsv_s: 0.6
hsv_v: 0.5
degrees: 5.0
translate: 0.12
scale: 0.5
shear: 1.0
perspective: 0.0003
close_mosaic: 40
```

注意：我不建议直接启用 YOLO 默认超大范围 `multi_scale: true`，因为随机到 640 时，小缺陷又会被压小。更建议自定义多尺度：

```text
随机尺度集合：[1024, 1280, 1536]
```

如果代码暂时不支持自定义集合，先固定 `imgsz=1280`。

---

## Phase 7：真实分布短微调

最近训练结果已经显示，长时间 C3 微调会退化；所以这里要短、轻、早停。

初始化：

```text
Phase 6 best.pt
```

配置：

```yaml
epochs: 30
imgsz: 1280
batch: 12
optimizer: AdamW
lr0: 0.00003
lrf: 1.0
cos_lr: false
patience: 8

mosaic: 0.0
mixup: 0.0
copy_paste: 0.0
erasing: 0.0
degrees: 1.0
translate: 0.05
scale: 0.2
shear: 0.0
perspective: 0.0
hsv_h: 0.005
hsv_s: 0.2
hsv_v: 0.2
```

冻结建议：

```text
冻结 backbone 前 70%
只微调 neck + detect head + 注意力模块
```

保存策略：

```text
每个 epoch 保存一次
最终不要默认用 last.pt
选择 best_mAP50-95.pt 和 best_F2.pt
```

---

## Phase 8：Hard Negative Mining

这是提升 Precision 的关键阶段。

步骤：

```text
1. 用 Phase 7 模型跑训练集、验证集、额外无缺陷图
2. 收集置信度 0.2–0.7 的误检 crop
3. 人工或半自动确认它们为正常结构
4. 加入 negative_crops
5. 用低学习率再训练 15–30 epoch
```

配置：

```yaml
epochs: 30
imgsz: 1280
batch: 12
optimizer: AdamW
lr0: 0.00002
patience: 8

mosaic: 0.0
copy_paste: 0.0
erasing: 0.0
```

负样本比例：

```text
正样本 : 难负样本 = 1 : 1
```

重点采集：

```text
正常螺栓但反光
正常开口销但边缘模糊
绝缘子阴影
线夹边缘
切片边界伪影
运动模糊区域
```

---

## Phase 9：阈值校准与部署验证

项目最终要求 Precision 和 Recall，而不是只看 mAP。README 中明确车载端和地面端都要求 Recall ≥90%、Precision ≥90%。

建议为每类单独搜索阈值：

```text
for each class:
    conf ∈ [0.05, 0.95]
    iou_nms ∈ [0.45, 0.70]
    选择满足 Recall ≥ 0.90 的最高 Precision 点
```

输出：

```yaml
VHBNM:
  conf: 0.42
  nms_iou: 0.55
VHBNL:
  conf: 0.36
  nms_iou: 0.55
SVHBNM:
  conf: 0.30
  nms_iou: 0.60
...
```

地面端 WBF 可沿用已有思路：一个模型偏 Precision，一个 P2 模型偏 Recall；README 中已有双卡异构 ensemble 和 WBF 融合设计。

---

# 6. 推荐实验矩阵

不要一次全改，建议这样做消融：

| 实验 | 初始化                     | 模型                   | 数据             | 目的          |
| -- | ----------------------- | -------------------- | -------------- | ----------- |
| E0 | COCO                    | 当前 YOLO11s-EMA-SimAM | 当前数据           | baseline    |
| E1 | COCO                    | 当前模型                 | 自制原生 crop      | 验证 crop 收益  |
| E2 | COCO                    | P2-Lite              | 自制原生 crop      | 验证 P2 收益    |
| E3 | COCO → DeepPCB/NEU/GC10 | P2-Lite              | 自制 crop        | 验证公开预训练收益   |
| E4 | COCO → TT100K → 工业缺陷    | P2-Lite              | 自制 crop        | 验证小目标预训练收益  |
| E5 | E4 + Hard Negative      | P2-Lite              | 自制 crop + 难负样本 | 冲 Precision |
| E6 | E5 + 阈值校准 + TensorRT    | P2-Lite              | 原始 127MP 全流程   | 部署验收        |

建议最终报告只保留 E0/E1/E2/E3/E5/E6 六组即可。

---

# 7. 一套可执行的训练路线

最终我建议你按这个顺序执行：

```text
Step 1：准备自制 1024/1280 原生 crop 数据集
Step 2：新建 YOLO11s-P2-EMA-SimAM-Lite
Step 3：COCO 权重初始化
Step 4：DeepPCB + NEU-DET + GC10-DET 做 generic_defect 公开预训练
Step 5：加载 public_defect_pretrain.pt，在自制 crop 上做 neck/head 适配
Step 6：1280 crop 主训练，弱 Mosaic，禁用 erasing/copy_paste
Step 7：短微调，早停，保存每轮模型
Step 8：Hard Negative Mining 再训练
Step 9：每类阈值校准
Step 10：原始 127MP 全流程测试 + TensorRT FP16 导出
```

---

## 最终建议

**可以使用公开数据集预训练，而且应该使用。**
但推荐方式不是“公开数据 + 自制数据直接混训到最终模型”，而是：

```text
COCO
→ DeepPCB / NEU-DET / GC10-DET / 可选 TT100K / 可选 Insulator Defect
→ 自制接触网原生 crop
→ hard negative
→ 阈值校准
```

最值得优先尝试的是：

```text
DeepPCB + GC10-DET + NEU-DET
```

如果 AutoDL 公开数据里能直接找到，就直接复制到实例路径；找不到的话，建议手动下载后上传到 `/root/autodl-tmp`。这套路线对你当前“小数据 + 微小缺陷 + 规则工业结构 + 实时部署”的约束最稳，也最容易在实验报告中证明每一步的贡献。

[1]: https://faculty.neu.edu.cn/songkc/en/zdylm/263265?utm_source=chatgpt.com "NEU surface defect database"
[2]: https://www.mdpi.com/1424-8220/20/6/1562?utm_source=chatgpt.com "Deep Metallic Surface Defect Detection: The New ..."
[3]: https://datasetninja.com/gc10-det?utm_source=chatgpt.com "GC10-DET Dataset"
[4]: https://github.com/tangsanli5201/DeepPCB?utm_source=chatgpt.com "tangsanli5201/DeepPCB: A PCB defect dataset."
[5]: https://universe.roboflow.com/insulator-defect-detection/insulator-defect-detection-veowd?utm_source=chatgpt.com "Insulator Defect Detection Computer Vision Model"
[6]: https://universe.roboflow.com/dataset-oziyh/insulator_models?utm_source=chatgpt.com "insulator_models Object Detection Model by dataset"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10403183/?utm_source=chatgpt.com "Detection of railway catenary insulator defects based ... - PMC"
[8]: https://docs.ultralytics.com/datasets/detect/tt100k?utm_source=chatgpt.com "TT100K Traffic Sign Dataset"
[9]: https://cg.cs.tsinghua.edu.cn/traffic-sign/?utm_source=chatgpt.com "Tsinghua-Tencent 100k (traffic signs)"
[10]: https://www.mvtec.com/research-teaching/datasets/mvtec-ad?utm_source=chatgpt.com "Industrial anomaly detection benchmark dataset"
[11]: https://registry.opendata.aws/visa/?utm_source=chatgpt.com "Visual Anomaly (VisA) - Registry of Open Data on AWS"
[12]: https://www.vicos.si/resources/kolektorsdd2/?utm_source=chatgpt.com "Kolektor Surface-Defect Dataset 2 (KolektorSDD2 / KSDD2)"
[13]: https://www.autodl.com/docs/public_data/ "AutoDL帮助文档"
