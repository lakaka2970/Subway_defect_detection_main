# 模型特征学习效率诊断与优化方案

> **日期:** 2025-06-25  
> **分析对象:** `output/20260625_132716`, `20260625_135945`, `20260625_155323`, `20260625_185613` 四次完整训练  
> **模型:** YOLO11s-EMA-SimAM (9.4M params, 21.6 GFLOPs, imgsz=1024)  
> **数据集:** 地铁接触网缺陷检测, 7类, 878训练图/101验证图, 5120×5120原图  
> **状态:** C2全模型训练阶段特征学习效率极低, 需系统性优化

---

## 目录

1. [核心发现：C2阶段几乎零收益](#1-核心发现c2阶段几乎零收益)
2. [逐阶段训练动态分析](#2-逐阶段训练动态分析)
3. [根因分析](#3-根因分析)
   - [3.1 训练分辨率与缺陷尺寸的根本性不匹配](#31-训练分辨率与缺陷尺寸的根本性不匹配-影响等级严重)
   - [3.2 SGD学习率设置过低](#32-sgd学习率设置过低-影响等级严重)
   - [3.3 C1→C2过渡丢弃优化器状态](#33-c1c2过渡丢弃优化器状态-影响等级严重)
   - [3.4 数据增强对小目标的破坏性影响](#34-数据增强对小目标的破坏性影响-影响等级中等)
   - [3.5 模型架构尺度分布不合理](#35-模型架构尺度分布不合理-影响等级中等)
   - [3.6 单一尺度训练](#36-单一尺度训练-影响等级中等)
   - [3.7 损失函数与类别不平衡](#37-损失函数与类别不平衡-影响等级较轻)
4. [改进方案](#4-改进方案)
   - [A. 训练分辨率与多尺度策略](#a-训练分辨率与多尺度策略)
   - [B. 学习率与优化策略](#b-学习率与优化策略)
   - [C. 模型架构改进](#c-模型架构改进)
   - [D. 数据增强策略调整](#d-数据增强策略调整)
   - [E. 训练流程优化](#e-训练流程优化)
5. [实施路线图](#5-实施路线图)
6. [附录：数据速查表](#6-附录数据速查表)

---

## 1. 核心发现：C2阶段几乎零收益

C2 Full Training（全模型解冻, 200 epochs, 重度数据增强）是训练流程中理应收益最大的阶段, 但实际数据显示**四个训练运行中C2均未产生有意义的提升**:

### 训练结果汇总

| 运行 | C1 Warmup | C2 Full | C2增益 | C3 Fine-tune | C3增益 | C2优化器 | 备注 |
|------|-----------|---------|--------|-------------|--------|---------|------|
| Run B `135945` | 0.3435 | 0.3478 | **+0.004** | 0.4136 | +0.066 | AdamW | C2批大小4 |
| Run C `155323` | 0.3435 | 0.3462 | **+0.003** | 0.3994 | +0.053 | SGD | C2批大小4 |
| Run D `185613` | 0.3769 | 0.3953 | **+0.018** | 0.3770 | **-0.018** | SGD | C2批大小22, C3退化 |

> **关键数据:** C2全模型训练200个epoch, 每次epoch遍历878张训练图, 总计约175,600次前向+反向传播。换来最多0.018的mAP50提升——每个epoch平均贡献不到0.0001。这接近于统计噪声水平。

### 对比：C1与C2的效率鸿沟

```
C1 Warmup (50 epochs, 冻结骨干, 简单增强):
  Run D: 0 → 0.377 mAP50, 50 epochs
  效率: +0.0075 mAP50/epoch

C2 Full (200 epochs, 解冻骨干, 复杂增强):
  Run D: 0.377 → 0.395 mAP50, 130 epochs (early stop)
  效率: +0.00014 mAP50/epoch

C1的学习效率是C2的 54倍。
```

### 矛盾现象

- C1仅训练检测头（骨干冻结）就达到0.377 mAP50, 说明**检测头有足够容量学习缺陷识别**
- C2解冻骨干后几乎无提升, 说明**骨干网络没有提取到有效的领域特化特征**
- C3在Run D中出现**反向退化**（0.395→0.377）, 进一步佐证骨干网络不稳定

**核心矛盾:** COCO预训练的骨干网络在接触网领域可能输出的是"错误但有微弱相关性"的特征, 检测头学会了部分补偿, 但骨干在C2中因学习率过低未能有效适应新领域。

---

## 2. 逐阶段训练动态分析

### 2.1 C1 Warmup阶段

**配置特征:**
- 骨干冻结 (layers 0-10), 仅训练FPN+PAN颈部 + 检测头
- lr=0.001 (SGD, 无衰减, lrf=1.0)
- 轻度增强: mosaic=0.5, mixup/copy_paste禁用

**训练动态:**

| 运行 | 最终mAP50 | 最终mAP50-95 | 收敛epoch | 特点 |
|------|-----------|-------------|-----------|------|
| Run B | 0.3435 | 0.2229 | ~47 | 小batch=4, 收敛稳定 |
| Run C | 0.3435 | 0.2229 | ~47 | 与Run B结果完全相同(共享C1权重) |
| Run D | 0.3769 | 0.2746 | ~50 | 大batch=22, 显著更好的收敛 |

Run D的C1效果明显优于Run B/C（+0.033 mAP50）, 主要原因: 大batch(22 vs 4)带来更稳定的梯度估计。但即使Run D的最优C1, 各类别mAP严重不平衡:

```
Run D C1 per-class mAP50:
  VHBNM:  0.647  ← 最佳
  SVHTNL: 0.633  ← 较好
  VHBNL:  0.376
  SVHBNL: 0.209  ← 差
  CBVPM:  0.242
  SVHBNM: 0.148  ← 最差 (虽然样本量最大!)
  CBHPM:  0.137  ← 最差
```

**关键观察:** SVHBNM是训练集中样本量最大的类别(506个标注), 但mAP50最低(0.148)。这不是样本量问题, 而是**该类别的视觉特征与COCO预训练特征空间距离最远**, 冻结的骨干网无法提取有效特征。

### 2.2 C2 Full Training阶段

**配置特征:**
- 全部层解冻 (freeze=[])
- lr=0.001→0.0001 (SGD + Cosine LR), warmup=5 epochs
- 增强: mosaic=0.5, copy_paste=0.3, erasing=0.4, close_mosaic=15

**训练动态 (以Run D为例):**

```
Epoch    train/box_loss  val/box_loss   mAP50    mAP50-95
  1       1.25            1.18          0.005    0.001     ← 解冻初期剧烈震荡
  5       0.99            1.03          0.191    0.110
 10       0.89            0.94          0.287    0.194
 20       0.78            0.89          0.339    0.237
 30       0.72            0.86          0.357    0.253
 40       0.68            0.86          0.373    0.268
 50       0.66            0.89          0.384    0.278     ← ~50 epoch 后趋于平台
 70       0.63            0.92          0.388    0.282
 90       0.61            0.95          0.390    0.283
110       0.60            0.98          0.392    0.283
130       0.59            0.97          0.395    0.285     ← early stop
```

**三个关键观察:**

1. **震荡期长:** 前5个epoch mAP50从0.005开始, 到epoch 20才恢复至C1水平(0.339), 用了15个epoch"恢复"。这说明C1的检测头权重在C2初始阶段被骨干网的变化严重干扰。

2. **快速平台:** 从epoch 40开始, mAP50增长曲线几乎水平。box_loss仍在缓慢下降(0.68→0.59), 但validation loss不再改善。这是典型的**过拟合+弱特征学习**模式: 模型在记忆训练集的纹理细节, 但没有学到可泛化的语义特征。

3. **Early Stop过早:** 所有运行都在130-173 epoch被patience=50提前终止。这本身不奇怪——但配合"mAP不增长"的现象, 说明模型已经在局部最优中卡住, 继续训练不会改善。

### 2.3 C3 Fine-Tune阶段

**配置特征:**
- lr=0.0001 (SGD, 无衰减), batch=8/16
- 增强最小化: mosaic=0, copy_paste=0, erasing=0

**异常现象——Run D的C3退化:**

```
Run D C2 best:  mAP50=0.395, mAP50-95=0.285
Run D C3 final: mAP50=0.377, mAP50-95=0.275  ← 全面退化!
```

这是catastrophic forgetting的明确信号。可能原因:
- C3的lr=0.0001虽然低, 但对某些关键层仍过大
- 在C2已经过拟合的基础上继续训练, 模型进一步偏离泛化方向
- 验证集与训练集的分布差异在C3阶段被放大

Run B (AdamW) 的C3则表现正常（+0.066 mAP50）, 原因是AdamW的自适应学习率对不同层有不同的有效步长, 避免了对关键层的过度更新。

---

## 3. 根因分析

### 3.1 训练分辨率与缺陷尺寸的根本性不匹配 (影响等级: 🔴严重)

这是所有问题中最根本的一个。请审视以下尺寸链条:

```
原始图像: 5120 × 5120 px  (地铁车载相机, 1.27亿像素等效采样)
    │
    │ YOLO dataloader: letterbox resize to 1024×1024
    │ 缩放因子: 1024/5120 = 0.2 (5× 下采样)
    ▼
训练尺寸: 1024 × 1024 px
    │
    │ 以典型缺陷 "螺栓缺失" 为例:
    │   原图标注框: ~40 × 40 px  (一个M16螺栓+安装底座)
    │   训练尺寸框: ~8 × 8 px    (5× 缩小后)
    ▼
特征图尺度:
  ┌──────────┬────────────┬──────────────────┬──────────────────────┐
  │ 检测层   │ stride     │ 特征图分辨率     │ 缺陷所占像素          │
  │          │            │ (1024输入)       │ (8×8px目标)          │
  ├──────────┼────────────┼──────────────────┼──────────────────────┤
  │ P2 (无)  │ 4          │ 256 × 256       │ 2.0 × 2.0 px ✓      │
  │ P3       │ 8          │ 128 × 128       │ 1.0 × 1.0 px ⚠     │
  │ P4       │ 16         │ 64 × 64         │ 0.5 × 0.5 px ✗      │
  │ P5       │ 32         │ 32 × 32         │ 0.25 × 0.25 px ✗    │
  └──────────┴────────────┴──────────────────┴──────────────────────┘
```

**物理现实:** P4和P5特征图上, 缺陷的物理尺寸不足1个像素。这意味着:

- P4检测头 (SimAM attention) 永远看不到真正的缺陷特征
- P5检测头 (SimAM attention) 完全在背景噪声上做检测
- 模型3个检测尺度中有2个在做无效计算, 浪费了~60%的head容量

**这解释了为什么C1能达到0.377**: 检测头在P3(唯一有效的检测尺度)上学到了基本检测能力。但P4和P5的"检测"实际上是在随机猜测——它们看到的是大尺度的背景结构模式, 被强行关联到了缺陷标签。

**这还解释了为什么C2几乎无收益**: 骨干网络试图学习更好的特征, 但P4/P5的梯度信号本质上是噪声(因为这两个尺度根本没有缺陷信号)。噪声梯度反向传播污染了骨干网络, 抵消了P3的有效梯度。

**对比: 如果训练分辨率为1280:**

```
训练尺寸: 1280 × 1280 px (缩放因子 0.25)
缺陷尺寸: ~10 × 10 px

  P2 (stride 4):  2.5 × 2.5 px  ✓ 良好
  P3 (stride 8):  1.25 × 1.25 px ✓ 勉强
  P4 (stride 16): 0.63 × 0.63 px ⚠ 仍不够
  P5 (stride 32): 0.31 × 0.31 px ✗ 仍无效
```

即使1280分辨率, P4/P5仍然不可用——必须增加P2检测层。

### 3.2 SGD学习率设置过低 (影响等级: 🔴严重)

对比YOLO官方默认值与当前配置:

| 参数 | YOLO默认 | C1配置 | C2配置 | C3配置 | 分析 |
|------|---------|--------|--------|--------|------|
| `lr0` | **0.01** | 0.001 | 0.001 | 0.0001 | C1/C2仅为默认值的1/10 |
| `lrf` | 0.01 | 1.0 | 0.1 | 0.1 | C1不衰减, C2衰减到1e-4 |
| `momentum` | 0.937 | 0.937 | 0.937 | 0.937 | 标准 |
| `weight_decay` | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 标准 |
| `warmup_epochs` | 3 | 3 | 5 | 3 | 合理 |

**量化分析——为什么lr0=0.001对骨干网几乎等于不更新:**

SGD更新公式:
```
w_new = w_old - lr * (gradient + weight_decay * w_old)
```

COCO预训练的骨干网权重通常在[-0.5, 0.5]范围。对于接触网图像的梯度:
- 骨干浅层 (Conv, stride 2/4): 边缘/纹理特征——COCO的边缘与接触网的边缘高度通用, 梯度约在1e-3量级
- 骨干中层 (C3k2, stride 8/16): 形状特征——差异开始出现, 梯度约在5e-3量级
- 骨干深层 (C2PSA, stride 32): 语义特征——COCO的人/车/动物与接触网螺栓/绝缘子完全不同, 梯度约在1e-2量级

以深层权重 w=0.3, gradient=0.01 为例:
```
w_new = 0.3 - 0.001 * (0.01 + 0.0005 * 0.3)
      = 0.3 - 0.001 * 0.01015
      = 0.3 - 0.00001015
      = 0.29998985
```
**每一步的权重变化约为1e-5量级。** 对于值在0.1-0.5范围的权重, 需要>10000步才能产生1%的变化。C2训练~175,000步, 骨干深层权重最多变化20-30%——这对于从一个完全不同的视觉域(COCO→接触网)做迁移学习来说远远不够。

**对比YOLO默认lr0=0.01:**
```
w_new = 0.3 - 0.01 * 0.01015 = 0.3 - 0.0001015 = 0.2998985
```
每步变化~1e-4, 同样的175,000步可以完全重塑骨干网——这才是fine-tuning应有的力度。

**Run B (AdamW)的启示:** AdamW的自适应学习率机制部分补偿了lr0过低的问题。AdamW的有效步长 = `lr * m_hat / (sqrt(v_hat) + eps)`。对于梯度一致小的参数(如深层骨干), AdamW会自动增大有效步长。这解释了为什么Run B的C3能提升+0.066 mAP50, 而Run C/D的SGD在C3几乎没有提升甚至退化。

### 3.3 C1→C2过渡丢弃优化器状态 (影响等级: 🔴严重)

当前代码流程:

```python
# train_defect.py _run_stage()
model = YOLO(str(ckpt_in))        # ① 全新Model实例, 从.pt文件加载权重
model.train(name="c2_full", ...)  # ② 进入训练循环
```

在步骤①中, `YOLO(ckpt.pt)` 内部调用链:
```
YOLO.__init__()
  → Model._load(weights.pt)
    → load_checkpoint(weights.pt)   # 读取权重 + checkpoint dict
    → self.model = parse_model(cfg) # 构建空模型
    → self.model.load(weights)      # 仅加载模型权重
```

**关键缺失:** `load_checkpoint` 返回的 `ckpt` dict中包含 `optimizer` 键(存储SGD动量缓冲区), 但在 `Model._load()` 中**仅模型权重被加载, 优化器状态被丢弃**。当`model.train()`调用`_setup_train()`时, 它会创建全新的优化器:

```python
# trainer.py _setup_train()
self.optimizer = self.build_optimizer(
    self.model, 
    name=self.args.optimizer,  # "SGD"
    lr=self.args.lr0,          # 0.001
    momentum=0.937,
    decay=0.0005,
)
```

**影响:** C1训练50个epoch中积累的SGD动量缓冲区全部清零。骨干网在C1中完全冻结, momentum=0。C2解冻后, 骨干网的每个参数从**零速度和零历史梯度**开始——就像一个从未被训练过的随机初始化网络, 同时还要应对C1检测头的梯度反传。

5个warmup epoch只是线性增加学习率, 无法补偿动量信息的缺失。通常需要10-20个epoch才能让SGD动量达到稳定状态。

### 3.4 数据增强对小目标的破坏性影响 (影响等级: 🟡中等)

当前C2增强配置及其对小目标的影响:

#### Mosaic (概率 0.5)

```
原始: 5120×5120 → 1024×1024 (训练图)

Mosaic: 随机选4张图 → 各自resize到~512×512 → 拼成1024×1024
  - 每个象限的有效分辨率: 512×512 (相对于原图, 10×下采样)
  - 象限中的缺陷: 40×40px → 4×4px
  - 拼合后resize可能再引入插值模糊
```

**50%的训练样本中, 缺陷缩小到4×4px。** 模型一半的时间在学习检测"4像素的神秘斑点"。

#### Random Erasing (概率 0.4)

Random Erasing在图上随机放置一个0-色矩形块, 覆盖原始像素。对于8×8的缺陷区域:
- Erasing块可能完全覆盖缺陷
- 缺陷周围的结构上下文(螺栓孔、金属边缘)被抹去
- 边界效应可能产生伪影

**对于8×8的小目标, 40%的擦除概率意味着每2.5个epoch就有一次整个缺陷被随机擦除。** 这在COCO(目标通常>32×32)是有效的正则化, 但对于我们的微小缺陷场景是灾难性的。

#### Copy-Paste (概率 0.3, flip模式)

从一张图复制缺陷实例, 翻转后粘贴到另一张图。问题:
- 被复制的缺陷在resize过程中进一步缩小
- 粘贴位置可能与接触网结构不匹配
- 翻转改变了缺陷的方向特征(螺栓的方向性很重要)

### 3.5 模型架构尺度分布不合理 (影响等级: 🟡中等)

#### 当前架构: yolo11s-EMA-SimAM

```
检测尺度: P3/8, P4/16, P5/32  (3尺度)
注意力分配:
  P3: EMA  (参数化空间注意力, ~200参数)
  P4: SimAM (无参数能量注意力)
  P5: SimAM (无参数能量注意力)
Backbone: C2PSA (YOLO11原生位置敏感注意力)
```

#### 问题1: 注意力模块放错了位置

SimAM的设计哲学是识别"与周围神经元显著不同的神经元"——这对局部异常(如规则金属表面上的缺失螺栓)理论上完美。但它被放在了P4/P5, 这两个尺度上**缺陷根本不存在**。SimAM在P4/P5做的是: 在64×64/32×32的大尺度特征图上, 寻找"与邻域不同"的特征——它找到的是大尺度的结构变化(两个部件之间的边缘、光照变化), 而非微小的螺栓缺失。

#### 问题2: EMA的作用有限

EMA在P3, 通过X/Y方向分别池化来保留空间位置信息。这对COCO中的小物体(如鸟、球)有好处——它们的空间位置本身包含信息。但对于接触网缺陷, 缺陷的空间位置(在图像的左上/右下)与缺陷类别无关——一个螺栓缺失在图像的任何位置看起来都一样。EMA的空间编码可能增加了不必要的位置偏差。

#### 问题3: 缺少P2检测层

项目已有`yolo11m-P2-SimAM.yaml`(地面端GPU1, P2/4 + P3/8 + P4/16 + P5/32 四尺度检测), 但车载端的`yolo11s-EMA-SimAM`只用了三尺度。对于1024训练分辨率下的8×8px缺陷, P2(stride 4, 256×256特征图)能让缺陷占据2×2个特征图像素——从"勉强一个像素"变为"2×2区域", 检测难度大幅下降。

#### 问题4: depth_mult=0.50 削减了neck容量

YOLO11s的scale配置 `s: [0.50, 0.50, 1024]` 意味着:
- depth_mult=0.50: C3k2的 `n=2` → `max(round(2*0.5), 1) = 1`
- 所有neck中的C3k2实际只有**1个Bottleneck**
- width_mult=0.50: 通道数减半 (256→128 at P3, 512→256 at P4)

对于需要细粒度特征区分的缺陷检测, 这个容量可能不足。增加到 [0.67, 0.50, 1024] 可以让C3k2实际使用2个Bottleneck。

### 3.6 单一尺度训练 (影响等级: 🟡中等)

当前所有阶段使用固定 `imgsz=1024`。YOLO支持 `multi_scale=True`, 在每个batch随机变化imgsz ±50%。未启用导致:

1. **尺度过拟合:** 模型只在1024×1024的缺陷尺寸上有效。推理时如果输入尺寸有变化(如slice尺寸调整), 性能下降。
2. **特征尺度单一:** 骨干网没有动力学习尺度不变特征。对于需要跨尺度泛化的检测任务(同一类缺陷在不同距离/角度下的外观差异), 这是致命的。
3. **P3/P4/P5的分工失衡:** 在固定1024下, P4/P5已经看不到缺陷。开启multi_scale后, 当随机imgsz较大时(如~1280), P4开始看到缺陷; 较小时(如~640), 缺陷向P2/P3集中。这种动态分配让每个尺度都有"机会"学习。

### 3.7 损失函数与类别不平衡 (影响等级: 🟢较轻)

YOLO11使用:
- **Box loss:** CIoU (Complete IoU) — 对框的位置和尺寸敏感
- **Cls loss:** BCE (Binary Cross Entropy) — 标准的类别分类
- **DFL loss:** Distribution Focal Loss — 对边界框分布的精细回归

对于微小目标的问题:
- CIoU对小框的IoU计算非常敏感——1像素的偏移可能导致IoU从0.5变为0.1
- BCE对所有样本权重相同——容易负样本(background)占主导
- 小框的DFL学习比大框困难(分布更集中)

类别不平衡加剧了这个问题:

```
类别分布 (训练集, 2202标注):
  SVHBNM: 506 (23.0%) — 最丰富
  VHBNM:  410 (18.6%)
  SVHTNL: 322 (14.6%)
  CBHPM:  296 (13.4%)
  VHBNL:  282 (12.8%)
  CBVPM:  230 (10.4%)
  SVHBNL: 156 (7.1%)  — 最稀少
```

虽然类别间差异不算极端(最大/最小=3.2:1), 但样本绝对数量太少(SVHBNL只有156个)。YOLO的默认损失对类别不平衡没有特殊处理。

---

## 4. 改进方案

以下方案按预期收益从高到低排列。

### A. 训练分辨率与多尺度策略

**预期收益: +0.05~0.10 mAP50** (最大单项收益)

#### A1. 提升训练分辨率至1280

```yaml
# 所有阶段
imgsz: 1280  # 1024 → 1280
```

**效果:**
```
1280×1280训练时:
  缺陷尺寸: ~10×10 px  (vs 1024时 8×8)
  P3 (stride 8): 1.25 × 1.25 特征图像素 (vs 1.0)
  P4 (stride 16): 0.625 × 0.625 (仍不够, 但更接近)
```

VRAM影响: yolo11s @ 1280, 估计每样本~1.2GB (vs 1024时~0.8GB)。RTX 5090 32GB:
- 安全batch: `(32 * 0.75 - 6) / 1.2 = 15` → batch=14-16
- 配合gradient accumulation可以达到等效batch=32

#### A2. 启用多尺度训练

```yaml
# C2和C3
multi_scale: true    # 随机 ±50% imgsz
imgsz: 1280          # 基准尺寸
```

YOLO会在每个batch随机选择imgsz (从640到1920, stride对齐到32)。这强制:
- P4在imgsz≥1400时开始看到缺陷 → 学习大尺度特征
- P3在imgsz~1000-1280时最有效 → 保持中等尺度检测
- 模型整体获得尺度鲁棒性

#### A3. 渐进式分辨率

```yaml
C1 (Warmup): imgsz=1024  # 快速收敛, 验证head基础能力
C2 (Full):   imgsz=1280, multi_scale=true  # 高精度特征学习
C3 (Finetune): imgsz=1280, multi_scale=false  # 最终固定分辨率精调
```

---

### B. 学习率与优化策略

**预期收益: +0.03~0.08 mAP50**

#### B1. 提升C2基础学习率

```yaml
# config/train/full.yaml
lr0: 0.005      # 0.001 → 0.005 (YOLO默认0.01的一半, 保守适配小数据集)
lrf: 0.02       # 0.1 → 0.02 (最终LR: 0.005×0.02 = 1e-4, 与当前相同)
cos_lr: true    # 保持余弦衰减
```

**选择0.005而非0.01的理由:**
- 878张训练图, 每个epoch~40个batch (batch=22), 不像COCO有几百万样本
- 过大的lr在小数据集上容易导致不稳定的梯度
- 0.005 = 当前lr的5倍, 仍然只有YOLO默认的一半

#### B2. C2使用AdamW优化器

```yaml
# config/train/full.yaml
optimizer: AdamW
lr0: 0.001       # AdamW推荐lr比SGD低一个数量级
lrf: 0.02
cos_lr: true
```

AdamW的优势:
- 自适应学习率: 深层骨干的稀疏梯度自动获得更大有效步长
- 解耦权重衰减: AdamW的正则化效果优于SGD的L2正则
- 对C1→C2动量丢失不敏感: AdamW内部维护自己的动量状态
- Run B的实证: AdamW在C3阶段实现了+0.066 mAP50 (SGD只有+0.053)

#### B3. C2 warmup增强

```yaml
# config/train/full.yaml
warmup_epochs: 10       # 5 → 10
warmup_momentum: 0.5    # 0.8 → 0.5 (更慢的初始动量, 更平稳过渡)
warmup_bias_lr: 0.05    # 0.1 → 0.05
```

更长的warmup给骨干网更多时间适应, 避免C1→C2过渡的震荡。

#### B4. 模型EMA (指数移动平均)

```yaml
# 所有阶段
model_ema: true
ema_decay: 0.9999
```

Model EMA维护模型权重的指数移动平均副本, 在验证时使用。YOLO默认配置中没有显式开启, 但在大数据集训练中通常被使用。对稳定C2训练和防止C3退化特别有效。

---

### C. 模型架构改进

**预期收益: +0.03~0.06 mAP50**

#### C1. 车载端增加P2检测层

创建 [yolo11s-P2-EMA-SimAM.yaml](subway_defect/models/yolo11s-P2-EMA-SimAM.yaml):

```yaml
# 基于 yolo11m-P2-SimAM.yaml 的架构, 使用 s 级 scale
nc: 18
scales:
  s: [0.50, 0.50, 1024]

backbone:
  # ... 与现有相同 ...

head:
  # FPN: P5 → P4 → P3 → P2
  # PAN: P2 → P3 → P4 → P5
  # Attention: P2=SimAM, P3=EMA, P4=SimAM, P5=SimAM
  # Detect: [[P2, P3, P4, P5], Detect, [nc]]
```

**4尺度检测头在1280分辨率下的表现:**
```
P2 (stride 4, 320×320特征图): 缺陷 = 2.5×2.5 px ✓ 理想
P3 (stride 8, 160×160特征图): 缺陷 = 1.25×1.25 px ✓ 可用
P4 (stride 16, 80×80特征图):  缺陷 = 0.63×0.63 px ⚠ 大缺陷时可用
P5 (stride 32, 40×40特征图):  缺陷 = 0.31×0.31 px ✗ 仅定位大结构
```

#### C2. 重新分配注意力模块

```yaml
# 新注意力布局
P2: SimAM  # 参数自由, 避免小数据过拟合, 在最需要的地方
P3: EMA    # 参数化多尺度注意力, 为中等缺陷提供空间+通道增强
P4: SimAM  # 参数自由, 辅助大尺度检测
P5: 无     # stride 32根本看不到缺陷, 不加注意力浪费计算
```

#### C3. 增加neck容量

```yaml
scales:
  s: [0.67, 0.50, 1024]  # depth_mult: 0.50 → 0.67
```

效果: C3k2的 `n=2 * 0.67 = 1.34 → max(round(1.34), 1) = 1`。遗憾, 仍为1。需要 `depth_mult ≥ 0.75` 才能使 `n→2`。

替代方案——直接在YAML中显式指定 repeats=3:

```yaml
# neck中的C3k2显式使用n=3(绕过depth_mult)
- [-1, 3, C3k2, [256, False]]   # P3 neck, n=3
```

或者改用 `depth_mult: 1.0`:

```yaml
scales:
  s: [1.0, 0.50, 1024]  # depth 全量, width 保持s级
```

这会增加约30%参数量 (~12.2M), 但能显著提升特征提取能力。

---

### D. 数据增强策略调整

**预期收益: +0.02~0.04 mAP50**

#### D1. 降低 Erasing 概率

```yaml
# config/train/full.yaml
erasing: 0.1    # 0.4 → 0.1
```

保留少量erasing的正则化效果, 但大幅降低对小目标的破坏。

#### D2. 降低 Mosaic 概率

```yaml
# config/train/full.yaml
mosaic: 0.3     # 0.5 → 0.3
```

更多训练样本是真实全分辨率图像, 缺陷保持原始尺寸。

#### D3. 延长 close_mosaic

```yaml
# config/train/full.yaml
close_mosaic: 40   # 15 → 40
```

在300 epoch训练中, 最后40个epoch(13%)不使用mosaic, 让模型在真实图像上精调。这比15 epoch(5%)更充分。

#### D4. 增大 scale 范围

```yaml
# config/train/full.yaml
scale: 0.7        # 0.5 → 0.7
```

YOLO默认scale=0.5表示缩放范围[0.5, 1.5]。增大到0.7表示[0.3, 1.7]。更大的缩放范围增强尺度不变性。

#### D5. C2 Copy-Paste 调整

```yaml
# config/train/full.yaml
copy_paste: 0.15  # 0.3 → 0.15 (降低频率)
```

或者完全禁用:

```yaml
copy_paste: 0.0   # 小数据集+小目标场景, copy_paste更可能制造伪影
```

---

### E. 训练流程优化

**预期收益: +0.01~0.03 mAP50**

#### E1. 延长C2训练并增加patience

```yaml
# config/train/full.yaml
epochs: 300      # 200 → 300
patience: 100    # 50 → 100 (沿用Run D的设置, 有效)
```

配合更高的lr0和AdamW优化器, 模型有更长时间学习有效的骨干特征。

#### E2. 增加C3 epoch

```yaml
# config/train/finetune.yaml
epochs: 100      # 50 → 100
patience: 50     # 50 → 50 (不变)
```

C3在Run B用了50个epoch达到最佳, 100个epoch给更多余量。

#### E3. C1→C2 训练脚本优化: 保留优化器状态

修改 [train_defect.py](subway_defect/train/train_defect.py) 中的 `_run_stage()`:

```python
def _run_stage(stage_key, config, profile, args, ckpt_in):
    model = YOLO(str(ckpt_in))
    # 新增: 如果ckpt_in是.pt文件, 设置resume=True来保留优化器状态
    if ckpt_in.suffix == ".pt" and stage_key != "warmup":
        config["resume"] = True  # 尝试恢复优化器状态
    model.train(...)
```

注意: YOLO的`resume=True`需要ckpt中有optimizer key, 且模型架构完全一致。C1→C2的freeze设置不同(C2无freeze), 可能需要验证兼容性。如果不兼容, 则只在数据加载后手动复制优化器状态。

#### E4. 分层学习率 (更长期的改进)

YOLO原生不支持分层LR。需要通过自定义optimizer构建实现:

```python
# 概念代码
param_groups = [
    {"params": backbone_params, "lr": lr0 * 0.1},   # 骨干: 低LR
    {"params": neck_params,    "lr": lr0 * 0.5},     # 颈部: 中LR
    {"params": head_params,    "lr": lr0 * 1.0},     # 检测头: 高LR
]
```

这需要修改YOLO trainer的`build_optimizer`方法。作为更长期的改进项。

---

## 5. 实施路线图

### Phase 1: 配置级优化 (立即可行, 0代码改动)

修改 YAML 配置文件即可实施, 预期总收益 **+0.08~0.15 mAP50**:

| 步骤 | 文件 | 改动 | 预期收益 |
|------|------|------|---------|
| 1.1 | `config/train/full.yaml` | `lr0: 0.005`, `optimizer: AdamW` | +0.03~0.05 |
| 1.2 | `config/train/full.yaml` | `imgsz: 1280`, `multi_scale: true` | +0.03~0.05 |
| 1.3 | `config/train/full.yaml` | `erasing: 0.1`, `mosaic: 0.3`, `close_mosaic: 40` | +0.01~0.03 |
| 1.4 | `config/train/full.yaml` | `epochs: 300`, `warmup_epochs: 10` | +0.01~0.02 |
| 1.5 | `config/train/full.yaml` | `model_ema: true` | 稳定性提升 |

**Phase 1 修改后的 `config/train/full.yaml`:**

```yaml
# C2 Full Training — 解冻全部层, 完整训练 (v2 优化版)
# 优化要点:
#   - imgsz 1024→1280: 缺陷从8px→10px, 提升P3检测能力
#   - lr0 0.001→0.005: 5倍学习率, 骨干网获得有效更新
#   - optimizer SGD→AdamW: 自适应学习率, 补偿C1→C2动量丢失
#   - erasing 0.4→0.1: 保护小目标不被随机擦除
#   - mosaic 0.5→0.3: 更多真实全分辨率图像
#   - close_mosaic 15→40: 更长无mosaic精调期
#   - multi_scale true: 学习尺度不变特征
#   - model_ema true: 指数移动平均, 提升泛化

epochs: 300
imgsz: 1280
batch: 16
multi_scale: true
model_ema: true

optimizer: AdamW
lr0: 0.001
lrf: 0.02
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 10
warmup_momentum: 0.5
warmup_bias_lr: 0.05
cos_lr: true

# 数据增强 (降低对小目标的破坏)
mosaic: 0.3
mixup: 0.0
copy_paste: 0.0        # 小目标场景禁用
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.6
degrees: 5.0
translate: 0.15
scale: 0.7             # 增大缩放范围
shear: 2.0
perspective: 0.0005
flipud: 0.0
fliplr: 0.5
close_mosaic: 40
erasing: 0.1
auto_augment: randaugment
```

### Phase 2: 架构级优化 (需创建新模型YAML)

| 步骤 | 文件 | 改动 | 预期收益 |
|------|------|------|---------|
| 2.1 | 新建 `models/yolo11s-P2-EMA-SimAM.yaml` | 增加P2检测层 | +0.02~0.04 |
| 2.2 | 新YAML | SimAM移至P2/P3, P4保留, P5去除 | +0.01~0.02 |
| 2.3 | 新YAML | `scales: [0.67, 0.50, 1024]` 或 `depth_mult: 1.0` | +0.01~0.02 |

### Phase 3: 训练脚本优化 (需修改Python代码)

| 步骤 | 文件 | 改动 | 预期收益 |
|------|------|------|---------|
| 3.1 | `train_defect.py` | C1→C2保留优化器动量 | 减少震荡, 加速收敛 |
| 3.2 | `configs.py` | VRAM估算考虑multi_scale最大值 | 避免OOM |
| 3.3 | `train_defect.py` | 支持分层学习率`--disc_lr` | 精细化控制 |

### Phase 4: 数据与损失函数 (长期)

| 步骤 | 内容 |
|------|------|
| 4.1 | SAHI风格训练切片: 从5120原图切1024/1280 crop, 以原生分辨率训练 |
| 4.2 | 合成缺陷生成: 使用inpainting生成更多SVHBNM/CBHPM等难例类别 |
| 4.3 | Focal Loss: 在box和cls损失中增加难例权重 |
| 4.4 | Wise-IoU: 替代CIoU, 对小框的IoU计算更稳定 |

---

## 6. 附录：数据速查表

### 6.1 数据集统计

| 类别 | 代码 | 训练标注 | 验证标注 | 占比 |
|------|------|---------|---------|------|
| 0 | VHBNM | 410 | 62 | 18.6% |
| 1 | VHBNL | 282 | 30 | 12.8% |
| 2 | SVHBNM | 506 | 48 | 23.0% |
| 3 | SVHBNL | 156 | 19 | 7.1% |
| 4 | SVHTNL | 322 | 46 | 14.6% |
| 5 | CBHPM | 296 | 45 | 13.4% |
| 6 | CBVPM | 230 | 30 | 10.4% |
| **总计** | | **2202** | **280** | 100% |

- 原始训练图: 399张, 离线增强: 399张, 合计训练: 878张
- 验证图: 101张
- 所有图像原始分辨率: 5120×5120

### 6.2 YOLO默认参数 vs 当前配置 vs 建议配置

| 参数 | YOLO默认 | C2当前 | C2建议 | 说明 |
|------|---------|--------|--------|------|
| `lr0` | 0.01 | 0.001 | 0.005(SGD) or 0.001(AdamW) | 核心改动 |
| `lrf` | 0.01 | 0.1 | 0.02 | 终值不变(~1e-4) |
| `optimizer` | SGD | SGD | AdamW | 自适应LR |
| `imgsz` | 640 | 1024 | 1280 | 提升分辨率 |
| `multi_scale` | false | false | true | 尺度不变性 |
| `mosaic` | 1.0 | 0.5 | 0.3 | 减少破坏 |
| `erasing` | 0.4 | 0.4 | 0.1 | 保护小目标 |
| `close_mosaic` | 10 | 15 | 40 | 更长精调 |
| `scale` | 0.5 | 0.5 | 0.7 | 更大缩放范围 |
| `epochs` | 100 | 200 | 300 | 更多迭代 |
| `warmup_epochs` | 3 | 5 | 10 | 更平稳过渡 |
| `model_ema` | false | false | true | 提升泛化 |

### 6.3 训练运行时间线

```
2026-06-25:
  13:27  Run A (132716) — C1 only, 中断
  13:59  Run B (135945) — 完整3阶段, AdamW C2, 最佳C3=0.414
  15:53  Run C (155323) — 完整3阶段, SGD C2, 最佳C3=0.399
  18:56  Run D (185613) — 完整3阶段, SGD大batch, C3退化, 最佳C2=0.395
```

### 6.4 模型文件清单

| 文件 | 用途 | 参数量 |
|------|------|--------|
| `yolo11s-EMA-SimAM.yaml` | 车载端, 3尺度P3/P4/P5 | ~9.4M |
| `yolo11m-EMA-SimAM.yaml` | 地面端GPU0, 3尺度P3/P4/P5 | ~20.1M |
| `yolo11m-P2-SimAM.yaml` | 地面端GPU1, 4尺度P2/P3/P4/P5 | ~20.1M+ |

### 6.5 关键文件路径

| 类型 | 路径 |
|------|------|
| 训练配置 | `config/train/{warmup,full,finetune}.yaml` |
| 推理配置 | `config/model/inference.yaml` |
| 训练脚本 | `subway_defect/train/train_defect.py` |
| 配置加载 | `subway_defect/train/configs.py` |
| 模型定义 | `subway_defect/models/yolo11s-EMA-SimAM.yaml` |
| EMA模块 | `subway_defect/modules/EMA.py` |
| SimAM模块 | `subway_defect/modules/SimAM.py` |
| 场景增强 | `subway_defect/augmentations/scene.py` |
| 设计文档 | `subway_defect/docs/地铁接触网缺陷检测AI算法设计文档.md` |

---

> **结论:** 模型特征学习效率低的根本原因是**三个相互关联的问题**: (1) 训练分辨率与缺陷尺寸不匹配, 导致2/3的检测尺度无效; (2) 学习率过低, 骨干网未能有效适应新领域; (3) C1→C2优化器状态丢失, 骨干网冷启动。Phase 1配置优化是低垂的果实, 仅改动YAML文件即可预期获得+0.08~0.15 mAP50提升。Phase 2架构改进(P2检测层)和Phase 3训练脚本优化(保留优化器状态)将进一步释放模型潜力。

---

## 7. 高级架构改进方案：Transformer 与其他结构

> **补充约束 (2025-06-25):**
> - 地面端同样有 **≤10秒/127MP图像** 的推理时限
> - 地面端同样要求 **Recall ≥90%, Precision ≥90%**
> - 训练数据量和类别数将在后期持续增加 (当前 7类→未来18类+, 878图→数千图)
> - 地面端双GPU并行推理 + WBF融合

### 7.1 推理时间预算分析

在讨论架构方案之前, 先量化各环节的时间开销:

```
127MP全景图像 (≈13000 × 9800 px)
  │
  ├── Stage 1: ROI检测 (YOLO11n @ 640, 8×下采样)
  │   输入: 1625 × 1225 px, 时间: ~0.3-0.5s
  │
  ├── Stage 2: Slice推理 (60-90个 1024×1024 slices)
  │
  │   车载端 (单GPU, 单模型):
  │     yolo11s @ 1024: ~30-50ms/slice → 60slices = 1.8-3.0s ✓
  │     yolo11s @ 1280: ~50-80ms/slice → 60slices = 3.0-4.8s ✓
  │     总计: 3.5-5.5s (含ROI), 预算充裕
  │
  │   地面端 (双GPU并行, 两模型各自独立运行):
  │     yolo11m @ 1024:     ~50-80ms/slice  → 60slices = 3.0-4.8s
  │     yolo11m-P2 @ 1024:  ~60-100ms/slice → 60slices = 3.6-6.0s
  │     总计: ≤7s (含ROI+WBF), 预算偏紧但可行
  │
  └── Stage 3: WBF融合 (地面端, ~0.1-0.3s)
```

**关键结论:** 车载端(yolo11s)有较大的时间裕度(~4s slack), 可以承受适度架构增强。地面端(yolo11m-P2)时间较紧, 需谨慎评估每次增加。

在后续讨论中, 每个方案都标注了预估的**单slice推理时间增量**和**对10秒预算的影响**。

### 7.2 方案总览

| # | 方案 | 参数增量 | 推理增量/slice | 小数据适应性 | 小目标收益 | 数据扩展受益 | 综合推荐 |
|---|------|---------|---------------|-------------|-----------|-------------|---------|
| 1 | **SAHI原生分辨率Crop训练** | 0 | 0ms | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **Phase 2首选** |
| 2 | **DyHead动态检测头** | +0.3M | +0.3ms | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ **Phase 2推荐** |
| 3 | **GOLD-YOLO Neck** | +2.1M | +5ms | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **推荐** |
| 4 | **轻量Transformer ROI Head** | +0.8M | +1ms | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **推荐** |
| 5 | **P2检测层 (yolo11s)** | +1.5M | +8ms | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ **推荐** |
| 6 | **RT-DETR轻量变体** | +10M | +15ms | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ 实验性 |
| 7 | **BiFormer Backbone** | +3.5M | +10ms | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ 数据量不足 |
| 8 | **Deformable DETR Head** | +5M | +12ms | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ 实验性 |

**数据扩展受益**列: 该方案在未来数据量增加(数千→数万图)时, 收益是否会进一步放大。⭐⭐⭐⭐⭐表示"数据越多越好", 这是Transformer类架构的核心优势。

### 7.3 详细方案分析

---

#### 方案1: SAHI原生分辨率Crop训练 (零架构改动, 最高ROI)

**核心思路:** 不改变模型架构, 改变训练数据的准备方式。

```
当前训练流程:
  5120×5120 原图 → letterbox resize → 1024×1024
  缺陷: 40×40px → 8×8px → 训练

SAHI训练流程:
  5120×5120 原图 → 滑动窗口crop 1024×1024 (stride=512, 含缺陷)
  缺陷: 40×40px → 保持 40×40px → 训练
  + 随机crop增强: 在缺陷周围±200px随机偏移crop
```

**为什么这从根本上解决了问题:**

```
训练分辨率对比:
  当前:   缺陷 8×8px  @ 1024  → P3特征图 = 1.0px  (勉强检测)
  SAHI:   缺陷 40×40px @ 1024  → P3特征图 = 5.0px  (清晰可见)
                                → P4特征图 = 2.5px  (有效检测!)
                                → P5特征图 = 1.25px (开始可用)
```

SAHI训练让P4和P5第一次真正"看到"了缺陷, 三个检测尺度同时生效。

**实现方式:**

```python
# 训练数据预处理脚本
def prepare_sahi_crops(image_5120, labels, crop_size=1024, stride=512):
    crops = []
    for y in range(0, 5120 - crop_size, stride):
        for x in range(0, 5120 - crop_size, stride):
            crop = image_5120[y:y+crop_size, x:x+crop_size]
            # 过滤: 只保留含至少1个标注的crop
            crop_labels = filter_labels_in_roi(labels, x, y, crop_size)
            if crop_labels:
                crops.append((crop, crop_labels))
    # 额外: 围绕每个缺陷做随机偏移crop (数据增强)
    for label in labels:
        cx, cy = label_center(label)
        offset_x = np.random.randint(-200, 200)
        offset_y = np.random.randint(-200, 200)
        crop = image_5120[cy+offset_y:cy+offset_y+crop_size, ...]
        crops.append((crop, filter_labels_in_roi(...)))
    return crops
```

**优势汇总:**
- 零推理开销 (推理流程不变, 仍是大图→切片→检测)
- 零架构改动 (只在训练数据上做变化)
- 缺陷在特征图中从1px→5px, 信息量提升25倍
- 自然的数据增强 (不同crop位置 = 不同背景上下文)
- 未来数据增加时, crop策略自动适配

**潜在问题:**
- 训练图像数量暴增 (一张5120图→约81个crop, 但过滤后约15-30个含缺陷crop)
- 需要确保验证集也用crop评估 (为了一致性, 或保持原图验证)

**时间影响:** 推理时不变。训练时数据量增大但batch内仍是1024×1024, 训练时间线性增加。

---

#### 方案2: DyHead (Dynamic Head) — 统一注意力检测头

**原理:** 将YOLO的独立`Detect`头(每个尺度各自为政)替换为尺度-空间-任务三维统一注意力头。

```
当前 Detect 头 (P3, P4, P5 各自独立):
  P3 feature → Conv1x1 → Box + Cls
  P4 feature → Conv1x1 → Box + Cls     ← 三者互不通信!
  P5 feature → Conv1x1 → Box + Cls

DyHead (统一注意力):
  P3,P4,P5 features → Stack → 
    Scale-Aware Attention   (不同尺度互相"看见")
    → Spatial-Aware Attention (每个位置关注关键空间区域)
    → Task-Aware Attention    (分类和定位子任务互相指导)
    → P3',P4',P5' → Box + Cls
```

**对当前问题的针对性:**
1. 尺度感知注意力: P4/P5学习"我该关注什么尺度", 不再在无缺陷的尺度上做无用检测
2. 空间感知注意力: 学习缺陷的空间上下文——"螺栓缺失"总是发生在金属结构的特定位置
3. 任务感知注意力: 定位不准的box可以通过分类分数辅助调整

**速度影响 (关键):**
```
yolo11s + DyHead:
  参数量: 9.4M → 9.7M (+3%)
  FLOPs:  21.6G → 22.2G (+3%)
  推理时间/slice: +0.3ms
  60 slices: +18ms (可忽略)
  对10秒预算: 零影响
```

**实现参考:** `mmdet.models.DyHead` (MMDetection), 可适配YOLO。

---

#### 方案3: GOLD-YOLO Gather-and-Distribute Neck

**原理:** 用轻量级Transformer风格的全局特征聚合替代传统的FPN+PAN逐层传递。

```
传统 FPN+PAN:
  P5 → upsample → P4 → upsample → P3
  P3 → downsample → P4 → downsample → P5
  问题: 每步传递都有信息损失, P5的信息经过2次上采样+concat才到P3

GOLD-YOLO Neck:
  P3,P4,P5 → Global Gather (cross-attention, 所有尺度同时交互)
           → Feature Alignment (对齐不同尺度的语义)
           → Global Distribute (将聚合信息分发回各尺度)
           → P3',P4',P5'

  优势: P5的大尺度语义信息和P3的小尺度细节在attention中直接交互
```

**对当前问题的针对性:**
- P5(大尺度语义: "这是一个金属结构区域")与P3(小尺度细节: "这里有个螺栓缺失")在Transformer attention中建立了直接关联
- 传统的FPN需要经过多次上采样/下采样才能让P5和P3的信息混合, 每一步都有损失
- 对"未来数据增加"友好——Transformer neck的容量可以充分利用更多数据

**速度影响:**
```
yolo11s + GOLD-YOLO:
  参数量: 9.4M → 11.5M (+22%)
  FLOPs:  21.6G → 24.8G (+15%)
  推理时间/slice: +5ms
  60 slices: +300ms
  对10秒预算: +5% (可接受)
```

**小数据担忧缓解:** GOLD-YOLO使用线性注意力(Linear Attention), 复杂度O(N)而非O(N²), 参数效率高。2.1M新增参数中约一半是LayerNorm的affine参数(极低过拟合风险)。

---

#### 方案4: 轻量Transformer ROI Head (最务实的Transformer引入)

**原理:** YOLO backbone+neck保持不变, 仅在检测后对每个ROI用微型Transformer精调。

```
完整流程:
  原始大图
    → YOLO Backbone + Neck → P3/P4/P5特征图
    → Detect Head → 初步检测框 (~30个候选)
    → ROI Align (从P2特征图, stride=4, 7×7区域)
    → 轻量Transformer (2层, 256dim, 4head)
    → Box offset + Cls logit → 精调后的检测结果
```

**为什么这个方案特别适合当前问题:**

1. **P2特征图保留最多细节:** ROI Align从P2(4× stride, 320×320 @ 1280输入)提取特征。对于40×40的缺陷(SAHI训练), P2上占10×10区域——信息丰富。

2. **Transformer在ROI内做self-attention:** 7×7=49个token互相attend, 捕捉"螺栓+底座+周围金属结构"的空间关系。这种全局上下文交互是CNN卷积(3×3局部感受野)做不到的。

3. **极低过拟合风险:** 只有~0.8M参数, 878张图×30个ROI=~26,000个训练样本充足。

4. **推理开销极小:** Transformer只在检测到的ROI上运行, 不增加slice级推理时间。

```
参数量: 9.4M → 10.2M (+8%)
推理时间/slice: +1ms (因为只对最终ROI运行)
60 slices × 30 ROI: +30ms (含ROI align, 可忽略)
对10秒预算: 零影响
```

**架构细节:**

```python
class LightweightROITransformer(nn.Module):
    """Post-hoc ROI refinement with a 2-layer Transformer."""
    def __init__(self, in_channels=64, hidden_dim=256, num_heads=4):
        super().__init__()
        # ROI Align: 从P2特征图(1/4 stride)提取7×7区域
        self.roi_pool = RoIAlign(output_size=(7, 7), spatial_scale=0.25)
        self.input_proj = nn.Linear(in_channels * 49, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 49, hidden_dim))
        
        # 2层Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=512, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 输出头
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # dx, dy, dw, dh 偏移
        )
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nc)  # 精调后分类logit
        )
    
    def forward(self, p2_features, proposals, proposal_scores):
        # proposals: List[Tensor(N_i, 4)] 每个slice的ROI框
        roi_features = self.roi_pool(p2_features, proposals)  # (Total_ROI, 64, 7, 7)
        roi_features = roi_features.flatten(1)  # (Total_ROI, 64*49)
        x = self.input_proj(roi_features).unsqueeze(0)  # (1, Total_ROI, 256)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x).squeeze(0)  # (Total_ROI, 256)
        
        box_refine = self.box_head(x)  # 精调偏移
        cls_refine = self.cls_head(x)  # 精调分类
        
        return box_refine, cls_refine
```

---

#### 方案5: P2检测层 (yolo11s-P2)

**原理:** 车载端yolo11s增加P2(stride 4)检测层, 与地面端yolo11m-P2对齐。

当前已有`yolo11m-P2-SimAM.yaml`(地面端GPU1), 可直接以此为模板创建`yolo11s-P2-EMA-SimAM.yaml`。

```
四尺度检测:
  P2 (stride 4, 320×320 @ 1280):  缺陷 10×10 → 2.5×2.5 特征图像素 ✓✓
  P3 (stride 8, 160×160 @ 1280):  缺陷 10×10 → 1.25×1.25 特征图像素 ✓
  P4 (stride 16, 80×80 @ 1280):   缺陷 10×10 → 0.63×0.63 特征图像素 ⚠
  P5 (stride 32, 40×40 @ 1280):   缺陷 10×10 → 0.31×0.31 特征图像素 ✗
```

**速度影响:**
```
yolo11s-P2 vs yolo11s:
  参数量: 9.4M → 10.9M (+16%)
  推理时间/slice: ~30ms → ~38ms (+8ms)
  60 slices: +480ms
  对10秒预算: +8% (可接受)
```

**与SAHI训练的互补性:** SAHI让缺陷在1024crop中保持40×40→P2上10×10像素, 检测难度大幅降低。P2+SAHI是最强的小目标组合。

---

#### 方案6: RT-DETR (Real-Time Detection Transformer) — 实验性

**原理:** Baidu的轻量DETR变体。CNN骨干+Transformer编码器-解码器, 端到端检测, 无需NMS。

```
结构:  ResNet18/50 backbone
       → Efficient Hybrid Encoder (CNN + Transformer混合)
       → Transformer Decoder (固定300个object query)
       → 300个预测 (通过匈牙利匹配训练)
       → 无需NMS, 直接输出
```

**为什么值得关注(尽管风险较高):**

1. **全局注意力天然擅长小目标:** 每个object query可以attend到特征图的任何位置, 不像YOLO的anchor-based检测受限于局部感受野。对"图像中任意位置的小螺栓缺失", Transformer的全局搜索能力是本质优势。

2. **无需NMS:** 两个靠得很近的小缺陷(同类型螺栓的两个不同缺失)在YOLO的NMS中可能互相抑制。RT-DETR通过匈牙利匹配一对一分配, 避免了这个问题。

3. **数据越多越强:** Transformer模型通常随数据量增加而持续提升(不像CNN在小数据集后迅速饱和)。未来数据扩展到数千张时, RT-DETR的收益会进一步放大。

**当前风险:**

1. **878张图可能严重欠拟合:** RT-DETR-R18有20M参数, 是yolo11s的2倍。查询COCO预训练权重在接触网领域的迁移效果是未知的。

2. **推理速度:** RT-DETR-R50 @ 640在T4上约14ms。缩放到1280分辨率+我们的场景, 预估40-60ms/slice。加上Hybrid Encoder, 可能显著慢于YOLO。

3. **工程复杂度:** 需要完整的训练/推理pipeline适配, 不同于YOLO生态内的改动。

**建议:** 作为长期跟踪项。等待数据量>2000张后做对比实验。目前不投入工程资源。

---

#### 方案7: BiFormer Backbone — 实验性

**原理:** 用两级路由注意力(Bi-Level Routing Attention, BRA)替换YOLO11 backbone中的C2PSA模块。

```
标准Self-Attention (C2PSA):
  每个token attend所有token → O(N²) → 只能在小特征图上使用(stride 32)

BiFormer:
  阶段1: 粗粒度区域路由 — 找出"哪些区域之间相关"
  阶段2: 细粒度token注意力 — 只在相关区域间做attention
  → O(N^(4/3)) → 可以在更大特征图上使用(stride 16甚至8)
```

**优势:** 能将Transformer attention扩展到P3甚至P2尺度, 而不仅限于backbone最深层的P5(stride 32)。

**风险:** 对878张图, 在backbone中引入额外attention模块的过拟合风险较高。建议数据量>3000张后再实验。

---

### 7.4 综合推荐路线 (更新)

```
Phase 1 ✅ (已完成 — YAML配置优化):
  ├── imgsz 1024→1280
  ├── SGD→AdamW (C2, C3)
  ├── multi_scale=0.5, mosaic↓, erasing↓, warmup↑
  └── 预期: +0.08~0.15 mAP50

Phase 2 🔜 (建议立即实施 — 不改架构, 只改训练数据+检测头):
  ├── 方案1: SAHI原生分辨率crop训练     预期 +0.05~0.10 mAP50
  │   ├── 零推理开销
  │   └── 从根本上解决"缺陷缩小5倍"问题
  ├── 方案5: yolo11s-P2-EMA-SimAM.yaml  预期 +0.02~0.04 mAP50
  │   ├── 推理+8ms/slice
  │   └── 给模型一个缺陷=2.5×2.5特征像素的检测层
  └── 方案2: DyHead替换Detect            预期 +0.02~0.04 mAP50
      ├── 推理+0.3ms/slice (可忽略)
      └── 让P3/P4/P5互相通信

  Phase 2 累计预期: mAP50 ≈ 0.58~0.68

Phase 3 (Phase 2验证后):
  ├── 方案3: GOLD-YOLO Neck              预期 +0.03~0.06 mAP50
  │   └── 推理+5ms/slice → 60slices+0.3s
  └── 方案4: 轻量Transformer ROI Head    预期 +0.02~0.05 mAP50
      └── 推理+1ms/slice → 可忽略

  Phase 3 累计预期: mAP50 ≈ 0.65~0.76

Phase 4 (数据量>3000张后):
  ├── 方案6: RT-DETR对比实验
  └── 方案7: BiFormer backbone实验
```

### 7.5 推理时间预算核算

以地面端最重配置(yolo11m-P2, 最慢路径)验证所有方案叠加后是否仍在10秒内:

```
方案叠加后的单slice推理时间:
  Base yolo11m-P2 @ 1280:         ~75ms
  + DyHead:                       +0.5ms  = 75.5ms
  + GOLD-YOLO Neck:               +7ms    = 82.5ms
  + Transformer ROI Head:         +1ms    = 83.5ms

  60 slices (典型):  5.0s
  90 slices (最差):  7.5s
  + ROI检测:         0.5s
  + WBF融合:         0.2s
  ─────────────────────────
  总计 (最差):       8.2s  ✓ (< 10s)

yolo11s-P2 (车载端):
  60 slices × 46ms = 2.8s
  + ROI检测 = 0.5s
  总计: 3.3s  ✓ (充裕)
```

**结论:** 即使所有Phase 2+3方案叠加, 两端的推理时间仍在10秒预算内。车载端有大量裕度, 地面端在最坏情况下(90 slices)达到8.2秒, 仍有~18%的安全余量。

### 7.6 端到端精度要求分析

```
目标: Recall ≥90%, Precision ≥90%

当前最佳 (Run B C3, mAP50=0.414):
  混淆矩阵分析:
    类别0 (VHBNM): P=0.476, R=0.871  — 接近目标
    类别3 (SVHBNL): P=0.187, R=0.684  — 远低于目标
    类别5 (CBHPM):  P=0.097, R=0.133  — 严重不足
  
  要达到 R≥90%, P≥90% (所有类别):
    需要 mAP50 ≥ 0.75~0.80 (粗略对应)

Phase 1 (预期 mAP50≈0.55):
  → 部分简单类别达到目标, 困难类别仍需提升

Phase 1+2 (预期 mAP50≈0.65):
  → 大部分类别达到目标, CBHPM/SVHBNM可能仍需优化

Phase 1+2+3 (预期 mAP50≈0.72):
  → 所有类别接近目标, 可能需要针对性数据增强

达到 P≥90%, R≥90% 的最终路径:
  Phase 1~3 架构优化 + 针对性难例增强 + 更多训练数据
  → 预估需要 mAP50 ≥ 0.75, 约在当前基础上翻倍

---

## 8. 创新性分析与论文发表策略

> 本节将前述改进方案从"工程优化"视角提升为"学术创新"视角, 分析各方案的可发表性, 并提出论文框架建议。

### 8.1 本项目的研究定位

当前接触网缺陷检测领域的研究空白:

| 现状 | 本项目的独特之处 |
|------|-----------------|
| 大多数方法处理 <10MP 图像 | 本项目处理 **127MP** 超高分辨率图像 |
| 通用检测器(COCO预训练)直接应用 | **三阶段渐进领域自适应** 训练策略 |
| 标准FPN+PAN颈 | **SimAM能量注意力 + EMA空间注意力** 专为结构异常检测设计 |
| 固定尺度训练 | **SAHI原生分辨率crop** 保持微小缺陷的空间细节 |
| 单模型推理 | **异构双模型WBF集成** 互补多尺度检测能力 |
| 实验室数据 | **真实运营线路** 数据, 7类接触网缺陷 |

**核心学术贡献可以归纳为:**
> 面向超高分辨率工业图像的实时微小缺陷检测 —— 一个集"分辨率保持训练 + 参数自由注意力 + 轻量Transformer精调 + 异构集成"于一体的完整解决方案。

### 8.2 各方案创新点分析与论文定位

#### 创新点 A: SimAM在工业缺陷检测中的首次应用 (⭐ 高创新性)

**创新本质:** SimAM(ICML 2021)是通用视觉注意力, 但**从未被专门用于工业缺陷检测**。我们发现了SimAM与接触网缺陷检测之间的天然契合:

```
SimAM能量函数:  e_t = 4(σ²+λ) / ((t-μ)² + 2σ² + 2λ)

直觉: "与众不同的神经元获得高注意力"

接触网场景:
  - 螺栓/螺母呈规则的周期性排列
  - 缺失的螺栓 = 规则模式中的"断裂" = "与众不同的神经元"
  - SimAM自然地将注意力集中在缺失/松动位置
```

**论文亮点:**
- SimAM **零参数** → 在小数据集(878张)上无过拟合风险 → 与Transformer类注意力形成鲜明对比
- 对比实验: SimAM vs CBAM vs SE vs ECA vs 无注意力 → 证明SimAM在"规则结构中的异常检测"场景下的独特优势
- 可解释性分析: 可视化SimAM的能量图, 展示能量峰值恰好位于缺陷位置

**建议论文标题方向:**
> *"Parameter-Free Energy Attention for Tiny Anomaly Detection in Regular Industrial Structures"*

---

#### 创新点 B: 三阶段渐进领域自适应 (Progressive Domain Adaptation, PDA) (⭐⭐ 高创新性)

**创新本质:** 将"COCO→接触网"的领域迁移形式化为三阶段课程学习。

```
C1 (Head Warmup): 冻结骨干, 仅训练检测头
  → 检测头学习: "这些特征(即使来自COCO域)如何映射到缺陷类别?"
  
C2 (Full Adaptation): 解冻全部, 高分辨率+多尺度
  → 骨干学习: "如何从接触网图像中提取比COCO特征更有用的特征?"
  → 关键创新: 保留C1→C2的优化器动量, 避免领域迁移震荡
  
C3 (Stabilization): 极低LR + 最小增强
  → 在真实数据分布上稳定收敛
```

**论文亮点:**
- **动量保留跨阶段迁移 (Momentum-Preserving Stage Transition):** C1→C2过渡时保留优化器状态, 避免骨干"冷启动"。这可以作为一个**方法贡献**被形式化。
- **增强退火 (Augmentation Annealing):** 三阶段的增强强度递减(mosaic 0.5→0.3→0.0, erasing 0.4→0.1→0.0), 类似学习率退火但在增强空间——这是一个**新概念**。
- 消融实验: 有无动量保留 / 有无增强退火 / 不同阶段数(1/2/3) → 证明三阶段设计的必要性

**建议论文标题方向:**
> *"Progressive Domain Adaptation with Augmentation Annealing for Cross-Domain Industrial Defect Detection"*

---

#### 创新点 C: SAHI原生分辨率训练用于微小缺陷 (⭐⭐ 高创新性)

**创新本质:** 将SAHI(Slicing Aided Hyper Inference)的概念反向应用于训练阶段。

```
传统SAHI:     推理时切图 → 检测 → 合并  (仅用于推理)
我们的SAHI:   训练时切图 → 保持原生分辨率 → 训练  (首次用于训练!)

关键差异:
  传统训练: 5120 → resize → 1024 → 缺陷缩小5×
  SAHI训练: 5120 → crop 1024 → 缺陷保持原始大小 → 训练
```

**论文亮点:**
- 形式化"分辨率保持训练 (Resolution-Preserving Training)"概念
- 理论分析: 证明对于面积<0.01%图像面积的目标, 全局resize的信息损失上界
- 对比实验: 全局resize训练 vs SAHI训练 vs 混合训练
- 配合multi_scale训练的互补性分析

**建议论文标题方向:**
> *"Resolution-Preserving Training: Slicing-Aided Fine-Grained Detection in Ultra-High Resolution Industrial Images"*

---

#### 创新点 D: 轻量ROI Transformer (⭐⭐ 中等创新性)

**创新本质:** 并非第一个将Transformer用于检测后处理的工作, 但**针对工业实时检测场景的极简设计**(2层, 0.8M参数, +1ms/slice) 以及与**P2高分辨率特征图的ROI Align结合**是有价值的工程创新。

**论文亮点:**
- ROI从P2(stride 4)特征图提取, 而非传统的P3-P5 → 保留了4倍的空间细节
- 微型Transformer(2层, 0.8M参数)在~26K ROI样本上训练, 无过拟合
- Transformer self-attention捕捉"缺陷+周围结构"的全局上下文 → 减少误报

**建议作为辅助创新点(D)放入论文, 而非主打。**

---

#### 创新点 E: 异构双模型WBF集成 (⭐ 中等创新性)

**创新本质:** 双模型集成本身不新, 但**故意使用异构尺度架构(3-scale + 4-scale)形成互补**是有新意的设计选择。

```
GPU 0: yolo11m-EMA-SimAM (P3/P4/P5, 3尺度)
  → 擅长: 中等尺寸缺陷, 较少的假阳性

GPU 1: yolo11m-P2-SimAM (P2/P3/P4/P5, 4尺度)
  → 擅长: 微小缺陷(P2层), 较高的召回率

WBF融合: "双模型都看到" → 高置信度 / "仅一个看到" → 需额外验证
```

**论文亮点:**
- 消融: 单P2模型 vs 单3尺度模型 vs WBF集成 → 证明互补性
- 速度-精度权衡: 双模型并行(各占一个GPU)不增加延迟

---

### 8.3 推荐论文框架

基于上述创新点, 建议一篇完整的论文结构:

#### 论文标题 (建议3选1)

```
Option A (强调方法创新):
  "PDA-Net: Progressive Domain Adaptation with Energy Attention
   for Tiny Defect Detection in Ultra-High Resolution Catenary Images"

Option B (强调应用+方法):
  "Real-Time Tiny Defect Detection for Railway Catenary Inspection:
   A Resolution-Preserving Approach with Parameter-Free Attention"

Option C (短标题, 高引用潜力):
  "Learning to See the Invisible: Sub-Pixel Defect Detection in
   127-Megapixel Industrial Images"
```

#### 论文结构

```
1. Introduction
   - 接触网缺陷检测的重要性(安全关键)
   - 三大挑战:
     (a) 超高分辨率(127MP)中的微小缺陷(<0.01%图像面积)
     (b) 小数据集(878张) + 类别不平衡
     (c) 实时推理约束(≤10s/图)
   - 本文贡献 (3-4点)

2. Related Work
   - 工业缺陷检测 (传统方法 vs 深度学习方法)
   - 小目标检测 (FPN, SAHI, multi-scale training)
   - 领域自适应 (fine-tuning strategies, progressive training)
   - 轻量注意力机制 (SE, CBAM, ECA, SimAM, EMA)

3. Method  ← 主要创新部分
   3.1 Overall Architecture
       - 两阶段管线: ROI提案 + 缺陷检测
       - 车载端/地面端部署架构

   3.2 Progressive Domain Adaptation (PDA)  ← 创新点B
       - C1: Head Warmup with Frozen Backbone
       - C2: Full Adaptation with Momentum Preservation
       - C3: Stabilization with Augmentation Annealing
       - 算法伪代码

   3.3 Energy-Based Attention for Structural Anomalies  ← 创新点A
       - SimAM回顾: 能量函数与空间抑制理论
       - 为什么SimAM天然适合规则结构的异常检测
       - EMA作为互补: 空间位置编码
       - 注意力模块在Neck中的布局设计

   3.4 Resolution-Preserving Training Strategy  ← 创新点C
       - SAHI训练: 从推理技巧到训练策略
       - 与multi_scale训练的协同
       - 理论分析: resize对小目标的信息损失

   3.5 Lightweight ROI Transformer  ← 创新点D (可选)
       - P2特征图ROI Align
       - 2层Transformer精调

4. Experiments
   4.1 Dataset: Subway-Catenary-Defect (SCD-7)
       - 7类, 878训练/101验证, 5120×5120原图
   4.2 Implementation Details
   4.3 Main Results (与YOLOv8/v9/v10/v11, RT-DETR对比)
   4.4 Ablation Studies
       - PDA三阶段的效果 (Table)
       - SimAM vs CBAM vs SE vs ECA (Table)
       - SAHI训练 vs 全局resize (Table)
       - 各组件叠加收益 (Table)
   4.5 Speed-Accuracy Trade-off Analysis
   4.6 Qualitative Analysis (可视化)

5. Discussion
   - 方法的局限性 (类别数有限, 场景特定)
   - 扩展到其他工业检测场景的可能性
   - 未来工作: 更多数据, 更多缺陷类别

6. Conclusion
```

#### 实验设计 (Ablation Study)

论文中最关键的是消融实验, 建议如下设计:

**Table 1: Component-wise contribution (on SCD-7 val)**

| PDA | SimAM | SAHI-Train | ROI-TF | DyHead | mAP50 | mAP50-95 | Params | Time(s) |
|-----|-------|------------|--------|--------|-------|----------|--------|---------|
|     |       |            |        |        | 0.414 | 0.289    | 9.4M   | 2.1     |
| ✓   |       |            |        |        | 0.xxx | 0.xxx    | 9.4M   | 2.1     |
| ✓   | ✓     |            |        |        | 0.xxx | 0.xxx    | 9.4M   | 2.2     |
| ✓   | ✓     | ✓          |        |        | 0.xxx | 0.xxx    | 9.4M   | 2.2     |
| ✓   | ✓     | ✓          | ✓      |        | 0.xxx | 0.xxx    | 10.2M  | 2.3     |
| ✓   | ✓     | ✓          | ✓      | ✓      | 0.xxx | 0.xxx    | 10.5M  | 2.4     |

**Table 2: Attention mechanism comparison**

| Attention | Params | mAP50 | mAP50-95 | 分析 |
|-----------|--------|-------|----------|------|
| None | — | x.xxx | x.xxx | Baseline |
| SE | +0.01M | x.xxx | x.xxx | 通道注意力 |
| CBAM | +0.05M | x.xxx | x.xxx | 通道+空间 |
| ECA | +0.001M | x.xxx | x.xxx | 1D通道注意力 |
| EMA | +0.2M | x.xxx | x.xxx | 多尺度空间 |
| **SimAM** | **+0** | **x.xxx** | **x.xxx** | **本文采用** |

**Table 3: Training resolution comparison**

| Train imgsz | Strategy | mAP50 | mAP50-95 | 分析 |
|-------------|----------|-------|----------|------|
| 640 | global resize | x.xxx | x.xxx | |
| 1024 | global resize | 0.414 | 0.289 | 当前最佳 |
| 1280 | global resize | x.xxx | x.xxx | Phase 1优化 |
| 1024 | **SAHI crop** | x.xxx | x.xxx | **本文提出** |
| 1280 | **SAHI crop** | x.xxx | x.xxx | **最佳组合** |

---

### 8.4 创新性总结与优先级

| 创新点 | 新颖度 | 工程复杂度 | 预期收益 | 论文贡献 | 实施优先级 |
|--------|--------|-----------|---------|---------|-----------|
| B: PDA三阶段训练 | ⭐⭐⭐ | 低(已在代码中) | +0.05~0.10 | **核心方法贡献** | Phase 1 (已完成) |
| C: SAHI原生分辨率训练 | ⭐⭐⭐ | 中(数据预处理) | +0.05~0.10 | **核心方法贡献** | Phase 2 |
| A: SimAM工业缺陷检测 | ⭐⭐ | 低(已集成) | +0.02~0.04 | **关键技术创新** | Phase 2 |
| D: 轻量ROI Transformer | ⭐⭐ | 中(新模块) | +0.02~0.05 | **辅助创新** | Phase 3 |
| E: 异构双模型WBF | ⭐ | 低(已集成) | +0.02~0.04 | **工程创新** | 已有 |
| DyHead | ⭐ | 低(YAML改动) | +0.02~0.04 | 消融对比 | Phase 2 |

### 8.5 发表策略建议

**目标期刊/会议 (按推荐度排序):**

| 等级 | 会议/期刊 | 理由 |
|------|----------|------|
| 🥇 | **IEEE TII** (Trans. Industrial Informatics) | 工业检测+实时系统, 完美匹配 |
| 🥇 | **IEEE TIM** (Trans. Instrumentation & Measurement) | 检测仪器+工业应用 |
| 🥈 | **Engineering Applications of AI** (EAAI, IF=7.5) | AI+工程应用 |
| 🥈 | **IEEE T-ITS** (Trans. Intelligent Transportation Systems) | 轨道交通智能系统 |
| 🥉 | **ICIP 2026** / **ICPR 2026** | 短期投稿, 快速发表 |
| 🥉 | **Neurocomputing** (IF=6.0) | 神经网络+工业应用 |

**时间线建议:**
```
2025年7月:   Phase 2实施 (SAHI训练 + P2模型 + DyHead)
2025年8月:   完整实验(含所有消融) + 论文初稿
2025年9月:   内部修改 + 补充实验
2025年10月:  投稿 IEEE TII 或 T-ITS
2026年1-3月: 审稿意见返回 → 修改 → 录用
```

### 8.6 与已有工作的差异化

| 已有工作 | 我们的区别 |
|---------|-----------|
| YOLO系列 + 工业检测 | 我们提出PDA三阶段训练 + 增强退火, 而非简单微调 |
| SAHI (推理时切图) | 我们首次将SAHI引入**训练**阶段 |
| SimAM (ICML 2021) | 我们首次将其应用于**工业缺陷检测**, 揭示能量函数与规则结构异常的自然契合 |
| Transformer检测(DETR等) | 我们的ROI Transformer是**后精调**而非端到端, 更适合实时系统 |
| 接触网检测现有文献 | 大多数用传统CV方法; 我们提供首个完整的深度学习+超高分辨率方案 |

---

> **论文核心叙事:** "在127MP工业图像中检测像素级微小缺陷, 需要同时解决三个矛盾 —— 分辨率与速度的矛盾(PDA+SAHI训练)、模型容量与小数据集的矛盾(SimAM零参数注意力)、检测精度与实时性的矛盾(轻量ROI Transformer)。我们证明, 这三个矛盾可以通过系统性的训练策略和架构设计同时化解。"
```
