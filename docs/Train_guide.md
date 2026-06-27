# 地铁接触网缺陷检测系统

基于 YOLO11 + EMA/SimAM 注意力的两阶段 AI 缺陷检测系统，用于福州地铁接触网超高清图像（1.27 亿像素）的智能化分析。

> 完整规格说明见 [SPECIFICATION.md](../SPECIFICATION.md)

## 项目概述

本系统通过车载高速相机采集的接触网图像，自动识别螺栓松动/脱落、开口销缺失、绝缘子破损等 18 类缺陷，替代人工巡检。系统支持**车载端**（单 RTX 4090，离线运行）和**地面端**（双 RTX 4090，WBF 融合）两种部署形态。

### 核心指标

| 指标 | 车载端 | 地面端 |
| --- | --- | --- |
| GPU 配置 | 单卡 RTX 4090 | 双卡 RTX 4090 |
| 输入规格 | 1.27 亿像素（~13000×9800） | 1.27 亿像素 |
| 单张推理耗时 | ≤ 10 秒 | ≤ 10 秒 |
| 检出率 Recall | ≥ 90% | ≥ 90% |
| 准确率 Precision | ≥ 90% | ≥ 90% |
| 模型加载延迟 | ≤ 1 秒 | ≤ 1 秒 |
| 提报率 | — | ≤ 5% |
| 部署形态 | 完全离线（工控机） | 内网服务 |

## 项目结构

```
Subway_defect_detection_main/
├── subway_defect/                    # 项目主包
│   ├── modules/                      # EMA、SimAM、ECA 注意力模块
│   ├── models/                       # 模型 YAML 配置文件（4 个变体）
│   ├── pipeline/                     # 推理管道（切片器、两阶段、WBF 融合）
│   ├── train/                        # 训练模块（超参数预设、CLI 脚本、回调系统）
│   ├── augmentations/                # 数据增强（场景模拟、CopyPaste）
│   ├── deployment/                   # 部署（TensorRT 导出、FastAPI 服务）
│   ├── synthetic/                    # 合成数据生成（Inpainting）
│   └── classes.py                    # 缺陷类别中央注册表 (单一事实来源)
├── subway_yolo/                      # Vendored YOLO 框架（已精简）
│   ├── engine/                       # Model、Trainer、Predictor、Validator、Exporter
│   ├── nn/                           # tasks、modules、Extramodule（EMA/SimAM 桥接）
│   ├── models/yolo/                  # 仅 detect + classify
│   ├── data/                         # 数据加载、增强
│   ├── cfg/                          # 配置 + YOLO11 模型定义
│   ├── utils/                        # 核心工具函数
│   └── optim/                        # 优化器
├── tests/                            # 测试套件
│   ├── test_attention_modules.py     # EMA/SimAM/ECA 单元 + 模型集成
│   ├── test_augmentations.py         # 增强管道 + 训练配置
│   └── test_pipeline.py              # 切片器 + WBF 融合 + 部署
├── scripts/                                  # 数据集工具脚本 + AutoDL 配置
│   ├── setup_autodl.sh                    # AutoDL 云平台环境配置
│   ├── prepare_dataset.py                # 一键自制数据集准备
│   ├── split_dataset.py                  # 按源图分组 train/val 划分
│   ├── generate_native_crops.py          # 原生分辨率 crop 生成 (v2: 类别平衡 + 位置去偏置)
│   ├── validate_dataset.py               # 数据集完整性校验
│   ├── multi_source_dataset_builder.py   # 多源公开数据集构建器 (v2: +SDD2+RSDDs -DeepPCB)
│   ├── multi_source_pretrain_yaml.py     # 多阶段训练配置生成器 (v2: 1A/1B 拆分)
│   ├── setup_coco_from_autodl.py         # COCO 数据集设置 (AutoDL)
│   ├── generate_scene_augmentations.py   # 场景增强 (v2: +震动+白平衡, 类别感知)
│   ├── generate_synthetic_defects.py     # Inpainting 合成缺陷
│   ├── generate_defect_copy_paste.py     # ★ 缺陷感知 Copy-Paste (v2 新增)
│   ├── collect_hard_negatives.py         # ★ 难负样本收集 (v2 新增)
│   ├── calibrate_thresholds.py           # ★ 每类阈值校准 (v2 新增)
│   ├── audit_labels.py                    # ★ 标注质量审计 (v2 新增)
│   ├── fix_classes_txt.py                # classes.txt 自动修复
│   └── create_defect_data_yaml.py        # defect_data.yaml 创建
├── config/                                 # YAML 集中配置
│   ├── model/inference.yaml                # 推理参数
│   └── train/                              # 训练超参数 (Legacy + Modern)
│       └── pretrain/                       # 五阶段预训练配置
├── test_fixtures/                           # 测试夹具 (coco8/dota8/imagenet10, 按需下载)
├── weights/                                 # 预训练权重 (yolo11n/s/m + yolo26n)
├── data/                                    # 项目训练数据 (gitignored)
│   ├── Defect_dataset/                      # 自制接触网数据集
│   └── multi_datasets/                      # 多源公开数据集
├── docs/                                    # 设计文档 + 接口规范
├── pyproject.toml                    # 项目配置（包名 subway_defect）
├── README.md                         # 本文件
├── SPECIFICATION.md                  # 完整规格说明书
└── LICENSE                           # AGPL-3.0
```

## 环境要求

- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.0（推荐 CUDA 12.1）
- **GPU**: NVIDIA GPU，VRAM ≥ 8 GB（推荐 RTX 4090）
- **操作系统**: Windows 10/11 Pro（车载端/地面端）/ Linux（训练端）

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Subway_defect_detection_main

# 安装（可编辑模式，会同时安装 subway_defect 和 subway_yolo 两个包）
pip install -e .

# 验证安装
python -c "from subway_defect.modules.EMA import EMA; from subway_defect.modules.SimAM import SimAM; print('OK')"
```

### 2. 验证模型构建

```bash
# 验证四个模型 YAML 均可正常构建
python -c "
from subway_yolo import YOLO
for cfg in ['subway_defect/models/yolo11s-EMA-SimAM.yaml',
            'subway_defect/models/yolo11s-P2-EMA-SimAM.yaml',
            'subway_defect/models/yolo11m-EMA-SimAM.yaml',
            'subway_defect/models/yolo11m-P2-SimAM.yaml']:
    model = YOLO(cfg)
    print(f'{cfg}: {sum(p.numel() for p in model.model.parameters()):,} params')
"
```

### 3. 运行测试

```bash
# 运行全部项目测试
pytest tests/ -v

# 按模块运行
pytest tests/test_attention_modules.py -v   # EMA/SimAM 模块 + 模型集成
pytest tests/test_augmentations.py -v       # 增强管道 + 配置
pytest tests/test_pipeline.py -v            # 切片器 + WBF + 部署

# 使用 --slow 标志运行慢速集成测试
pytest tests/ --slow -v
```

## 架构设计

### 两级级联推理管道

```
127MP 原始图像 (13000×9800)
       │
       ▼
┌─────────────────────┐
│ Stage 1: ROI 提案器  │  ← 降采样 1/8 (~1625×1225)
│ YOLO11n-ROI          │     检测结构区域（非缺陷）
│ 4 类结构区域          │     切片: 640×640 × 9 片
│ Recall ≥ 99%（硬约束）│     耗时: ~27ms
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ROI 映射 & 去重      │  ← 框映射回原始分辨率 + 边缘扩展
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Stage 2: 缺陷检测    │  ← 仅对 ROI 区域做 1024 切片
│ YOLO11s/m-EMA-SimAM │     切片数: 60~90（降低 50-67%）
│ 18 类缺陷            │     耗时: ~2s
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ WBF 融合 / 全局 NMS  │  ← 跨切片去重 + 坐标映射
└─────────────────────┘

总耗时: 27ms + 500ms + 2000ms + 200ms + 2500ms(解码) ≈ 5.2s
```

### 模型注意力改造

| 位置 | 模块 | 参数量 | 延迟 | 论文 |
| --- | --- | --- | --- | --- |
| P2 检测分支 (四尺度模型) | **SimAM** | **0** | +0.1ms | ICML 2021 |
| P3 检测分支 | **EMA + SimAM** (串联) | ~200 + 0 | +0.5ms | ICASSP 2023 + ICML 2021 |
| P4 检测分支 | **SimAM** | **0** | +0.1ms | ICML 2021 |
| P5 (三尺度) / P5 (四尺度) | 无 / **ECA** | 0 / ~100 | — / +0.1ms | CVPR 2020 |
| Backbone 末端 | C2PSA（保留） | — | — | YOLO11 原生 |

**v2 注意力布局 (2025-06-25 优化)**:
- **P2=SimAM**: 微小缺陷定位主力 — 零参数能量注意力, 识别与邻域"显著不同"的神经元
- **P3=EMA+SimAM**: 最关键的检测尺度 — EMA 提供 X/Y 方向空间编码, SimAM 进一步增强局部异常敏感性
- **P4=SimAM**: 中等目标上下文辅助
- **P5=无/ECA**: 32× 下采样下缺陷不可见, 移除注意力以避免放大背景结构噪声

- **EMA（Efficient Multi-Scale Attention）**：X/Y 双方向池化，保留空间位置信息，增强对小目标（螺栓、开口销 ~8×8 px）的定位能力
- **SimAM（Simple Parameter-Free Attention）**：基于神经科学空间抑制理论，零参数 → 零过拟合风险，对局部异常（如螺栓缺失）天然敏感
- **ECA（Efficient Channel Attention）**：1D 卷积自适应核, 极轻量通道注意力 (< 100 参数), 用于 P5 替代 SimAM

### 双卡异构 Ensemble（地面端）

```
GPU 0: YOLO11m-EMA-SimAM        GPU 1: YOLO11m-P2-SimAM
(ECA 通道选择, 3 尺度)            (P2 四尺度小目标特化)
         │                              │
         └──────────┬───────────────────┘
                    ▼
          ┌─────────────────┐
          │  WBF 融合引擎    │
          │  IoU=0.55        │
          │  双模型≥0.50     │
          │  单模型≥0.75     │
          │  最终≥0.60       │
          └─────────────────┘
```

## 模型选型

### 模型变体

| 模型 | 用途 | 参数量 | GFLOPs | 检测尺度 | 注意力 |
|------|------|--------|--------|---------|--------|
| YOLO11n-ROI | Stage 1 结构区域 | 2.6M | 6.6 | P3/P4/P5 | 无 |
| YOLO11s-EMA-SimAM | 车载端主方案 (三尺度) | ~9.5M | ~22 | P3/P4/P5 | P3:EMA+SimAM, P4:SimAM |
| **YOLO11s-P2-EMA-SimAM** | **车载端 P2 方案** (推荐) | **~9.8M** | **~30** | **P2/P3/P4/P5** | **P2:SimAM, P3:EMA, P4:SimAM** |
| YOLO11m-EMA-SimAM | 地面端 GPU 0 | 20.1M | 68.5 | P3/P4/P5 | P3:EMA, P4:SimAM, P5:ECA |
| YOLO11m-P2-SimAM | 地面端 GPU 1 | ~25M | ~90 | P2/P3/P4/P5 | P2:SimAM, P3:SimAM, P4:SimAM, P5:ECA |

### 选型策略

**车载端（单 RTX 4090，≤ 10s）**: 推荐 yolo11s-P2-EMA-SimAM (四尺度小目标增强), 备选 yolo11s-EMA-SimAM (三尺度)

**地面端（双 RTX 4090，提报率 ≤ 5%）**: GPU 0 YOLO11m-EMA-SimAM + GPU 1 YOLO11m-P2-SimAM → WBF 融合

## 训练流程

### 推荐路线（一键自动化）

**最省心、效果最好的方式**——一行命令跑完核心训练：

```bash
python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0
```

这条命令自动完成：安全检查 → 参数调优 → 生成配置 → 多阶段训练 → 输出部署模型。

核心流程只有 5 个阶段（Stage 1 拆分为 1A+1B）：

```
Stage 1A              Stage 1B             Stage 2              Stage 3             Stage 4
公开缺陷 Head 预热 → 公开缺陷 Backbone → 领域适配         →   主训练         →    短微调
NEU+GC10+SDD2+RSDDs   低 LR 充分训练       → 7类接触网           1280 全解冻          低lr 零增强
1 类 generic_defect    1 类 generic_defect  冻结 backbone         80 epoch             15 epoch, 早停
40 epoch, 1024         60 epoch, 1024       40 epoch              ★ 最大mAP提升         → 部署模型
```

> **核心原则**：每个阶段产出可独立验证，不达标不进下一阶段。

<details>
<summary><b>手动逐步训练 / 自定义阶段 / 更多选项</b></summary>

#### 指定阶段

```bash
# 只跑 Stage 1+2
python scripts/train_pipeline.py --stages 1 2 --model yolo11s-EMA-SimAM --device 0

# 预览不训练
python scripts/train_pipeline.py --stages 1 2 3 4 --dry-run

# 从失败阶段继续
python scripts/train_pipeline.py --stages 3 4 --model yolo11m-EMA-SimAM --device 0

# 手动 batch/workers
python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0 --batch 24 --workers 8
```

#### 自动安全检查

| 检查项 | 不通过时 |
|--------|---------|
| Python ≥ 3.10 | 中止 |
| GPU + VRAM ≥ 6 GiB | 中止 |
| 模型 YAML 存在 | 中止 |
| 阶段配置文件存在 | 自动调用 `multi_source_pretrain_yaml.py` 生成 |
| 数据集路径有效 + 有图片 | 中止 |
| COCO 预训练权重 | 警告（自动下载） |

#### 自动参数调优

| 参数 | 逻辑 |
|------|------|
| `batch` | 基于 VRAM × model family × imgsz 自动估算 |
| `workers` | `min(8, cpu_cores)` |
| `cache` | 首阶段启用 disk cache, 后续阶段复用 |

</details>

---

### 各阶段详解

#### ★ Stage 1A: 公开缺陷 Neck/Head 预热

```text
目的: 让 neck/head 学习工业异常纹理——"哪里有异常"
初始化: COCO yolo11s.pt / yolo11m.pt
数据: NEU-DET + GC10-DET + KolektorSDD2 + RSDDs → 合并为 generic_defect (1类)
      (DeepPCB 已弃用 —— PCB 电路板缺陷与金属件视觉特征差异大)
```

```bash
python scripts/train_pipeline.py --stages 1a --model yolo11m-EMA-SimAM --device 0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 40 | 仅预热 neck/head |
| imgsz | 1024 | 匹配公开数据集分辨率 |
| lr0 | 0.001 | AdamW 初始学习率 |
| mosaic | 0.2 | 轻量 mosaic, 保留结构 |
| erasing | 0 | 不做擦除, 保护小缺陷纹理 |
| freeze | [0..10] | 冻结全部 backbone 层 |

#### ★ Stage 1B: 公开缺陷 Backbone 适应

```text
目的: 低学习率解冻 backbone, 充分适配工业异常特征
初始化: Stage 1A best.pt
```

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 60 | backbone 深层适应 |
| imgsz | 1024 | 一致 |
| lr0 | 0.0003 | 低 LR 保护预训练权重 |
| freeze | [0..5] | 仅冻结浅层 backbone |

**验收**: loss 平稳下降, 无 spike。输出: `weights/stage1a_public_head.pt` → `weights/stage1b_public_backbone.pt`

---

#### ★ Stage 2: 领域适配

```text
目的: 从公开缺陷域切换到真实接触网 7 类
初始化: Stage 1B best.pt (nc=1 → nc=7, Detect cls layers 重建)
数据: subway_crops (1024 原生分辨率 crop, 7 类接触网缺陷)
```

```bash
python scripts/train_pipeline.py --stages 2 --model yolo11m-EMA-SimAM --device 0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 40 | v2: 50→40, 预训练 backbone 收敛快 |
| imgsz | 1024 | 与 Stage 1 一致 |
| lr0 | 0.0008 | v2: cosine schedule |
| mosaic | 0.1 | 几乎关闭, 最小化失真 |
| freeze | [0..7] | 冻结 backbone 前 60% |

**验收**: mAP50 > 0.35, 7 类全部 AP > 0。输出: `weights/stage2_domain_adapt.pt`

---

#### ★ Stage 3: 主训练

```text
目的: 小目标尺度适应——贡献最大 mAP 提升的阶段
初始化: Stage 2 best.pt
数据: subway_crops (1280 原生分辨率, 7 类, 可选 Copy-Paste 增强)
```

```bash
python scripts/train_pipeline.py --stages 3 --model yolo11m-EMA-SimAM --device 0
```

| 参数 | 值 | 与 v1 的区别 |
|------|-----|-------------------|
| epochs | **80** | 120→80: peak 在 epoch ~57 |
| imgsz | **1280** | 缺陷从 8→10px |
| optimizer | **AdamW** | cos_lr schedule |
| lr0 | **0.0005** | 更稳定 |
| mosaic | **0.15** | 轻微失真, 保护小目标 |
| erasing | 0 | 保护 8-10px 小目标 |
| copy_paste | **0.05** | v2: 缺陷感知 Copy-Paste (离线生成) |
| close_mosaic | **35** | 最后 35 epochs 纯净训练 |
| patience | **28** | 防止后半程过拟合 |
| weight_decay | **0.00075** | 轻度增强正则化 |

**验收**: mAP50 比 Stage 2 提升 ≥ +0.05。输出: `weights/stage3_main.pt`

---

#### ★ Stage 4: 短微调

```text
目的: 在真实分布上精调, 早停防退化
初始化: Stage 3 best.pt
```

```bash
python scripts/train_pipeline.py --stages 4 --model yolo11m-EMA-SimAM --device 0
```

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | **15** | v2: 30→15, 短微调早停 |
| lr0 | **1e-5** | 极低学习率 |
| mosaic/erasing | 0 | 零增强, 只看真实图 |
| warmup_epochs | **0** | v2: 关闭 warmup 防 bias LR 尖刺 |
| patience | **5** | v2: 快速早停 |
| freeze | [0..9] | 冻结 backbone 前 70% |
| save_period | 1 | 每 epoch 保存, 选 best_mAP50-95 |

> **不要使用 last.pt**——last 往往已退化。用 `best_mAP50-95.pt`。

**验收**: mAP50 ≥ Stage 3, 验证集 Recall ≥ 0.90。
输出: `weights/stage4_best_finetune.pt` → **部署模型**。

---

<details>
<summary><b>可选阶段（点击展开）</b></summary>

##### Stage P2: TT100K P2 头预热 [⚠ ABLATION ONLY]

<small>**P2 四尺度模型已被实验证明整体性能更差**（mAP50 0.497→0.430, Precision 0.439→0.372），不推荐作为主线模型。仅当运行受控消融实验时才使用。</small>

<small>详见报告 `docs/plans/2026-06-27_方案C_当前模型性能分析_问题诊断与改进建议.md` 第 4 节。</small>

```bash
# [仅消融实验] P2 四尺度模型 + P2 头预热
python scripts/train_pipeline.py --stages p2 1a 1b 2 3 4 --model yolo11m-P2-EMA-SimAM
```

- 数据: TT100K (交通标志), 1 类 `tiny_object`, 80 epoch, 1024
- 输出: `weights/stage_p2_tiny_pretrain.pt`
- **注意**: pipeline 会自动显示 P2 弃用警告

##### Stage 5: 难负样本挖掘 + 阈值校准

<small>仅当 Stage 4 Precision < 90% 时建议运行。用 Stage 4 模型收集误检 → 加入训练集重训 → 每类单独阈值搜索。</small>

```bash
# Step 1: 收集误检 (FP) crop
python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt

# Step 2: 将 data/hard_negatives/ 下各子目录复制到训练数据中

# Step 3: 难负样本训练
python scripts/train_pipeline.py --stages 5 --model yolo11m-EMA-SimAM --device 0

# Step 4: 每类阈值校准 (或使用 --auto-hnm 自动执行)
python scripts/calibrate_thresholds.py --model weights/stage5_calibrated.pt
```

- 20 epoch, 零增强, lr=2e-5, 冻结 backbone 前 70%
- v2 改进: warmup_epochs=0 (防止 bias LR 尖刺)
- 输出: `weights/stage5_calibrated.pt` + `data/calibrated_thresholds/thresholds.json`

##### Stage 0: 数据完整性检查

```bash
train-defect --stages 0 --data data/subway_crops/subway_crops.yaml
```

- 标签可视化 + 框尺寸/长宽比统计 + 20 张过拟合测试
- 不训练, 仅产出报告

</details>

---

### 数据准备

训练前准备好两类数据集。

#### Phase 1A: 自制接触网数据集

```bash
python scripts/prepare_dataset.py           # 一键全流程
```

| 步骤 | 产出 |
|------|------|
| 拷贝原始数据 | `data/Defect_dataset/images/` + `labels/` |
| 标签修复 | 修正后的粗/精标签 |
| 按源图分组划分 | train/val 不共享源图 |
| 生成 data.yaml | `data/Defect_dataset/defect_data.yaml` |
| 校验 | 完整性报告 + 问题清单 |

#### Phase 1B: 多源公开数据集（AutoDL 云端）

```bash
python scripts/multi_source_dataset_builder.py
```

构建后目录：

```
data/multi_datasets/
├── public/gc10_det/       # GC10-DET (3570 张, 10 类金属缺陷)
├── public/neu_det/        # NEU-DET (1800 张, 6 类钢材缺陷)
├── public/kolektor_sdd2/  # ★ NEW: KolektorSDD2 (3335 张, 金属表面划痕/斑点)
├── public/rsdds/          # ★ NEW: RSDDs (195+ 张, 铁轨表面裂纹/压痕)
├── mixed_pretrain/        # 合并: 1 类 generic_defect
│   # DeepPCB 已弃用 — PCB 电路板缺陷特征与金属件差异大
└── subway_crops/          # 接触网原生分辨率 crop (train+val)
```

#### Phase 1C: 生成训练配置

```bash
python scripts/multi_source_pretrain_yaml.py --stages 1 2 3 4
```

产出 `config/train/pretrain/stage{1,2,3,4}_*.yaml`。

---

<details>
<summary><b>阶段编号对照表（旧编号 → 统一编号）</b></summary>

| 统一阶段 | 旧 Phase (代码) | 旧 S 系统 | 旧 C 系统 | 说明 |
|---------|:---:|:---:|:---:|------|
| Stage P2 | Phase 2 | — | — | TT100K P2头预热 (可选) |
| Stage 1A | Phase 3a | — | — | 公开缺陷 Neck/Head 预热 |
| Stage 1B | Phase 3b | — | — | 公开缺陷 Backbone 适应 |
| Stage 2 | Phase 4 | S1 | C1 | 领域适配 (1类→7类) |
| Stage 3 | Phase 5 | S2 | C2 | 主训练 (1280全解冻) |
| Stage 4 | Phase 6 | S3 | C3 | 短微调 (低lr零增强) |
| Stage 5 | Phase 7+8 | S4 | — | 难负样本+阈值校准 (可选) |

</details>

---

<details>
<summary><b>Legacy C1/C2/C3 三阶段（向后兼容，不推荐）</b></summary>

```bash
# 不加 --stages 时默认走 legacy 模式
train-defect --data data/Defect_dataset/defect_data.yaml --coco_pretrain --device 0

# 显式指定 legacy 阶段
train-defect --stages c1 c2 c3 --data data/Defect_dataset/defect_data.yaml --coco_pretrain --device 0
```

| 阶段 | epochs | imgsz | 优化器 | 关键限制 |
|------|--------|-------|--------|---------|
| C1 Head Warmup | 50 | 1024 | SGD | 仅训 head |
| C2 Full Training | 300 | 1280 | AdamW | 增强过强导致小目标退化 |
| C3 Fine-Tune | 100 | 1280 | AdamW | 时间长反而过拟合 |

> Legacy 配置在 `config/train/{warmup,full,finetune}.yaml`，推荐迁移到 `config/train/pretrain/`。

</details>

---

### 常见问题

| 问题 | 方案 |
|------|------|
| GPU OOM | 降低 `--batch` 至 8 或 4 |
| Stage 1 loss 不收敛 | 检查公开数据路径, 确认 `generic_defect` 标签正确 |
| Stage 3 收益不明显 | 确认 imgsz=1280 (非 1024), mosaic≤0.2 |
| Stage 4 mAP 反而下降 | 正常——降低 patience 或减少 epochs |
| 训练中断后恢复 | `python scripts/train_pipeline.py --stages <失败阶段> <后续阶段>` |
| P2 模型训练不收敛 | 先跑 Stage P2 (TT100K 预热); 但 P2 已被证明整体更差, 建议换用三尺度模型 |
| 误报过多 (Precision低) | 运行 `audit_labels.py` 审计标注, 再跑 Stage 5 难负样本 |
| 模型学到位置偏置 | 使用 `--debiasing` 标志运行 `generate_native_crops.py` |

> 训练超参数集中管理在 [`config/train/`](../config/train/)，推理参数在 [`config/model/inference.yaml`](../config/model/inference.yaml)。可直接修改 YAML，无需改代码。

---

### 标注质量标准

#### 松动类标注规范

VHBNL 和 SVHBNL 属于细粒度状态识别——"松动"不是独立物体，而是螺母/螺栓的位置状态异常。当前 Precision 低（VHBNL=0.367）的一个重要原因是标注边界模糊。

建议标注员遵循：

```text
normal（正常，不标框）
  螺母位置正常，与基座/螺纹/垫片相对关系无异常

loose（松动，标 VHBNL / SVHBNL / SVHTNL）
  螺母存在但相对基座有明显偏移
  螺母与垫片间可见异常间隙
  螺栓螺纹露出长度异常

missing（缺失，标 VHBNM / SVHBNM）
  目标位置应有螺母但不可见
  明显空缺，可见螺栓孔或空基座

uncertain（不确定，不进入训练集）
  光照/遮挡导致无法可靠判断
  边界模糊, 两位标注员意见不一致
```

**框尺寸建议**：
- 螺母缺失类：框应包含基座 + 螺栓孔, 提供结构上下文
- 螺母松动类：框应包含螺母 + 垫片 + 相邻基座, 便于判断相对位置
- 销钉缺口类：框应包含销钉头 + 开口销孔 + 腕臂底座局部

**一致性检查**：
- 同一源图中, 相似结构的标注尺度应一致
- 缺失 vs 松动边界需在标注团队内部达成一致

#### 标注质量审计

使用 `audit_labels.py` 工具自动标记可疑标注：

```bash
# 用 Stage 3/4 模型对验证集做审计
python scripts/audit_labels.py --model weights/stage3_main.pt --split val

# 聚焦 SVHBNM（问题最大的类别）
python scripts/audit_labels.py --model weights/stage3_main.pt --classes SVHBNM

# 对训练集做全量审计
python scripts/audit_labels.py --model weights/stage3_main.pt --split train --max-samples 200

# 仅统计不保存
python scripts/audit_labels.py --model weights/stage3_main.pt --dry-run
```

输出：
```
data/label_audit/
├── high_conf_fp/       # 模型高置信度误检 → 可能是漏标或标注错误
├── low_conf_fn/        # 模型漏检/低置信 → 可能是标注错误或极难样本
├── audit_summary.json  # 每类统计 + 优先级排序
└── audit_report.txt    # 人类可读报告 + 审核清单
```

**人工审核优先级**：按 `audit_report.txt` 中的 priority_score 排序，优先审查高分 FP 和 FN。

---

### 数据扩充最佳实践

生成 `subway_crops` 时推荐使用以下命令：

```bash
python scripts/generate_native_crops.py \
    --crop-size 1024 \
    --negatives-per-image 25 \
    --balance \
    --debiasing
```

各标志的作用：

| 标志 | 作用 | 何时使用 |
|------|------|---------|
| `--negatives-per-image 25` | 每张源图生成 25 个负样本 crop | 始终推荐（降低 FP） |
| `--balance` | 少数类（SVHBNL/CBVPM）额外生成 crop | 类别不均衡时 |
| `--debiasing` | 缺陷位置系统性变化（中心30%/偏中心30%/边缘25%/角落15%） | 始终推荐（防止位置过拟合） |

---

### 未来改进方向（尚未实施）

以下建议来自训练结果分析报告，已记录但暂未实施代码。作为下一轮优化的参考方向。

#### 方案 C：Tiny Transformer 二级复核分类器

**背景**（详见 `docs/plans/2026-06-27_方案C_当前模型性能分析_问题诊断与改进建议.md`）:

当前 YOLO 主检测器的瓶颈不是定位能力，而是**细粒度状态判别**——YOLO 能发现"可能有异常的区域"，但对"正常 vs 松动 vs 缺失"的区分不够稳定，尤其是 SVHBNM（1,197 个样本→mAP50 仅 0.251）和 VHBNL（Precision 仅 0.367）。

**设计思路**：

```
YOLO 候选框 → 难类+低置信候选 → Tiny Transformer 复核 → 分数融合 → 最终输出
```

**推荐方案**：

| 要素 | 推荐 |
|------|------|
| 模型 | MobileViT-S（车载端）/ Swin-Tiny（地面端） |
| 输入 | 224×224, 多尺度 crop (1.2x/2.0x/3.0x) |
| 类别 | 4 类状态: `normal` / `missing_nut` / `loose_nut` / `pin_gap` |
| 融合 | `s_final = 0.6 × s_yolo + 0.4 × s_cls` |
| 推理成本 | 50-100 个候选 × 1-2ms ≤ 200ms |

**实施要点**：

1. 从 YOLO 推理结果收集 TP/FP/FN crop → 构建分类器数据集
2. 按源图分组划分 train/val/test（防止同源泄漏）
3. 增强限于亮度/轻微模糊/轻微噪声（禁止大角度旋转/翻转）
4. 第一版只做过滤（不倒写类别），第二版再尝试类别修正
5. `normal : defect = 1.5 : 1` 到 `2 : 1`，强过采样 YOLO 误报区域

**预期收益**：Precision 0.44 → ≥0.60（第一阶段），SVHBNM mAP50 0.25 → ≥0.35

**需要新建的文件**：
- `scripts/build_classifier_dataset.py` — 从 YOLO TP/FP/FN 收集 crop
- `subway_defect/classification/secondary_classifier.py` — 训练/推理封装
- `config/classification/` — 分类器训练配置
- `scripts/train_secondary_classifier.py` — 训练入口

#### 其他未实施建议

| 建议 | 来源 | 状态 |
|------|------|------|
| SVHBNM 标注审计 | 报告 §5.2 | `audit_labels.py` 已提供工具, 待人工执行 |
| 位置去偏置 crop | 报告 §3.3.4 | `--debiasing` 标志已实现 |
| P2 模型正式弃用 | 报告 §4.3-4.4 | pipeline 已添加弃用警告 |
| 松动类标注规范 | 报告 §5.2 | 已写入本文档 §标注质量标准 |

---

## 推理部署

### 推理服务启动

```bash
# 车载端（单模型）
subway-server --port 8001 \
              --model output/<时间戳>/c3_finetune/weights/best.pt \
              --mode onboard

# 地面端（双 GPU Ensemble + WBF 融合）
subway-server --port 8001 \
              --model output/<时间戳>/c3_finetune/weights/best.pt \
              --model_b output/<时间戳>_p2/c3_finetune/weights/best.pt \
              --mode ground
```

### API 端点

> **接口规范遵循**: 本服务严格遵循与前后端（Spring Boot 后端 + Vue3 前端）约定的统一接口契约。
> 字段采用 **camelCase** 命名（与 Java 后端对齐），通过 Pydantic `alias` 同时兼容 snake_case。
> 完整规范见 `docs/开发接口规范/` 目录。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/dl/health` | 健康检查 + 已加载模型 + GPU 状态 |
| `POST` | `/api/dl/infer` | 单张图像缺陷检测（核心接口） |
| `POST` | `/api/dl/infer/batch` | 批量推理（地面端高吞吐） |
| `POST` | `/api/dl/model/load` | 加载/切换模型 |

### 健康检查

```bash
GET /api/dl/health
```

响应:
```json
{
  "status": "healthy",
  "loadedModels": [
    { "modelType": "onboard", "version": "yolov11s-v1.2", "loadedAt": "2026-06-26T10:00:00" }
  ],
  "gpuAvailable": true,
  "gpuMemoryUsedMB": 3200,
  "gpuMemoryTotalMB": 24564
}
```

### 单张推理（核心接口）

```bash
POST /api/dl/infer
Content-Type: application/json
```

请求（字段使用 camelCase）:
```json
{
  "imagePath": "/data/images/20260624_001.jpg",
  "modelType": "onboard",
  "confidenceThreshold": 0.40,
  "outputCoordType": "normalized",
  "extraParams": {
    "sliceSize": 1024,
    "sliceOverlap": 0.15,
    "highResRegions": []
  }
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `imagePath` | string | 是 | — | 图像文件路径（绝对路径） |
| `modelType` | string | 否 | `"onboard"` | `"onboard"`（车载端）或 `"ground"`（地面端） |
| `confidenceThreshold` | float | 否 | 0.40 | 缺陷检测置信度阈值 |
| `outputCoordType` | string | 否 | `"normalized"` | 坐标类型：`"normalized"`（归一化 0~1）或 `"pixel"`（像素） |
| `extraParams.sliceSize` | int | 否 | 1024 | 切片尺寸（像素） |
| `extraParams.sliceOverlap` | float | 否 | 0.15 | 切片重叠率 |
| `extraParams.highResRegions` | array | 否 | `[]` | 关键区域坐标（使用更高重叠率） |

响应:
```json
{
  "success": true,
  "imagePath": "/data/images/20260624_001.jpg",
  "processingTimeMs": 5234.5,
  "totalSlices": 72,
  "num_roi_regions": 12,
  "defects": [
    {
      "defectType": "VHBNM",
      "defectName": "垂直悬吊安装底座螺母缺失",
      "confidence": 0.923,
      "box": { "x": 0.3421, "y": 0.5612, "w": 0.0123, "h": 0.0089 },
      "coordType": "normalized",
      "sourceSlice": { "row": 3, "col": 5 }
    }
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `defects[].defectType` | string | 缺陷类型编码（16 类编码之一，见下方字典） |
| `defects[].defectName` | string | 缺陷中文名称 |
| `defects[].confidence` | float | 置信度 0~1 |
| `defects[].box` | object | 缺陷框（归一化中心点 x/y + 宽高 w/h） |
| `defects[].coordType` | string | 坐标类型，固定为 `"normalized"` |
| `defects[].sourceSlice` | object | 来源切片坐标（用于调试追溯） |

### 标准错误码

所有 DL 接口使用统一的错误响应格式：

```json
{
  "success": false,
  "errorCode": "DL_GPU_OOM",
  "message": "GPU 显存不足。...",
  "suggestion": "Try reducing batch size or slice size"
}
```

| 错误码 | HTTP 状态码 | 说明 | 后端处理策略 |
|--------|-----------|------|-------------|
| `DL_MODEL_NOT_LOADED` | 503 | 模型未加载 | 自动调用 `/model/load`，最多重试 2 次 |
| `DL_GPU_OOM` | 507 | GPU 显存不足 | 降低 batch/slice 尺寸后重试 |
| `DL_IMAGE_UNREADABLE` | 400 | 图像损坏/无法读取 | 标记 `analysis_status='skipped'`，继续下一张 |
| `DL_INFERENCE_TIMEOUT` | 504 | 推理超时（单张 > 30s） | 记录错误，跳过此张 |
| `DL_INTERNAL_ERROR` | 500 | 推理引擎内部异常 | 记录日志，重试 1 次 |

### 坐标系约定（三方统一）

缺陷框统一采用**归一化中心点 + 宽高**（与 YOLO 一致），取值 0~1：

```json
"box": { "x": 0.3421, "y": 0.5612, "w": 0.0123, "h": 0.0089 }
"coordType": "normalized"
```

前端 fabric.js 渲染时换算为像素坐标：`pixel_x = x * imageWidth`, `pixel_y = y * imageHeight`。

### TensorRT 加速导出

```bash
# FP16 导出（推荐，车载端必备）
export-tensorrt --model output/<时间戳>/c3_finetune/weights/best.pt --fp16

# INT8 导出（需校准数据集 200-500 张）
export-tensorrt --model best.pt --int8 \
                --calibration_data datasets/calibration/
```

## 合成数据生成

```bash
synthesize-defects --images datasets/images/train/ \
                   --labels datasets/labels/train/ \
                   --output datasets/synthetic/ \
                   --target_class 0 \
                   --limit 200
```

## 缺陷类别编码（16 类）

> **权威来源**: 与 `docs/接触网缺陷类型详解.docx` 和 `subway_defect/deployment/defect_dict.json` 保持一致。
> 当代码与文档出现分歧时，以此表为单一事实来源。当前训练数据覆盖前 7 类（标记 ★）。

### 刚性接触网（12 类）

| 编码 | 中文名称 | 严重等级 | 训练状态 |
| --- | --- | --- | :---: |
| `VHBNM` | 垂直悬吊安装底座螺母缺失 | serious | ★ |
| `VHBNL` | 垂直悬吊安装底座螺母松动 | serious | ★ |
| `SVHBNM` | 单支垂直悬吊槽钢底座螺母缺失 | serious | ★ |
| `SVHBNL` | 单支垂直悬吊槽钢底座螺母松动 | serious | ★ |
| `SVHTNL` | 单支垂直悬吊槽钢上方螺母松动 | normal | ★ |
| `RHTBNM` | 刚性悬挂吊柱顶板底面螺母缺失 | serious | |
| `RHTBNL` | 刚性悬挂吊柱顶板底面螺母松动 | serious | |
| `GWCSBNM` | 地线线夹托板安装底座螺母缺失 | serious | |
| `GWCSBNL` | 地线线夹托板安装底座螺母松动 | serious | |
| `GWCNM` | 地线线夹螺母缺失 | serious | |
| `GWCNL` | 地线线夹螺母松动 | serious | |
| `BSBM` | 汇流排中间接头螺栓缺失 | **critical** | |
| `INSD` | 绝缘子破损 | **critical** | |

### 柔性接触网（3 类）

| 编码 | 中文名称 | 严重等级 | 训练状态 |
| --- | --- | --- | :---: |
| `CBHPM` | 腕臂底座横向销钉缺口 | serious | ★ |
| `CBVPM` | 腕臂底座垂直销钉缺口 | serious | ★ |
| `DRPS` | 吊弦不受力 | serious | |

### 严重等级定义

| 等级 | 英文 | 说明 |
|------|------|------|
| minor | minor | 轻微异常，不影响运行 |
| normal | normal | 一般缺陷，需关注 |
| serious | serious | 严重缺陷，影响安全 |
| critical | critical | 危急缺陷，立即停运 |

### 前后端集成要点

1. **缺陷编码**: 后端 `t_defect.defect_type` 字段和前端展示均使用上述编码。`defect_dict.json` 为机器可读的完整映射文件
2. **坐标转换**: DL 默认输出归一化坐标 → 后端原样存储 → 前端渲染时乘以图像宽高
3. **模型版本**: 格式 `{base_model}-{scale}-v{major}.{minor}`，如 `yolo11s-v1.2`
4. **车载/地面双端**: 对前端暴露完全相同的 API 契约，前端仅切换 `baseURL`

## 持续训练管道

```
数据采集(工程车) → 主动学习筛选 → 人工标注 → 数据集版本管理 → 自动训练触发 → 模型注册 & 部署
```

**触发条件**:
- 新增标注数据 ≥ 200 张
- 新增缺陷类别
- 线上 Precision/Recall 低于验收线
- 定期重训练（每季度）

**部署准入条件**:
1. Recall ≥ 当前线上版本
2. Precision ≥ 当前线上版本
3. 无类别退化: 任何类 Recall 下降 ≤ 5%
4. 推理耗时不增加

## 设计文档

- [完整规格说明书](../SPECIFICATION.md) — 项目需求、架构、接口、数据、测试等完整规范
- [AI 算法设计文档](地铁接触网缺陷检测AI算法设计文档.md) — 核心算法完整设计
- [系统开发方案](开发方案(5.30)/0-总览.md) — 前后端 + DL 系统架构
- [实现计划 1: 核心模型](plans/2026-06-23-plan-1-core-model-architecture.md)
- [实现计划 2: 训练管道](plans/2026-06-23-plan-2-training-pipeline.md)
- [实现计划 3: 推理引擎](plans/2026-06-23-plan-3-inference-engine.md)
- [📊 分析: 模型特征学习效率诊断](plans/2026-06-25-analysis-feature-learning-efficiency.md) — C2 零收益根因分析
- [🔧 方案: 结构改进与训练流程优化](plans/2026-06-25-Structure-improvement-plam.md) — P2-Lite 架构 + 五阶段训练
- [📦 方案: 多源数据集训练建议](plans/2026-06-25-Multi-source-datasets-training-recomendation.md) — 公开数据集整合策略
- [📊 分析: 当前模型性能诊断与改进建议](plans/2026-06-27_方案C_当前模型性能分析_问题诊断与改进建议.md) — 三尺度 vs P2 对照 + 方案 C

## 技术参考

| 模块 | 参考论文 | 出处 |
| --- | --- | --- |
| YOLO11 | Ultralytics YOLO11 | https://docs.ultralytics.com/models/yolo11 |
| EMA | Efficient Multi-Scale Attention | ICASSP 2023 |
| SimAM | A Simple, Parameter-Free Attention Module | ICML 2021 |
| CopyPaste | Simple Copy-Paste is a Strong Data Augmentation Method | ICCV 2021 |
| WBF | Weighted Boxes Fusion: Ensembling boxes | arXiv 1910.13302 |
| SAHI | Slicing Aided Hyper Inference | arXiv 2206 |

## License

AGPL-3.0 License — 与 YOLO 框架保持一致。
