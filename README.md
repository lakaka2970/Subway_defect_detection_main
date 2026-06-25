# 地铁接触网缺陷检测系统

基于 YOLO11 + EMA/SimAM 注意力的两阶段 AI 缺陷检测系统，用于福州地铁接触网超高清图像（1.27 亿像素）的智能化分析。

> 完整规格说明见 [SPECIFICATION.md](SPECIFICATION.md)

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
│   ├── modules/                      # EMA、SimAM 注意力模块
│   ├── models/                       # 模型 YAML 配置文件（3 个变体）
│   ├── pipeline/                     # 推理管道（切片器、两阶段、WBF 融合）
│   ├── train/                        # 训练模块（超参数预设、CLI 脚本）
│   ├── augmentations/                # 数据增强（场景模拟、CopyPaste）
│   ├── deployment/                   # 部署（TensorRT 导出、FastAPI 服务）
│   ├── synthetic/                    # 合成数据生成（Inpainting）
│   └── docs/                         # 设计文档 + 前后端接口规范
│       ├── 地铁接触网缺陷检测AI算法设计文档.md
│       ├── plans/                    # 实现计划
│       │   ├── 2026-06-25-Multi-source-datasets-training-recomendation.md
│       │   ├── 2026-06-25-Structure-improvement-plam.md
│       │   └── 2026-06-25-analysis-feature-learning-efficiency.md
│       └── 开发方案(5.30)/            # 系统开发方案
├── subway_yolo/                      # Vendored YOLO 框架（已精简）
│   ├── engine/                       # Model、Trainer、Predictor、Validator、Exporter
│   ├── nn/                           # tasks、modules、Extramodule（EMA/SimAM 桥接）
│   ├── models/yolo/                  # 仅 detect + classify
│   ├── data/                         # 数据加载、增强
│   ├── cfg/                          # 配置 + YOLO11 模型定义
│   ├── utils/                        # 核心工具函数
│   └── optim/                        # 优化器
├── tests/                            # 测试套件
│   ├── test_attention_modules.py     # EMA/SimAM 单元 + 模型集成
│   ├── test_augmentations.py         # 增强管道 + 训练配置
│   └── test_pipeline.py              # 切片器 + WBF 融合 + 部署
├── scripts/
│   └── setup_autodl.sh               # AutoDL 云平台环境配置
├── tool/                             # 数据集工具脚本
│   ├── prepare_dataset.py            # 一键自制数据集准备
│   ├── split_dataset.py              # 按源图分组 train/val 划分
│   ├── validate_dataset.py           # 数据集完整性校验
│   ├── multi_source_dataset_builder.py   # 多源公开数据集构建器 (AutoDL)
│   ├── multi_source_pretrain_yaml.py     # 多阶段训练配置生成器
│   ├── generate_scene_augmentations.py   # 场景增强（隧道/日照/模糊）
│   └── generate_synthetic_defects.py     # Inpainting 合成缺陷
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
# 验证三个模型 YAML 均可正常构建
python -c "
from subway_yolo import YOLO
for cfg in ['subway_defect/models/yolo11s-EMA-SimAM.yaml',
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
| P3 检测分支 | **EMA** | ~200 | +0.4ms | ICASSP 2023 |
| P4/P5 检测分支 | **SimAM** | **0** | +0.1ms | ICML 2021 |
| Backbone 末端 | C2PSA（保留） | — | — | YOLO11 原生 |

- **EMA（Efficient Multi-Scale Attention）**：X/Y 双方向池化，保留空间位置信息，增强对小目标（螺栓、开口销 ~8×8 px）的定位能力
- **SimAM（Simple Parameter-Free Attention）**：基于神经科学空间抑制理论，零参数 → 零过拟合风险，对局部异常（如螺栓缺失）天然敏感

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
| YOLO11s-EMA-SimAM | 车载端主方案 | 9.5M | 21.7 | P3/P4/P5 | EMA + SimAM |
| YOLO11m-EMA-SimAM | 地面端 GPU 0 | 20.1M | 68.5 | P3/P4/P5 | EMA + SimAM + ECA |
| YOLO11m-P2-SimAM | 地面端 GPU 1 | ~25M | ~90 | P2/P3/P4/P5 | SimAM ×4 |

### 选型策略

**车载端（单 RTX 4090，≤ 10s）**: 主方案 YOLO11s-EMA-SimAM (FP16)，备选 YOLO11m-EMA-SimAM (INT8)

**地面端（双 RTX 4090，提报率 ≤ 5%）**: GPU 0 YOLO11m-EMA-SimAM + GPU 1 YOLO11m-P2-SimAM → WBF 融合

## 训练流程

### 训练总览

本项目采用 **"COCO 通用预训练 → 公开工业缺陷中间域预训练 → 自制接触网数据领域适配 → 阈值校准"** 的多源分层训练策略：

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              多源分层训练管线 (Multi-Source Training Pipeline)                         │
├────────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                    │
│  COCO 2017              公开工业缺陷数据集                   自制接触网数据                           │
│  (118K, 80 类)          DeepPCB + NEU-DET + GC10-DET        (~1880 crop, 7 类)                     │
│       │                       │                                   │                                │
│       ▼                       ▼                                   ▼                                │
│  ┌──────────┐   Phase 2   ┌──────────────┐   Phase 4-5   ┌──────────────┐   Phase 6-8   ┌────────┐ │
│  │ 基础权重  │ ─────────→  │ 公开缺陷预训练 │ ────────────→ │ 自制数据训练   │ ────────────→ │ 部署模型 │ │
│  │ yolo11s  │   (可选:     │ generic_defect│   neck+head   │ 7 类接触网缺陷 │   短微调+     │        │ │
│  │ .pt      │   Phase 3    │  120 epochs   │    适配       │ 120 epochs    │   Hard Neg    │        │ │
│  └──────────┘   TT100K    └──────────────┘               └──────────────┘               └────────┘ │
│                   P2 头                                                                           │
│                   预热                                                                             │
│                                                                                                    │
│  核心策略:                                                                                          │
│  • 所有公开缺陷数据集统一合并为 generic_defect 单类 — 让模型专注学习"异常区域在哪里"                    │
│  • 自制数据采用 1024/1280 原生分辨率 ROI crop，替代整图 resize                                        │
│  • 训练 → 短微调 → Hard Negative Mining → 每类阈值校准 五阶段闭环                                    │
│  • 每阶段有可独立验证的验收标准，问题可精确追溯                                                       │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**分层训练 vs 直接训练的优势**：

| 对比维度 | 直接训练 (原 C1→C2→C3) | 多源分层训练 (推荐) |
|---------|----------------------|-------------------|
| 数据规模 | 仅自制 ~1880 张 | 公开数万张 + 自制 ~1880 张 |
| Backbone 特征 | COCO 通用特征 | COCO → 工业缺陷纹理 → 接触网结构 |
| 小目标处理 | P3 特征图 ~1×1 px (不可见) | P2 头 + 原生分辨率 crop (~8-10 px 可见) |
| C2 阶段收益 | +0.003~0.018 mAP50 (几乎零收益) | 预期 +0.10~0.15 mAP50 |
| 类别语义冲突 | 无 (仅自制 7 类) | 无 (generic_defect 避免冲突) |

> **核心原则**: 每一阶段产出可独立验证，问题可精确追溯。如果某一阶段指标不达标，立即排查该阶段的问题，不进入下一阶段。

---

### Phase 1: 数据集准备

训练前必须完成**两类数据集**的构建：自制接触网数据集（Phase 1A）和多源公开工业缺陷数据集（Phase 1B）。

---

#### Phase 1A: 自制数据集准备

项目提供了 `tool/` 目录下的一键准备脚本：

```bash
# 一键执行：classes.txt 修复 → YAML 生成 → train/val 划分 → 场景增强 → 合成缺陷
python tool/prepare_dataset.py

# 校验数据集完整性
python tool/validate_dataset.py
```

**各步骤说明**：

| 步骤 | 脚本 | 产出 |
|------|------|------|
| ① 修复 classes.txt | `fix_classes_txt.py` | 7 类，无空行 |
| ② 生成 YAML 配置 | `create_defect_data_yaml.py` | `data/Defect_dataset/defect_data.yaml`（channels: 3, 兼容 COCO 预训练权重） |
| ③ train/val 划分 | `split_dataset.py` | ~400 train + ~100 val，按源图分组防泄露 |
| ④ 场景增强 | `generate_scene_augmentations.py` | ~1200 张增强变体（隧道/日照/模糊/天气） |
| ⑤ 合成缺陷 | `generate_synthetic_defects.py` | ~280 张 inpainting 合成缺失样本 |

**预期产出**：

| 指标 | 值 |
|------|-----|
| 训练集 | ~1880 张（399 原始 + ~1200 增强 + ~280 合成） |
| 验证集 | ~101 张（原始，无增强） |
| 类别数 | 7 类（VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM） |
| 图像尺寸 | 5120 × 5120（训练时 resize 至 1024） |

> **注意**：当前数据集仅覆盖 7 类缺陷，模型 YAML 中 `nc: 18` 为完整缺陷分类体系的占位值。训练时 dataset YAML 会自动将模型 `nc` 覆盖为实际类别数。新增标注数据后，只需更新 dataset YAML 的 `names` 列表即可扩展类别。

**✅ 验证通过标准**：运行 `python tool/validate_dataset.py` 输出 `[PASS] VALIDATION PASSED`。

---

#### Phase 1B: 多源公开数据集构建（AutoDL 云端）

本方案推荐使用 **DeepPCB + GC10-DET + NEU-DET** 三个公开工业缺陷数据集作为中间域预训练。项目提供 `multi_source_dataset_builder.py` 一键完成扫描→下载→格式转换→合并全流程。

**数据集价值评估**：

| 数据集 | 规模 | 缺陷类型 | 对项目的价值 | 优先级 |
|--------|------|---------|-------------|--------|
| **DeepPCB** | 1,500 对图 | 开路、短路、缺口、毛刺、针孔、多余铜 | ⭐最高 — PCB 规则结构缺失 ≈ 螺栓/开口销缺失 | 1 |
| **GC10-DET** | 3,570 张 | 冲孔、焊缝、月牙痕、水渍、油渍、夹杂、划痕等 10 类 | 高 — 金属表面缺陷纹理 | 1 |
| **NEU-DET** | 1,800 张 | 轧入氧化皮、斑块、裂纹、点蚀、夹杂、划痕 6 类 | 高 — 经典工业缺陷基准 | 1 |
| **TT100K** | 100K 张 | 交通标志 (221 类) | 中高 — 小目标定位训练 (仅 P2 头) | 2 (可选) |
| **Insulator Defect** | ~917 张 | 绝缘子破损/闪络 | 中高 — 语义域最近 | 2 (可选) |
| **MVTec AD / VisA** | 5K+/10K+ 张 | 异常检测 (mask) | 中 — 需 mask→bbox 转换 | 3 (可选) |

**关键设计决策 — 方案 B：全合并为 `generic_defect`**：

所有公开数据集的缺陷类别统一映射为单一类别 `generic_defect`（class_id=0），好处：
1. 避免公开类别与接触网 7 类产生语义冲突
2. 让模型专注学习"异常区域在哪里"而非"是什么类型的异常"
3. Box 回归分支和 Neck 特征可完整迁移到最终模型
4. 对小数据集场景更稳定，不易过拟合

```bash
# ===================== AutoDL 云端操作步骤 =====================

# Step 1 — 扫描 /root/autodl-pub 中已有的公开数据集
python tool/multi_source_dataset_builder.py --scan-only

# Step 2 — 完整构建（扫描 + 下载缺失 + 统一转 YOLO + 合并）
#   DeepPCB:   git clone https://github.com/tangsanli5201/DeepPCB.git
#   GC10-DET:  kaggle datasets download -d alex000kim/gc10det
#   NEU-DET:   kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database
python tool/multi_source_dataset_builder.py

# 如果 Kaggle CLI 未配置，仅使用 autodl-pub 已有数据
python tool/multi_source_dataset_builder.py --no-download

# 只构建特定数据集
python tool/multi_source_dataset_builder.py --datasets deeppcb neu_det gc10_det

# 启用可选数据集（TT100K、绝缘子、MVTec AD、VisA）
python tool/multi_source_dataset_builder.py --enable tt100k insulator_defect

# Dry-run：预览操作计划不执行
python tool/multi_source_dataset_builder.py --dry-run
```

**各数据集获取通道（已验证 2025-06）**：

| 数据集 | 获取方式 | 命令/URL |
|--------|---------|---------|
| COCO | AutoDL 公有数据 | `/root/autodl-pub/coco2017/` (必有) |
| DeepPCB | GitHub | `git clone https://github.com/tangsanli5201/DeepPCB.git` |
| GC10-DET | Kaggle | `kaggle datasets download -d alex000kim/gc10det` |
| NEU-DET | Kaggle | `kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database` |
| TT100K | 清华官网 / Ultralytics | `http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip` |
| Insulator Defect | Roboflow (需 API Key) | `pourya-shojaei/insatance-segmentation-insulator` |
| MVTec AD | TIB LDM | `https://service.tib.eu/ldmservice/dataset/mvtec-anomaly-detection--ad--dataset` |
| VisA | AWS S3 公开桶 | `aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar ./` |

**产出目录结构**：

```
data/multi_datasets/
├── public/                          # 各数据集独立目录（YOLO 格式）
│   ├── deeppcb/images/{train,val}/ labels/{train,val}/
│   ├── gc10_det/...
│   └── neu_det/...
├── mixed_pretrain/                  # 汇总 symlink + data.yaml
│   ├── images/{train,val}/          # → 所有 public 图像的 symlink
│   ├── labels/{train,val}/          # → 所有 public 标签的 symlink
│   └── data.yaml                    # nc:1, names:["generic_defect"]
└── _downloads/                      # 原始下载缓存
```

**✅ 验证通过标准**：

```bash
# 确认 mixed_pretrain 目录完整
ls data/multi_datasets/mixed_pretrain/data.yaml
ls data/multi_datasets/mixed_pretrain/images/train/ | wc -l   # 应 > 5000
```

---

#### Phase 1C: 生成多阶段训练配置

数据集构建完成后，使用 `multi_source_pretrain_yaml.py` 为各训练阶段生成对应的 YAML 配置文件：

```bash
# 生成所有可用的训练阶段配置
python tool/multi_source_pretrain_yaml.py

# 仅生成指定阶段
python tool/multi_source_pretrain_yaml.py --phases 2 3 4

# 自定义数据集根目录
python tool/multi_source_pretrain_yaml.py --root data/multi_datasets

# Dry-run：预览配置内容不写入文件
python tool/multi_source_pretrain_yaml.py --dry-run
```

**产出文件（输出至 `config/train/pretrain/`）**：

| 文件 | 对应阶段 | 用途 | 类别数 |
|------|---------|------|--------|
| `phase2_tiny_pretrain.yaml` | Phase 2 (可选) | TT100K P2 小目标头预热 | nc:1 `tiny_object` |
| `phase3_public_defect.yaml` | Phase 3 | DeepPCB+NEU+GC10 工业缺陷预训练 | nc:1 `generic_defect` |
| `phase4_neck_head_adapt.yaml` | Phase 4 | 公开缺陷域 → 接触网 7 类适配 | nc:7 自制类别 |
| `phase5_main_training.yaml` | Phase 5 | 1280 原生 crop 主训练 | nc:7 自制类别 |
| `phase6_short_finetune.yaml` | Phase 6 | 弱增强短微调 + 早停 | nc:7 自制类别 |

> 每个 YAML 文件均包含对应阶段的推荐超参数、增强策略和冻结方案，可直接用于训练。

---

### Phase 2: COCO 预训练权重

YOLO11 在 COCO 2017 数据集（118K 图像，80 类通用目标）上预训练，获得了强大的通用视觉特征提取能力。我们将其作为特征基础，在自制地铁数据集上进行迁移学习。

**权重自动下载**（推荐）：
训练命令添加 `--coco_pretrain` 标志，框架会自动匹配并下载对应规格的 COCO 权重（如 `yolo11s.pt`）。

**手动下载**（离线环境备选）：

```bash
# 车载端模型 (9.5M 参数)
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt \
    -O yolo_weights/yolo11s.pt

# 地面端模型 (20.1M 参数)
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt \
    -O yolo_weights/yolo11m.pt
```

> 所有权重文件统一存放于 `yolo_weights/` 目录。框架会自动在该目录查找预训练权重。

**✅ 验证通过标准**：
```bash
python -c "from subway_yolo import YOLO; m = YOLO('yolo11s.pt'); print('OK:', sum(p.numel() for p in m.model.parameters()), 'params')"
# 输出: OK: 9,484,000+ params
```

---

### Phase 2B: 公开缺陷数据中间域预训练（多源方案新增）

> **本阶段为多源分层训练方案的核心新增步骤。**如果 AutoDL 环境尚未构建多源数据集，请先完成 [Phase 1B](#phase-1b-多源公开数据集构建autodl-云端)。

**训练原理**：在 COCO 通用预训练基础上，使用 DeepPCB + NEU-DET + GC10-DET 三个公开工业缺陷数据集进行中间域预训练。所有缺陷类别统一映射为 `generic_defect` 单类，让 backbone/neck 学习工业异常纹理表征，避免公开类别与接触网 7 类产生语义冲突。

```
COCO yolo11s.pt → 公开缺陷预训练 (generic_defect, 120 epochs) → public_defect_pretrain.pt
     (可选: 先跑 Phase 2A TT100K P2 头预热 80 epochs)
```

**训练命令**：

```bash
# 使用 Phase 1C 生成的配置直接训练
train-defect \
    --data config/train/pretrain/phase3_public_defect.yaml \
    --model subway_defect/models/yolo11s-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0 \
    --name public_defect_pretrain

# 如果有 P2 模型且已完成 TT100K 预热
train-defect \
    --data config/train/pretrain/phase3_public_defect.yaml \
    --model subway_defect/models/yolo11m-P2-SimAM.yaml \
    --pretrained weights/p2_tiny_pretrain.pt \
    --device 0 \
    --name public_defect_pretrain
```

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `epochs` | 120 | 充分的中间域预训练 |
| `imgsz` | 1024 | 与后续自制数据训练一致 |
| `optimizer` | AdamW | 自适应学习率，补偿 COCO→defect 域偏移 |
| `lr0` → `lrf` | 0.001 → 0.00002 | Cosine 衰减 |
| `mosaic` | 0.2 | 降低 Mosaic，保护小缺陷不被破坏 |
| `mixup` / `copy_paste` / `erasing` | 0 / 0 / 0 | 关闭混合增强，训练干净缺陷特征 |
| `close_mosaic` | 30 | 最后 30 epoch 在真实图像上精调 |
| `freeze` | `[0..10]` (前 10 epoch) | 先冻结 backbone，后解冻全模型 |

**训练策略**：
1. **前 10 epoch**：冻结 backbone 前半部分，仅训练 neck + detect head
2. **第 11 epoch 后**：解冻全部层，backbone 使用较低学习率（分层 LR 或全局低 LR）
3. **验收重点**：不追求 mAP50 极致数值，重点观察训练是否稳定、P2/P3 分支是否有效、loss 是否正常下降

**✅ 验证通过标准**：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| best.pt 已保存 | 文件存在 | `ls output/<时间戳>/public_defect_pretrain/weights/best.pt` |
| 训练 loss | 稳定下降，无震荡 | 检查 `results.csv` 中的 `train/box_loss` 趋势 |
| mAP50 | 不必追求极致 | 公开数据 mAP50 仅供参考，重点看迁移效果 |
| 输出权重 | `weights/public_defect_pretrain.pt` | 后续 Phase 4/5 的初始化权重 |

---

### Phase 2A (可选): TT100K P2 小目标头预热

> 仅当使用 **P2 四尺度模型**（如 `yolo11s-P2-EMA-SimAM-Lite`）时才需要此阶段。

**训练原理**：新增的 P2 检测分支（stride=4）没有 COCO 预训练权重。使用 TT100K 交通标志数据集（大量小目标）对 P2 头进行短期预热，让 P2 分支先学会小目标定位。

```bash
# 使用 Ultralytics 内置 TT100K 自动下载
train-defect \
    --data TT100K.yaml \
    --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0 \
    --epochs 80 \
    --imgsz 1024 \
    --name p2_tiny_pretrain

# 或使用 Phase 1C 生成的配置
train-defect \
    --data config/train/pretrain/phase2_tiny_pretrain.yaml \
    --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0 \
    --name p2_tiny_pretrain
```

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `epochs` | 50–80 | 短期预热即可 |
| `imgsz` | 1024 | 匹配后续训练分辨率 |
| `optimizer` | AdamW | 与新架构磨合 |
| `mosaic` | 0.2 | 适度 Mosaic |
| `close_mosaic` | 20 | 最后 20 epoch 纯净训练 |

**✅ 验证通过标准**：`best.pt` 保存且验证 loss 正常下降。输出权重用于 Phase 2B 的 `--pretrained` 参数。

---

### (保留) Phase 3: 自制数据 Neck/Head 领域适配

**训练原理**：冻结 backbone 全部 11 层（`model.0 ~ model.10`），仅训练检测头（YOLO Head + EMA/SimAM 注意力）。COCO 学到的通用特征（边缘、纹理、形状）保持不变，检测头快速适应地铁接触网的特定目标尺寸和分布。

```bash
# 仅运行 C1（跳过 C2、C3），验证预热效果
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --model subway_defect/models/yolo11s-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0 \
    --skip_full \
    --skip_finetune \
    --name defect_detector
```

> 输出目录：`output/<时间戳>/c1_warmup/`

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `freeze` | `[0..10]` | 冻结所有 backbone 层 |
| `lr0` | 0.001 | 恒定学习率，不过早降低 |
| `mosaic` | 0.5 | 适度 Mosaic，不过度扭曲 |
| `mixup` / `copy_paste` | 0 / 0 | 此阶段关闭混合增强 |

> 训练超参数定义在 [`config/train/warmup.yaml`](config/train/warmup.yaml)，可直接修改。

**✅ 验证检查点**：

| 指标 | 目标值 | 如何查看 |
|------|--------|---------|
| best.pt 已保存 | 文件存在 | `ls output/<时间戳>/c1_warmup/weights/best.pt` |
| mAP50 | **> 0.30** | 训练日志或 `results.csv` 中的 `metrics/mAP50(B)` |
| Box Loss | **< 1.5** | `results.csv` 中的 `train/box_loss` |
| 训练曲线 | 持续下降，无发散 | `output/<时间戳>/c1_warmup/results.png` |

> **❌ 不通过**：Loss 持续不降 → 降低 LR 至 0.0005；mAP 始终 < 0.10 → 检查标注与图像是否匹配。

---

### Phase 4: Stage C2 — 全量训练（200 epochs）

**训练原理**：解冻 backbone，启用增强（Mosaic 0.5, CopyPaste 0.3），SGD + Cosine LR 从 0.001 衰减至 0.0001。最后 15 个 epoch 关闭 Mosaic 让模型适应真实分布。

```bash
# 从 C1 最佳权重开始，仅运行 C2（跳过 C1、C3）
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --pretrained output/<时间戳>/c1_warmup/weights/best.pt \
    --skip_warmup \
    --device 0 \
    --skip_finetune \
    --name defect_detector
```

> 将 `<时间戳>` 替换为 C1 运行时生成的实际时间戳。如果跳过 C1 验证直接用完整管道，见下方「一键训练」。

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `optimizer` | SGD | 与 C1 保持一致，避免优化器切换 |
| `lr0` → `lrf` | 0.001 → 0.0001 | Cosine 衰减，延长有效学习窗口 |
| `mosaic` | 0.5（最后 15 epoch 关闭） | 适度 Mosaic，后期切换真实分布 |
| `mixup` | 0.0 | 关闭，让模型学习真实分布 |
| `copy_paste` | 0.3 (mode="flip") | 适度实例增强 |
| `close_mosaic` | 15 | 最后 15 个 epoch 关闭 Mosaic |

> 训练超参数定义在 [`config/train/full.yaml`](config/train/full.yaml)，可直接修改。

**✅ 验证检查点**：

| 指标 | 目标值 | 如何查看 |
|------|--------|---------|
| mAP50 | **> 0.70** | `grep "mAP50" output/<时间戳>/c2_full/results.csv` 末行 |
| mAP50-95 | **> 0.40** | `grep "mAP50-95" output/<时间戳>/c2_full/results.csv` 末行 |
| Precision | **> 0.85** | `results.csv` 中的 `metrics/precision(B)` |
| Recall | **> 0.85** | `results.csv` 中的 `metrics/recall(B)` |
| 类别平衡 | **最低类 AP50 > 0.50** | 检查每类 AP，无类别崩塌 |

检查各类 AP 的快速命令：
```bash
python -c "
from subway_yolo import YOLO
model = YOLO('output/<时间戳>/c2_full/weights/best.pt')
metrics = model.val(data='data/Defect_dataset/defect_data.yaml', split='val')
for i, ap in enumerate(metrics.ap_class_index):
    print(f'  Class {i}: AP50={metrics.ap50[i]:.3f}')
"
```

> **❌ 不通过**：mAP 停滞在 < 0.5 → 增强可能过强（降低 mosaic 至 0.5）；某类 AP=0 → 该类样本不足（运行 `python tool/generate_synthetic_defects.py --target_class <id>`）；val loss 上升而 train loss 下降 → 过拟合（增加 `weight_decay` 或减少 epochs）。

---

### Phase 5: Stage C3 — 微调（50 epochs）

**训练原理**：关闭 Mosaic 和 CopyPaste（它们改变数据分布），仅保留轻量 HSV + 几何增强。低 LR (0.0001) 恒速微调，让模型收敛到真实数据分布上。

```bash
# 从 C2 最佳权重开始，仅运行 C3（跳过 C1、C2）
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --pretrained output/<时间戳>/c2_full/weights/best.pt \
    --skip_warmup \
    --skip_full \
    --device 0 \
    --name defect_detector
```

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `mosaic` / `mixup` / `copy_paste` | 0 / 0 / 0 | 全部关闭，训练 = 推理分布 |
| `lr0` | 0.0001 | 恒速低 LR，精细调整 |
| `batch` | 8 | 减小 batch，更精确的梯度 |
| `degrees` / `scale` / `shear` | 2.0° / 0.3 / 1.0° | 弱几何增强 |
| `erasing` | 0.0 | 关闭随机擦除，保留缺陷细节 |

> 训练超参数定义在 [`config/train/finetune.yaml`](config/train/finetune.yaml)，可直接修改。

**✅ 验证检查点**（最终验收指标）：

| 指标 | 目标值 | 说明 |
|------|--------|------|
| mAP50 | **≥ 0.75** | 主要精度指标 |
| Precision | **≥ 0.90** | 满足项目验收标准 |
| Recall | **≥ 0.90** | 满足项目验收标准 |
| 推理耗时 | **≤ 10 秒/张** | 包含切片+推理+融合全流程 |

---

### 完整一键训练

如果你已经准备好数据集（Phase 1），可以直接使用三阶段自动衔接命令：

```bash
# 车载端：从 COCO 预训练开始，自动完成 C1→C2→C3
# 输出目录: output/<时间戳>/{c1_warmup,c2_full,c3_finetune}/
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --model subway_defect/models/yolo11s-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0

# 地面端 GPU 0：使用更大的 YOLO11m
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --model subway_defect/models/yolo11m-EMA-SimAM.yaml \
    --coco_pretrain \
    --device 0
```

---

### 训练检查清单

打印此清单，逐项确认：

| # | 阶段 | 操作 | 预期结果 | 实际结果 | ✅ |
|---|------|------|---------|---------|---|
| 1A | 自制数据准备 | `python tool/prepare_dataset.py` | ~1880 train + ~100 val | | |
| 1B | 数据校验 | `python tool/validate_dataset.py` | `[PASS]` | | |
| 1C | 多源数据扫描 | `python tool/multi_source_dataset_builder.py --scan-only` | 列出 autodl-pub 中可用数据集 | | |
| 1D | 多源数据构建 | `python tool/multi_source_dataset_builder.py` | `data/multi_datasets/mixed_pretrain/data.yaml` 存在 | | |
| 1E | 训练配置生成 | `python tool/multi_source_pretrain_yaml.py` | `config/train/pretrain/phase3_public_defect.yaml` 存在 | | |
| 2 | COCO 权重 | `python -c "from subway_yolo import YOLO; YOLO('yolo11s.pt')"` | 无报错 | | |
| 2A | [可选] TT100K P2 预热 | `train-defect --data config/train/pretrain/phase2_tiny_pretrain.yaml ...` | best.pt 保存, loss 正常下降 | | |
| 2B | 公开缺陷预训练 | `train-defect --data config/train/pretrain/phase3_public_defect.yaml ...` | mAP50 参考值, loss 稳定下降<br>输出: `weights/public_defect_pretrain.pt` | | |
| 3 | Neck/Head 适配 | `train-defect --data config/train/pretrain/phase4_neck_head_adapt.yaml ...` | mAP50 > 0.35, Recall 明显提升 | | |
| 4 | 主训练 (C2) | `train-defect --data config/train/pretrain/phase5_main_training.yaml ...` | mAP50 > 0.70, P/R > 0.85<br>输出: `output/<ts>/main/` | | |
| 5 | 短微调 (C3) | `train-defect --data config/train/pretrain/phase6_short_finetune.yaml ...` | mAP50 ≥ 0.75, P/R ≥ 0.90<br>输出: `output/<ts>/finetune/` | | |
| 6 | 推理验证 | `subway-server --model <final.pt> --mode vehicle` | 推理耗时 ≤ 10s, 结果正确 | | |

> **多源训练关键路径**: 1A→1B→2→2B→3→4→5→6。如果 AutoDL 环境不可用或只需快速验证，可使用原 C1→C2→C3 路径 (1A→1B→2→3→4→5→6，跳过 1C/1D/1E/2B)。

---

### 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| C1 loss 不下降 | LR 不合适 | 修改 `config/train/warmup.yaml` 中 `lr0`: 0.001 → 0.0005 |
| C1 完成后 mAP50 < 0.15 | 标注格式或匹配错误 | 检查 label 文件名与 image 文件名是否一一对应 |
| C2 mAP50 停滞在 0.4-0.5 | 增强过强，模型学不到真实分布 | 降低 `mosaic` 至 0.3, 降低 `copy_paste` 至 0.1（修改 `config/train/full.yaml`） |
| 某类别 AP50 = 0 | 该类标注样本过少 | `python tool/generate_synthetic_defects.py --target_class <id> --limit_per_class -1` |
| Val loss 上升，train loss 下降 | 过拟合 | 增大 `weight_decay` (0.0005→0.001), 增加增强强度 |
| GPU 显存不足 (OOM) | batch size 过大 | 减小 `batch` 至 8 或 4（修改 `config/train/<stage>.yaml`） |
| 训练意外中断 | 断电/超时 | 从最近 checkpoint 恢复：`--pretrained output/.../weights/last.pt --skip_warmup` |
| 增强图片效果异常 | 场景增强参数不当 | 预览增强效果：`python -c "from subway_defect.augmentations.scene import *; ..."` 检查输出 |
| 合成缺陷区域不自然 | Inpainting 修复痕迹明显 | 降低目标类 bbox 面积（仅对大目标 inpainting 效果较好） |
| `multi_source_dataset_builder.py` 找不到 autodl-pub | AutoDL 实例未挂载公有数据 | 在 AutoDL 控制台"公开数据"菜单搜索并挂载；或使用 `--no-download` 跳过 |
| Kaggle 下载失败 | 未配置 Kaggle API Key | `pip install kaggle; kaggle configure` 或手动下载后放入 `_downloads/` |
| DeepPCB 标注解析为空 | GitHub zip 结构与预期不同 | 检查 `PCBData/group*/` 目录是否存在；`--dry-run` 预览 |
| 公开预训练后 mAP 反而下降 | 域偏移过大或类别语义冲突 | 确认使用 `generic_defect` 单类合并（方案 B）；降低公开预训练 epochs |
| P2 小目标头训练不收敛 | P2 分支无 COCO 预训练权重 | 先跑 Phase 2A (TT100K P2 预热 80 epochs) 再进 Phase 2B |

> 所有训练超参数集中管理在 [`config/train/`](config/train/) 目录（YAML 格式），推理参数在 [`config/model/inference.yaml`](config/model/inference.yaml)。可直接修改，无需改代码。

## 推理部署

### 推理服务启动

```bash
# 车载端（单模型）
# 将 <时间戳> 替换为实际训练输出的时间戳
subway-server --port 8001 \
              --model output/<时间戳>/c3_finetune/weights/best.pt \
              --mode vehicle

# 地面端（双 GPU Ensemble + WBF 融合）
subway-server --port 8001 \
              --model output/<时间戳>/c3_finetune/weights/best.pt \
              --model_b output/<时间戳>_p2/c3_finetune/weights/best.pt \
              --mode ground
```

### API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 + GPU 状态 |
| `POST` | `/api/dl/infer` | 单张图像缺陷检测 |
| `POST` | `/api/dl/model/load` | 加载/切换模型 |

推理请求示例：
```json
{
  "image_path": "/data/images/20260624_001.jpg",
  "model_type": "vehicle",
  "confidence_threshold": 0.40,
  "slice_size": 1024,
  "slice_overlap": 0.15
}
```

> 完整 API 接口规范（与前后端集成的契约）见 [SPECIFICATION.md §6](SPECIFICATION.md#6-api-接口规格) 和 `subway_defect/docs/开发方案(5.30)/` 目录。

### TensorRT 加速导出

```bash
# FP16 导出（推荐，车载端必备）
export-tensorrt --model output/<时间戳>/c3_finetune/weights/best.pt --fp16

# 自定义输出路径
export-tensorrt --model best.pt --fp16 --output /path/to/model.engine

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

## 缺陷类别编码（18 类）

### 刚性接触网（13 类）

| 编码 | 中文名称 | 严重等级 |
| --- | --- | --- |
| `rigid_base_nut_missing` | 垂直悬吊安装底座螺母缺失 | serious |
| `rigid_base_nut_loose` | 垂直悬吊安装底座螺母松动 | serious |
| `rigid_single_bracket_base_nut_missing` | 单支垂直悬吊槽钢底座螺母缺失 | serious |
| `rigid_single_bracket_base_nut_loose` | 单支垂直悬吊槽钢底座螺母松动 | serious |
| `rigid_single_bracket_upper_nut_loose` | 单支垂直悬吊槽钢上方螺母松动 | normal |
| `rigid_hanger_top_plate_nut_missing` | 刚性悬挂吊柱顶板底面螺母缺失 | serious |
| `rigid_hanger_top_plate_nut_loose` | 刚性悬挂吊柱顶板底面螺母松动 | serious |
| `rigid_ground_wire_clamp_nut_missing` | 地线线夹托板安装底座螺母缺失 | serious |
| `rigid_ground_wire_clamp_nut_loose` | 地线线夹托板安装底座螺母松动 | serious |
| `rigid_ground_wire_nut_missing` | 地线线夹螺母缺失 | serious |
| `rigid_ground_wire_nut_loose` | 地线线夹螺母松动 | serious |
| `rigid_busbar_joint_bolt_missing` | 汇流排中间接头螺栓缺失 | **critical** |
| `rigid_insulator_damage` | 绝缘子破损 | **critical** |

### 柔性接触网 + 通用缺陷（5 类）

| 编码 | 中文名称 | 严重等级 |
| --- | --- | --- |
| `flex_wrist_base_hori_pin_missing` | 腕臂底座横向销钉缺开口销 | serious |
| `flex_wrist_base_vert_pin_missing` | 腕臂底座垂直销钉缺开口销 | serious |
| `flex_dropper_no_force` | 吊弦不受力 | serious |
| `foreign_object` | 异物侵入 | normal |
| `component_deformation` | 部件变形 | normal |

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

- [完整规格说明书](SPECIFICATION.md) — 项目需求、架构、接口、数据、测试等完整规范
- [AI 算法设计文档](subway_defect/docs/地铁接触网缺陷检测AI算法设计文档.md) — 核心算法完整设计
- [系统开发方案](subway_defect/docs/开发方案(5.30)/0-总览.md) — 前后端 + DL 系统架构
- [实现计划 1: 核心模型](subway_defect/docs/plans/2026-06-23-plan-1-core-model-architecture.md)
- [实现计划 2: 训练管道](subway_defect/docs/plans/2026-06-23-plan-2-training-pipeline.md)
- [实现计划 3: 推理引擎](subway_defect/docs/plans/2026-06-23-plan-3-inference-engine.md)
- [📊 分析: 模型特征学习效率诊断](docs/plans/2026-06-25-analysis-feature-learning-efficiency.md) — C2 零收益根因分析
- [🔧 方案: 结构改进与训练流程优化](docs/plans/2026-06-25-Structure-improvement-plam.md) — P2-Lite 架构 + 五阶段训练
- [📦 方案: 多源数据集训练建议](docs/plans/2026-06-25-Multi-source-datasets-training-recomendation.md) — 公开数据集整合策略

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
