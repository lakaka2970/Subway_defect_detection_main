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

本项目采用 **"COCO 预训练基础能力 → 自制数据集迁移学习 → 场景微调"** 的三阶段训练策略：

```
COCO 2017 (118K 图像, 80 类)                   自制数据集 (~1880 张, 7 类)
        │                                                │
        ▼                                                ▼
  ┌──────────┐    Stage C1 (50 ep)    ┌──────────┐    Stage C2 (200 ep)    ┌──────────┐    Stage C3 (50 ep)    ┌──────────┐
  │ 预训练权重 │ ──────────────────→  │ Head 预热 │ ────────────────────→  │  全量训练  │ ────────────────────→  │   微调    │
  │ yolo11s.pt│   冻结 backbone        │ best.pt   │   解冻 + 强增强        │ best.pt   │   弱增强 + 低 LR       │ 最终模型  │
  └──────────┘                        └──────────┘                        └──────────┘                        └──────────┘
   mAP50: 0.55 (COCO)                  mAP50 > 0.30                        mAP50 > 0.70                        mAP50 ≥ 0.75
                                       Box Loss < 1.5                      Precision > 0.85                    Precision ≥ 0.90
                                       ~0.5h (RTX 4090)                    Recall > 0.85                       Recall ≥ 0.90
                                                                           ~4h (RTX 4090)                      ~0.5h (RTX 4090)
```

> **核心原则**: 每一阶段产出可独立验证，问题可精确追溯。如果某一阶段指标不达标，立即排查该阶段的问题，不进入下一阶段。

---

### Phase 1: 数据集准备

训练前必须完成数据集构建。项目提供了 `tool/` 目录下的一键准备脚本：

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

### Phase 3: Stage C1 — Head 预热（50 epochs）

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

> 输出目录：`output/defect_detector_<时间戳>_c1_warmup/`

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `freeze` | `[0..10]` | 冻结所有 backbone 层 |
| `lr0` | 0.001 | 恒定学习率，不过早降低 |
| `mosaic` | 0.5 | 适度 Mosaic，不过度扭曲 |
| `mixup` / `copy_paste` | 0 / 0 | 此阶段关闭混合增强 |

**✅ 验证检查点**：

| 指标 | 目标值 | 如何查看 |
|------|--------|---------|
| best.pt 已保存 | 文件存在 | `ls output/defect_detector_<时间戳>_c1_warmup/weights/best.pt` |
| mAP50 | **> 0.30** | 训练日志或 `results.csv` 中的 `metrics/mAP50(B)` |
| Box Loss | **< 1.5** | `results.csv` 中的 `train/box_loss` |
| 训练曲线 | 持续下降，无发散 | `output/defect_detector_<时间戳>_c1_warmup/results.png` |

> **❌ 不通过**：Loss 持续不降 → 降低 LR 至 0.0005；mAP 始终 < 0.10 → 检查标注与图像是否匹配。

---

### Phase 4: Stage C2 — 全量训练（200 epochs）

**训练原理**：解冻 backbone，启用全套增强（Mosaic 0.8 → 0, MixUp 0.15, CopyPaste 0.6），Cosine LR 从 0.001 衰减至 0.00001。这是模型能力提升最大的阶段。

```bash
# 从 C1 最佳权重开始，仅运行 C2（跳过 C1、C3）
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --pretrained output/defect_detector_<时间戳>_c1_warmup/weights/best.pt \
    --skip_warmup \
    --device 0 \
    --skip_finetune \
    --name defect_detector
```

> 将 `<时间戳>` 替换为 C1 运行时生成的实际时间戳。如果跳过 C1 验证直接用完整管道，见下方「一键训练」。

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `optimizer` | AdamW | 自适应学习率，适合长周期训练 |
| `lr0` → `lrf` | 0.001 → 0.00001 | Cosine 衰减，平滑收敛 |
| `mosaic` | 0.8（epoch 190 关闭） | 强 Mosaic 前期，后期关闭逼近真实分布 |
| `mixup` | 0.15 | 跨图混合，增加样本多样性 |
| `copy_paste` | 0.6 (mode="flip") | 同类内复制粘贴，缓解类别不平衡 |
| `close_mosaic` | 190 | 最后 10 个 epoch 关闭 Mosaic |

**✅ 验证检查点**：

| 指标 | 目标值 | 如何查看 |
|------|--------|---------|
| mAP50 | **> 0.70** | `grep "mAP50" output/..._c2_full/results.csv` 末行 |
| mAP50-95 | **> 0.40** | `grep "mAP50-95" output/..._c2_full/results.csv` 末行 |
| Precision | **> 0.85** | `results.csv` 中的 `metrics/precision(B)` |
| Recall | **> 0.85** | `results.csv` 中的 `metrics/recall(B)` |
| 类别平衡 | **最低类 AP50 > 0.50** | 检查每类 AP，无类别崩塌 |

检查各类 AP 的快速命令：
```bash
python -c "
from subway_yolo import YOLO
model = YOLO('output/defect_detector_<时间戳>_c2_full/weights/best.pt')
metrics = model.val(data='data/Defect_dataset/defect_data.yaml', split='val')
for i, ap in enumerate(metrics.ap_class_index):
    print(f'  Class {i}: AP50={metrics.ap50[i]:.3f}')
"
```

> **❌ 不通过**：mAP 停滞在 < 0.5 → 增强可能过强（降低 mosaic 至 0.5）；某类 AP=0 → 该类样本不足（运行 `python tool/generate_synthetic_defects.py --target_class <id>`）；val loss 上升而 train loss 下降 → 过拟合（增加 `weight_decay` 或减少 epochs）。

---

### Phase 5: Stage C3 — 微调（50 epochs）

**训练原理**：关闭 Mosaic 和 MixUp（它们改变数据分布），仅保留 CopyPaste 0.4 和轻量几何增强。低 LR (0.0001) 恒速微调，让模型收敛到真实数据分布上。

```bash
# 从 C2 最佳权重开始，仅运行 C3（跳过 C1、C2）
train-defect \
    --data data/Defect_dataset/defect_data.yaml \
    --pretrained output/defect_detector_<时间戳>_c2_full/weights/best.pt \
    --skip_warmup \
    --skip_full \
    --device 0 \
    --name defect_detector
```

| 关键参数 | 值 | 说明 |
|----------|-----|------|
| `mosaic` / `mixup` | 0 / 0 | 关闭，训练数据分布 = 推理分布 |
| `lr0` | 0.0001 | 恒定低 LR，精细调整 |
| `batch` | 8 | 减小 batch，更精确的梯度 |
| `degrees` / `scale` / `shear` | 2.0° / 0.3 / 1.0° | 弱几何增强 |
| `copy_paste` | 0.4 | 保留适度实例增强 |

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
# 输出目录: output/defect_detector_<时间戳>_c{1,2,3}_{warmup,full,finetune}/
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
| 1 | 数据准备 | `python tool/prepare_dataset.py` | ~1880 train + ~100 val | | |
| 2 | 数据校验 | `python tool/validate_dataset.py` | `[PASS]` | | |
| 3 | COCO 权重 | `python -c "from subway_yolo import YOLO; YOLO('yolo11s.pt')"` | 无报错 | | |
| 4 | C1 预热 | `train-defect ... --coco_pretrain --skip_full --skip_finetune` | mAP50 > 0.30, Box Loss < 1.5<br>输出: `output/..._c1_warmup/` | | |
| 5 | C2 主训练 | `train-defect ... --pretrained <c1_best> --skip_warmup --skip_finetune` | mAP50 > 0.70, P/R > 0.85<br>输出: `output/..._c2_full/` | | |
| 6 | C3 微调 | `train-defect ... --pretrained <c2_best> --skip_warmup --skip_full` | mAP50 ≥ 0.75, P/R ≥ 0.90<br>输出: `output/..._c3_finetune/` | | |
| 7 | 推理验证 | `subway-server --model <final.pt> --mode vehicle` | 推理耗时 ≤ 10s, 结果正确 | | |

---

### 常见问题排查

| 症状 | 可能原因 | 解决方案 |
|------|---------|---------|
| C1 loss 不下降 | LR 不合适 | 修改 `configs.py` 中 `lr0`: 0.001 → 0.0005 |
| C1 完成后 mAP50 < 0.15 | 标注格式或匹配错误 | 检查 label 文件名与 image 文件名是否一一对应 |
| C2 mAP50 停滞在 0.4-0.5 | 增强过强，模型学不到真实分布 | 降低 `mosaic` 至 0.5, 降低 `copy_paste` 至 0.3 |
| 某类别 AP50 = 0 | 该类标注样本过少 | `python tool/generate_synthetic_defects.py --target_class <id> --limit_per_class -1` |
| Val loss 上升，train loss 下降 | 过拟合 | 增大 `weight_decay` (0.0005→0.001), 增加增强强度 |
| GPU 显存不足 (OOM) | batch size 过大 | 减小 `batch` 至 8 或 4（修改 `configs.py`） |
| 训练意外中断 | 断电/超时 | 从最近 checkpoint 恢复：`--pretrained output/.../weights/last.pt --skip_warmup` |
| 增强图片效果异常 | 场景增强参数不当 | 预览增强效果：`python -c "from subway_defect.augmentations.scene import *; ..."` 检查输出 |
| 合成缺陷区域不自然 | Inpainting 修复痕迹明显 | 降低目标类 bbox 面积（仅对大目标 inpainting 效果较好） |

> 所有训练超参数集中管理在 [`subway_defect/train/configs.py`](subway_defect/train/configs.py)，可直接修改。

## 推理部署

### 推理服务启动

```bash
# 车载端（单模型）
# 将 <时间戳> 替换为实际训练输出的时间戳
subway-server --port 8001 \
              --model output/defect_detector_<时间戳>_c3_finetune/weights/best.pt \
              --mode vehicle

# 地面端（双 GPU Ensemble + WBF 融合）
subway-server --port 8001 \
              --model output/defect_detector_<时间戳>_c3_finetune/weights/best.pt \
              --model_b output/defect_detector_p2_<时间戳>_c3_finetune/weights/best.pt \
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
export-tensorrt --model output/defect_detector_<时间戳>_c3_finetune/weights/best.pt --fp16

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
