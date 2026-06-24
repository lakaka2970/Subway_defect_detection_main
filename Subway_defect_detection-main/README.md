# 地铁接触网缺陷检测系统

基于 YOLO11 + EMA/SimAM 注意力的两阶段 AI 缺陷检测系统，用于福州地铁接触网超高清图像（1.27 亿像素）的智能化分析。

## 项目概述

本系统通过车载高速相机采集的接触网图像，自动识别螺栓松动/脱落、开口销缺失、绝缘子破损等 18 类缺陷，替代人工巡检。系统支持**车载端**（单 RTX 4090，离线运行）和**地面端**（双 RTX 4090，WBF 融合）两种部署形态。

### 核心指标

| 指标 | 车载端 | 地面端 |
| --- | --- | --- |
| GPU 配置 | 单卡 RTX 4090 | 双卡 RTX 4090 |
| 输入规格 | 1.27 亿像素（~13000×9800） | 1.27 亿像素 |
| 单张推理耗时 | ≤ 10 秒 | — |
| 检出率 Recall | ≥ 90% | ≥ 90% |
| 准确率 Precision | ≥ 90% | ≥ 90% |
| AI 加载延迟 | ≤ 1 秒 | ≤ 2 秒 |
| 提报率 | — | ≤ 5% |
| 部署形态 | 完全离线（工控机） | 内网服务 |

## 架构设计

### 两级级联推理管道

```
127MP 原始图像 (13000×9800)
       │
       ▼
┌─────────────────────┐
│ Stage 1: ROI 提案器  │  ← 降采样 1/8 (~1625×1225)
│ YOLO11n-ROI          │     检测结构区域（非缺陷）
│ 4 类: bolt_region,   │     切片: 640×640 × 9片
│ joint_region,        │     耗时: ~27ms
│ insulator_region,    │     Recall ≥ 99%（关键约束）
│ support_region       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ROI 映射 & 去重      │  ← 框映射回原始分辨率 + 自适应重叠
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

总耗时估算: 27ms (ROI) + 500ms (映射) + 2000ms (检测) + 200ms (融合) + 2500ms (解码) ≈ 5.2s ✅
```

### 模型注意力改造

对 YOLO11 在两处关键位置进行精确改造：

| 位置 | 模块 | 作用 | 参数量 | 延迟 |
| --- | --- | --- | --- | --- |
| P3 检测分支（小目标） | **EMA** | X/Y 双方向空间+通道多尺度注意力 | ~200 | +0.4ms |
| P4/P5 检测分支 | **SimAM** | 能量函数局部异常感知，零参数 | **0** | +0.1ms |
| Backbone 末端 | C2PSA（保留） | 原生位置敏感注意力 | — | — |

- **EMA（Efficient Multi-Scale Attention, ICASSP 2023）**：对 X 和 Y 方向分别池化，保留空间位置信息，增强 P3 分支对小目标（螺栓、开口销 ~8×8 px）的定位能力
- **SimAM（Simple Parameter-Free Attention, ICML 2021）**：基于神经科学空间抑制理论，零参数 → 零过拟合风险，在有限标注数据下尤其关键。对"局部出现异常"（如螺栓缺失）天然敏感

### 双卡异构 Ensemble（地面端）

```
GPU 0: YOLO11m-EMA (ECA 增强通道选择)
GPU 1: YOLO11m-P2 (4 尺度小目标特化)
         │                    │
         └────────┬───────────┘
                  ▼
        ┌─────────────────┐
        │  WBF 融合引擎     │
        │  IoU=0.55        │
        │  双模型采信≥0.50  │
        │  单模型采信≥0.75  │
        └─────────────────┘
```

两个模型错误模式不同（ECA vs P2 互补），WBF 融合后提报率从单模型 ~8-10% 降至 ~3-5%。

## 项目结构

```
Subway_defect_detection/
├── subway_defect/                    # 主包
│   ├── modules/                      # 自定义注意力模块
│   │   ├── EMA.py                    #   Efficient Multi-Scale Attention
│   │   └── SimAM.py                  #   Simple Parameter-Free Attention
│   │
│   ├── models/                       # 模型 YAML 配置文件
│   │   ├── yolo11s-EMA-SimAM.yaml    #   车载端 (s 规模, 9.5M 参数)
│   │   ├── yolo11m-EMA-SimAM.yaml    #   地面端 GPU 0 (ECA 变体)
│   │   └── yolo11m-P2-SimAM.yaml     #   地面端 GPU 1 (P2 四尺度变体)
│   │
│   ├── pipeline/                     # 推理管道
│   │   ├── slicer.py                 #   智能切片器（处理 127MP 图像）
│   │   ├── two_stage.py              #   两级级联推理 (ROI → 缺陷检测)
│   │   └── wbf_fusion.py             #   WBF 双模型融合（地面端）
│   │
│   ├── train/                        # 训练模块
│   │   ├── configs.py                #   四阶段训练超参数预设
│   │   ├── train_roi.py              #   Stage B: ROI 提案器训练
│   │   └── train_defect.py           #   Stage C: 缺陷检测三阶段训练
│   │
│   ├── augmentations/                # 数据增强
│   │   ├── scene.py                  #   隧道/烈日/运动模糊/雨雾模拟
│   │   └── contactnet_copy_paste.py  #   接触网特化 CopyPaste 增强
│   │
│   ├── deployment/                   # 部署模块
│   │   ├── export_tensorrt.py        #   TensorRT FP16/INT8 导出
│   │   └── fastapi_server.py         #   FastAPI 推理服务
│   │
│   └── synthetic/                    # 合成数据
│       └── defect_synthesis.py       #   Inpainting 缺陷合成（缺失样本生成）
│
├── ultralytics/                      # Vendored Ultralytics 框架
│   └── nn/Extramodule/               #   扩展模块注册桥接
│
├── tests/                            # 测试套件
│   ├── test_attention_modules.py     #   EMA/SimAM 单元 + 模型集成测试
│   ├── test_augmentations.py         #   增强管道 + 训练配置测试
│   └── test_pipeline.py              #   切片器 + WBF 融合 + 部署测试
│
├── scripts/
│   └── setup_autodl.sh               # AutoDL 云平台环境配置脚本
│
├── subway_defect/docs/               # 设计文档
│   ├── 地铁接触网缺陷检测AI算法设计文档.md   # 核心算法设计
│   ├── plans/                        #   实现计划
│   └── 开发方案(5.30)/                #   系统开发方案（前端+后端+DL接口）
│
├── pyproject.toml                    # 项目配置 (setuptools)
└── requirements.txt                  # 依赖清单
```

## 环境要求

- **Python**: ≥ 3.10
- **PyTorch**: ≥ 2.0（推荐 CUDA 12.1）
- **GPU**: NVIDIA GPU，VRAM ≥ 8 GB（推荐 RTX 4090）
- **操作系统**: Linux（训练/地面端） / Windows 10/11（车载端）

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone <repo-url>
cd Subway_defect_detection

# 安装（可编辑模式）
pip install -e .

# 验证安装
python -c "from subway_defect.modules.EMA import EMA; from subway_defect.modules.SimAM import SimAM; print('OK')"
```

### 2. 验证模型构建

```bash
# 验证三个模型 YAML 均可正常构建
python -c "
from ultralytics import YOLO
for cfg in ['subway_defect/models/yolo11s-EMA-SimAM.yaml',
            'subway_defect/models/yolo11m-EMA-SimAM.yaml',
            'subway_defect/models/yolo11m-P2-SimAM.yaml']:
    model = YOLO(cfg)
    print(f'{cfg}: {sum(p.numel() for p in model.model.parameters()):,} params')
"
```

### 3. 运行测试

```bash
# 运行全部测试
pytest tests/ -v

# 按模块运行
pytest tests/test_attention_modules.py -v   # EMA/SimAM 模块 + 模型集成
pytest tests/test_augmentations.py -v       # 增强管道 + 配置
pytest tests/test_pipeline.py -v            # 切片器 + WBF + 部署
```

## 训练流程

### Stage A: COCO 预训练（使用官方权重）

```bash
# 下载官方 COCO 预训练权重（自动）
# yolo11n.pt / yolo11s.pt / yolo11m.pt
# 下载地址: https://docs.ultralytics.com/models/yolo11/
```

### Stage B: ROI 提案器训练

```bash
# 训练结构区域检测模型（YOLO11n，轻量）
train-roi --data datasets/roi/roi_data.yaml \
          --model yolo11n.yaml \
          --epochs 200 \
          --device 0
```

训练目标：ROI Recall ≥ 99%（极低置信阈值 0.10-0.15，宁可多报，绝不漏报）

### Stage C: 缺陷检测三阶段训练

```bash
# 完整三阶段训练
# C1: Head 预热 (50 epochs, 冻结 backbone)
# C2: 全量训练 (200 epochs, 强增强, Cosine LR)
# C3: 微调     (50 epochs, 弱增强, 低 LR)

train-defect --data datasets/defects/defect_data.yaml \
             --model subway_defect/models/yolo11s-EMA-SimAM.yaml \
             --coco_pretrain \
             --device 0
```

可选跳过阶段：
```bash
# 从已有权重加载，跳过 C1 预热
train-defect --data datasets/defects/defect_data.yaml \
             --pretrained runs/defect_detector_c1_warmup/weights/best.pt \
             --skip_warmup

# 跳过 C3 微调
train-defect --data datasets/defects/defect_data.yaml --skip_finetune
```

### 训练超参数说明

| 阶段 | Epochs | Optimizer | LR | Batch | Mosaic | CopyPaste | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B (ROI) | 200 | SGD | 0.01 | 32 | 0.8 | 0 | 轻量模型，640 分辨率 |
| C1 (预热) | 50 | SGD | 0.001 | 16 | 0.5 | 0 | 冻结 backbone，只训练 head |
| C2 (主训练) | 200 | AdamW | 0.001→0.00001 | 16 | 0.8→0.1 | 0.6 | 全增强，Cosine LR |
| C3 (微调) | 50 | AdamW | 0.0001 | 8 | 0 | 0.4 | 接近真实分布收敛 |

### 数据增强体系

```
第四层: 合成生成 ─── Inpainting 缺失合成 + CG 渲染罕见缺陷  (离线, 50-80张/类)
第三层: 混合增强 ─── CopyPaste + Mosaic9                (在线, 中后期)
第二层: 定制增强 ─── 隧道/烈日/运动模糊/雨雾              (在线, 全程)
第一层: 基础扩增 ─── 翻转/缩放/HSV 偏移                  (在线, 全程)
```

## 推理部署

### 推理服务启动

```bash
# 车载端（单模型）
subway-server --port 8001 \
              --model runs/defect_detector_c2_full/weights/best.pt \
              --mode vehicle

# 地面端（双 GPU Ensemble + WBF 融合）
subway-server --port 8001 \
              --model runs/defect_detector_c2_full/weights/best.pt \
              --model_b runs/defect_detector_p2/weights/best.pt \
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

### TensorRT 加速导出

```bash
# FP16 导出（推荐，车载端必备）
export-tensorrt --model runs/defect_detector_c2_full/weights/best.pt --fp16

# INT8 导出（需校准数据集 200-500 张）
export-tensorrt --model runs/defect_detector_c2_full/weights/best.pt --int8 \
                --calibration_data datasets/calibration/
```

## 合成数据生成

```bash
# 通过 Inpainting 移除正常组件，生成"缺失"类缺陷样本
synthesize-defects --images datasets/images/train/ \
                   --labels datasets/labels/train/ \
                   --output datasets/synthetic/ \
                   --target_class 0 \
                   --limit 200
```

## 缺陷类型编码

### 刚性接触网缺陷（13 类）

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

## 模型选型策略

### 车载端（单 RTX 4090，≤ 10s）

```
主方案: YOLO11s-EMA-SimAM (FP16, 9.5M, ~5s)
备选方案: YOLO11m-EMA-SimAM (INT8, 20.1M, ~7s)
决策: 同时训练 s/m，实车验证后根据实测结果选择
```

### 地面端（双 RTX 4090，提报率 ≤ 5%）

```
GPU 0: YOLO11m-EMA-SimAM  (ECA 通道选择变体, 3 尺度)
GPU 1: YOLO11m-P2-SimAM   (P2 四尺度小目标特化)
融合: WBF (Weighted Boxes Fusion)
```

## 持续训练管道

```
数据采集(工程车) → 主动学习筛选 → 人工标注 → 数据集版本管理 → 自动训练触发 → 模型注册 & 部署

触发条件:
  · 新增标注数据 ≥ 200 张
  · 新增缺陷类别
  · 线上 Precision/Recall 低于验收线
  · 定期重训练（每季度）

部署准入条件:
  ① Recall ≥ 当前线上版本
  ② Precision ≥ 当前线上版本
  ③ 无类别退化: 任何类 Recall 下降 ≤ 5%
  ④ 推理耗时不增加
```

## 技术参考

| 模块 | 参考论文 | 出处 |
| --- | --- | --- |
| YOLO11 | Ultralytics YOLO11 | https://docs.ultralytics.com/models/yolo11 |
| EMA | Efficient Multi-Scale Attention | ICASSP 2023 |
| SimAM | A Simple, Parameter-Free Attention Module | ICML 2021 |
| CopyPaste | Simple Copy-Paste is a Strong Data Augmentation Method | ICCV 2021 |
| WBF | Weighted Boxes Fusion: Ensembling boxes | arXiv 1910.13302 |
| SAHI | Slicing Aided Hyper Inference | arXiv 2206 |

## 设计文档

- [地铁接触网缺陷检测 AI 算法设计文档](subway_defect/docs/地铁接触网缺陷检测AI算法设计文档.md) — 核心算法完整设计
- [系统开发方案总览](subway_defect/docs/开发方案(5.30)/0-总览.md) — 前后端 + DL 系统架构
- [实现计划 1: 核心模型架构](subway_defect/docs/plans/2026-06-23-plan-1-core-model-architecture.md)
- [实现计划 2: 训练管道](subway_defect/docs/plans/2026-06-23-plan-2-training-pipeline.md)
- [实现计划 3: 推理引擎](subway_defect/docs/plans/2026-06-23-plan-3-inference-engine.md)

## License

AGPL-3.0 License — 与 Ultralytics YOLO 框架保持一致。
