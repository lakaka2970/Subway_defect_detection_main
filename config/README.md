# 项目配置目录

本目录集中管理训练、验证和推理参数，便于查看和修改。

## 目录结构

```
config/
├── train/                     # 训练超参数
│   ├── warmup.yaml            # C1 — Head Warmup [LEGACY — 推荐用 pretrain/]
│   ├── full.yaml              # C2 — Full Training [LEGACY — 推荐用 pretrain/]
│   ├── finetune.yaml          # C3 — Fine-Tune   [LEGACY — 推荐用 pretrain/]
│   └── pretrain/              # 现代五阶段训练配置 (推荐)
│       ├── stage1_neck_head_warmup.yaml
│       ├── stage2_scale_adaptation.yaml
│       ├── stage3_short_finetune.yaml
│       └── stage4_hard_negative.yaml
├── model/                     # 模型推理参数
│   └── inference.yaml         # 推理 & 验证默认值
└── README.md                  # 本文件
```

---

## 训练模式

### 现代五阶段 (推荐)

使用 `config/train/pretrain/` 下的配置, 配合 `--stages` 和 `--pretrain-config-dir` 参数:

```bash
train-defect --data data/subway_crops/subway_crops.yaml \
    --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \
    --coco_pretrain --device 0 --stages 1 2 3 --pretrain-config-dir
```

| 阶段 | 文件 | epochs | imgsz | 优化器 | lr0 | 关键特性 |
|------|------|--------|-------|--------|-----|----------|
| S1 | `stage1_neck_head_warmup.yaml` | 50 | 1024 | AdamW | 0.001 | 冻结 backbone 前 60%, 训练 neck+head+attention |
| S2 | `stage2_scale_adaptation.yaml` | 120 | 1280 | AdamW | 0.0008 | 全解冻, mosaic=0.2, 无 erasing/copy_paste, patience=40 |
| S3 | `stage3_short_finetune.yaml` | 30 | 1280 | AdamW | 3e-5 | 短微调, patience=8, 每 epoch 保存, 选择 best_mAP50-95 |
| S4 | `stage4_hard_negative.yaml` | 30 | 1280 | AdamW | 2e-5 | Hard negative mining, 零增强, 每类阈值校准 |

**与 legacy 三阶段的关键区别**:
- 每阶段可使用 **不同的数据集** (stage YAML 中指定 `data`/`nc`/`names`)
- AdamW 全程替代 SGD (自适应学习率, 不依赖动量连续性)
- 极轻增强 (mosaic≤0.2, 无 erasing/copy_paste) 保护小目标
- 短 C3 + early stopping 防止过拟合退化
- 训练过程自动记录 per-class AP、loss dynamics、hard examples

### Legacy 三阶段 (向后兼容)

保留原有 `config/train/warmup|full|finetune.yaml`, 不加 `--stages` 时默认使用:

| 阶段 | 文件 | epochs | imgsz | 优化器 | lr0 | 关键特性 |
|------|------|--------|-------|--------|-----|----------|
| C1 | `warmup.yaml` | 50 | 1024 | SGD | 0.001 | 冻结backbone, mosaic=0.5 |
| C2 | `full.yaml` | 300 | 1280 | AdamW | 0.001 | multi_scale=0.5, mosaic=0.3, erasing=0.1 |
| C3 | `finetune.yaml` | 100 | 1280 | AdamW | 5e-5 | 极轻量增强, copy_paste/erasing禁用 |

### v2 优化要点 (2025-06-25)

基于四次完整训练的深入分析（详见 `docs/plans/2026-06-25-analysis-feature-learning-efficiency.md`）：

| 改动 | C1 | C2 | C3 | 原因 |
|------|-----|-----|-----|------|
| imgsz | — | 1024→**1280** | 1024→**1280** | 缺陷从8→10px, 提升P3检测能力 |
| optimizer | — | SGD→**AdamW** | SGD→**AdamW** | 自适应LR补偿C1→C2动量丢失 |
| lr0 | — | — | 1e-4→**5e-5** | AdamW下更低精调LR |
| epochs | — | 200→**300** | 50→**100** | 更多迭代匹配高分辨率 |
| multi_scale | — | 新增 **0.5** | — | 尺度不变特征学习 |
| mosaic | — | 0.5→**0.3** | — | 减少对小目标的缩略破坏 |
| copy_paste | — | 0.3→**0** | — | 小目标场景禁用, 避免伪影 |
| erasing | — | 0.4→**0.1** | — | 保护8-10px小目标 |
| scale | — | 0.5→**0.7** | — | 更大缩放范围增强尺度不变性 |
| close_mosaic | — | 15→**40** | — | 更长无mosaic纯净训练 |
| warmup_epochs | — | 5→**10** | 3→**5** | 更长过渡, 补偿骨干冷启动 |
| warmup_momentum | — | 0.8→**0.5** | — | 更慢初始动量, 平稳过渡 |

**修改后需要重新训练才能生效。**

---

## 推理配置 (`config/model/inference.yaml`)

包含 YOLO 通用推理、两阶段管线、WBF 融合等全部推理参数。

**修改后重启推理服务即可生效，无需重新训练。**

### 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `conf` | 0.25 | 检测置信度阈值 |
| `iou` | 0.7 | NMS IoU 阈值 |
| `max_det` | 300 | 每图最大检测数 |
| `imgsz` | 1024 | 推理图像尺寸 |
| `roi_conf` | 0.15 | ROI 阶段置信度 |
| `defect_conf` | 0.40 | 缺陷检测置信度 |
| `slice_size` | 1024 | 切片尺寸 |
| `wbf_iou` | 0.55 | WBF 融合 IoU |

---

## VRAM 估算参考

| 模型 | imgsz=1024 | imgsz=1280 |
|------|-----------|-----------|
| yolo11n | ~0.25 GB/sample | — |
| yolo11s | ~0.80 GB/sample | ~1.25 GB/sample |
| yolo11s-P2 | ~1.05 GB/sample | ~1.64 GB/sample |
| yolo11m | ~1.10 GB/sample | ~1.72 GB/sample |
| yolo11m-P2 | ~1.40 GB/sample | ~2.19 GB/sample |

RTX 4090 (24GB): yolo11s @ 1280 推荐 batch=10~12, yolo11s-P2 @ 1280 推荐 batch=6~8。
RTX 5090 (32GB): yolo11s @ 1280 推荐 batch=14~16, yolo11s-P2 @ 1280 推荐 batch=10~12。
开启 multi_scale 后推荐额外降低 20-30% batch size（以峰值 imgsz 估算）。
