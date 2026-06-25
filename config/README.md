# 项目配置目录

本目录集中管理训练、验证和推理参数，便于查看和修改。

## 目录结构

```
config/
├── train/                # 训练超参数
│   ├── warmup.yaml       # C1 — Head Warmup (冻结 backbone, 50 epochs)
│   ├── full.yaml         # C2 — Full Training (完整训练, 300 epochs, v2优化版)
│   └── finetune.yaml     # C3 — Fine-Tune (微调, 100 epochs, v2优化版)
├── model/                # 模型推理参数
│   └── inference.yaml    # 推理 & 验证默认值
└── README.md             # 本文件
```

## 训练配置 (`config/train/`)

每个 YAML 文件对应一个训练阶段，参数会传递给 `YOLO.train()`：

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

## VRAM 估算参考

| 模型 | imgsz=1024 | imgsz=1280 |
|------|-----------|-----------|
| yolo11s | ~0.80 GB/sample | ~1.25 GB/sample |
| yolo11m | ~1.10 GB/sample | ~1.72 GB/sample |
| yolo11m-P2 | ~1.40 GB/sample | ~2.19 GB/sample |

RTX 5090 (32GB): yolo11s @ 1280 推荐 batch=14~16, 开启 multi_scale 后推荐 batch=10~12（以峰值 imgsz 估算）。
