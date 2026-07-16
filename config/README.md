# 项目配置目录

本目录集中管理训练、验证和推理参数，便于查看和修改。

## 目录结构

```
config/
├── train/                     # 训练超参数
│   ├── warmup.yaml            # [LEGACY] C1 Head Warmup
│   ├── full.yaml              # [LEGACY] C2 Full Training
│   ├── finetune.yaml          # [LEGACY] C3 Fine-Tune
│   └── pretrain/              # 统一训练阶段配置 (推荐, v2 优化)
│       ├── stage_p2_tiny_pretrain.yaml     # Stage P2: P2头预热 (可选, 生成)
│       ├── stage1a_public_head.yaml        # Stage 1A: 公开缺陷 Head/Neck 预热 (生成)
│       ├── stage1b_public_backbone.yaml    # Stage 1B: 公开缺陷 Backbone 适应 (生成)
│       ├── stage2_domain_adapt.yaml        # Stage 2: 领域适配 (v2: 40ep)
│       ├── stage3_main_training.yaml       # Stage 3: 主训练 (v2: 80ep, copy_paste=0.05)
│       ├── stage4_short_finetune.yaml      # Stage 4: 短微调 (v2: 15ep, warmup=0)
│       └── stage5_hard_negative.yaml       # Stage 5: 难负样本+阈值校准 (v2: 脚本已实现)
├── model/                     # 模型推理参数
│   └── inference.yaml         # 推理 & 验证默认值
└── README.md                  # 本文件
```

---

## 训练模式

### 统一训练阶段 (推荐)

**推荐使用一键自动化**：

```bash
python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0
```

手动逐步训练：

```bash
train-defect --data data/subway_crops/subway_crops.yaml \
    --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \
    --coco_pretrain --device 0 --stages 1 2 3 4 --pretrain-config-dir
```

| 阶段 | 文件 | epochs | imgsz | 优化器 | lr0 | 关键特性 |
|------|------|--------|-------|--------|-----|----------|
| Stage P2 | `stage_p2_tiny_pretrain.yaml` | 80 | 1024 | AdamW | 0.001 | [消融实验] P2头预热 — 四尺度模型已证明更差, 不推荐 |
| Stage 1A | `stage1a_public_head.yaml` | 40 | 1024 | AdamW | 0.001 | Neck/Head预热, freeze[0..10], gc10+neu+sdd2+rsdds |
| Stage 1B | `stage1b_public_backbone.yaml` | 60 | 1024 | AdamW | 0.0003 | Backbone适应, freeze[0..5], 低LR |
| Stage 2 | `stage2_domain_adapt.yaml` | 40 | 1024 | AdamW | 0.0008 | 1类→7类, freeze[0..7], cos_lr |
| Stage 3 | `stage3_main_training.yaml` | 80 | 1280 | AdamW | 0.0005 | 全解冻, copy_paste=0.05, patience=28 |
| Stage 4 | `stage4_short_finetune.yaml` | 15 | 1280 | AdamW | 1e-5 | 零增强, warmup=0, patience=5 |
| Stage 5 | `stage5_hard_negative.yaml` | 20 | 1280 | AdamW | 2e-5 | (可选) 难负样本+阈值校准, warmup=0 |

**关键特性**:
- 每阶段可独立使用不同数据集 (`data`/`nc`/`names` 在 YAML 内指定)
- AdamW + cos_lr 全程 (v2: 自适应学习率 + 余弦退火调度)
- 极轻增强保护小目标 (mosaic≤0.15, erasing=0)
- v2: 缺陷感知 Copy-Paste (Stage 3, 仅小目标)
- v2: 类别感知场景增强 (震动模糊 + 白平衡偏移 + 少数类过采样)
- v2: 图像切片类别平衡 (少数类额外 offset 变体)
- v2: 位置去偏置 crop (--debiasing: 中心/偏中心/边缘/角落 系统性变化)
- v2: 标注质量审计工具 (audit_labels.py — FP/FN 标记 + 人工审核辅助)
- 每阶段产出可独立验证，问题可精确定位
- v2: 完整 Stage 5 脚本链 (collect_hard_negatives → train → calibrate_thresholds)
- P2 四尺度模型标注为 [消融实验] — 三尺度 YOLO11m-EMA-SimAM 为主线

<details>
<summary><b>Legacy C1/C2/C3 三阶段（向后兼容，不推荐）</b></summary>

不加 `--stages` 时默认使用 `config/train/{warmup,full,finetune}.yaml`:

| 阶段 | 文件 | epochs | imgsz | 优化器 | lr0 | 限制 |
|------|------|--------|-------|--------|-----|------|
| C1 | `warmup.yaml` | 50 | 1024 | SGD | 0.001 | 仅训 head |
| C2 | `full.yaml` | 300 | 1280 | AdamW | 0.001 | 增强过强, 小目标退化 |
| C3 | `finetune.yaml` | 100 | 1280 | AdamW | 5e-5 | 时间长反而过拟合 |

</details>

### v2 优化历史 (2025-06-25) — 已融入统一阶段

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

| 模型 | imgsz=640 | imgsz=1024 | imgsz=1280 |
|------|----------|-----------|-----------|
| yolo11n | ~0.25 GB/sample | ~0.45 GB/sample | ~0.70 GB/sample |
| yolo11s | ~0.35 GB/sample | ~0.80 GB/sample | ~1.25 GB/sample |
| yolo11s-P2 | ~0.45 GB/sample | ~1.05 GB/sample | ~1.64 GB/sample |
| yolo11m | ~0.55 GB/sample | ~1.10 GB/sample | ~1.72 GB/sample |
| yolo11m-P2 | ~0.70 GB/sample | ~1.40 GB/sample | ~2.19 GB/sample |

RTX 4090 (24GB): yolo11s @ 1280 推荐 batch=10~12, yolo11s-P2 @ 1280 推荐 batch=6~8。
RTX 5090 (32GB): yolo11s @ 1280 推荐 batch=14~16, yolo11s-P2 @ 1280 推荐 batch=10~12。
开启 multi_scale 后推荐额外降低 20-30% batch size（以峰值 imgsz 估算）。
