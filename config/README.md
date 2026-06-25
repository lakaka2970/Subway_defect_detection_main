# 项目配置目录

本目录集中管理训练、验证和推理参数，便于查看和修改。

## 目录结构

```
config/
├── train/                # 训练超参数
│   ├── warmup.yaml       # C1 — Head Warmup (冻结 backbone)
│   ├── full.yaml         # C2 — Full Training (完整训练)
│   └── finetune.yaml     # C3 — Fine-Tune (微调)
├── model/                # 模型推理参数
│   └── inference.yaml    # 推理 & 验证默认值
└── README.md             # 本文件
```

## 训练配置 (`config/train/`)

每个 YAML 文件对应一个训练阶段，参数会传递给 `YOLO.train()`：

| 阶段 | 文件 | 用途 | epochs | 数据增强 |
|------|------|------|--------|----------|
| C1 | `warmup.yaml` | 冻结 backbone 训练检测头 | 50 | 轻量 |
| C2 | `full.yaml` | 解冻全部层完整训练 | 200 | 中等 |
| C3 | `finetune.yaml` | 低 LR 微调 | 50 | 极轻量 |

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
