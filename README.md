# 地铁接触网缺陷检测 AI 模块

基于 YOLO11 + EMA/SimAM 注意力的两阶段深度学习缺陷检测系统，用于福州地铁接触网超高清图像（1.27 亿像素）的智能化分析。

> **用途**: 本项目作为大项目仓库的 AI/深度学习算法模块，为前后端提供 RESTful 推理服务。
>
> 详细训练指南见 [docs/Train_guide.md](docs/Train_guide.md)，完整规格见 [SPECIFICATION.md](SPECIFICATION.md)。

---

## 目录结构

```
Subway_defect_AI/
├── subway_defect/                 # 核心 AI 包
│   ├── deployment/                # FastAPI 推理服务 + TensorRT 导出
│   │   ├── fastapi_server.py      # ★ 推理服务入口（前后端调用的接口）
│   │   ├── defect_dict.json       # 缺陷字典（code → 中文名 → 严重等级）
│   │   └── export_tensorrt.py     # TensorRT 引擎导出
│   ├── pipeline/                  # 两阶段推理管道（切片 → ROI → 检测 → WBF 融合）
│   ├── train/                     # 训练模块（CLI + 多阶段管道 + 回调系统）
│   ├── modules/                   # EMA / SimAM / ECA 注意力模块
│   ├── models/                    # 模型 YAML 配置（4 个变体）
│   ├── augmentations/             # 数据增强（隧道/日照/模糊/雨雾 + CopyPaste）
│   ├── synthetic/                 # Inpainting 合成缺陷生成
│   └── classes.py                 # 缺陷类别中央注册表（16 类，单一事实来源）
│
├── subway_yolo/                   # Vendored YOLO11 框架（精简，含自定义注意力桥接）
│   ├── engine/                    # Model / Trainer / Predictor / Validator / Exporter
│   ├── nn/                        # 网络模块 + Extramodule（EMA/SimAM 注册）
│   ├── models/yolo/               # detect + classify 实现
│   └── utils/                     # 核心工具函数
│
├── config/                        # YAML 集中配置
│   ├── model/inference.yaml       # 推理参数（修改后重启服务即可生效）
│   └── train/                     # 训练超参数（Legacy 三阶段 + Modern 五阶段）
│
├── scripts/                       # 数据集工具脚本
│   ├── prepare_dataset.py         # 一键数据集准备
│   ├── generate_native_crops.py   # 原生分辨率 crop 生成
│   ├── validate_dataset.py        # 数据集完整性校验
│   └── multi_source_dataset_builder.py  # 多源公开数据集构建器
│
├── tests/                         # 测试套件（pytest, 773 用例）
├── weights/                       # 预训练权重（检测 n/s/m + COCO 分类/分割/姿态/旋转框）
├── pyproject.toml                 # 包配置 + CLI 入口点
├── README.md                      # 本文件 — 结构与使用说明
├── SPECIFICATION.md               # 完整规格说明书（需求/架构/API/数据/模型/测试）
└── LICENSE                        # AGPL-3.0
```

---

## 快速开始

### 环境要求

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0（推荐 CUDA 12.1）
- **GPU** NVIDIA GPU，VRAM ≥ 8 GB（推荐 RTX 4090）

### 安装

```bash
pip install -e .
```

### 启动推理服务

```bash
# 车载端（单模型）
subway-server --port 8001 \
    --model output/<时间戳>/c3_finetune/weights/best.pt \
    --mode onboard

# 地面端（双 GPU WBF 融合）
subway-server --port 8001 \
    --model output/<时间戳>/c3_finetune/weights/best.pt \
    --model_b output/<时间戳>_p2/c3_finetune/weights/best.pt \
    --mode ground
```

### 验证

```bash
# 健康检查
curl http://localhost:8001/api/dl/health

# 单张推理
curl -X POST http://localhost:8001/api/dl/infer \
  -H "Content-Type: application/json" \
  -d '{"imagePath":"/path/to/image.jpg","modelType":"onboard","confidenceThreshold":0.4}'

# 运行测试
pytest tests/ -v
```

---

## API 接口

> 接口遵循 `docs/开发接口规范/` 中定义的三方统一契约，字段使用 **camelCase**（与 Java 后端对齐）。

### 端点一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/dl/health` | 健康检查 + 已加载模型 + GPU 状态 |
| `POST` | `/api/dl/infer` | 单张图像缺陷检测（核心接口） |
| `POST` | `/api/dl/infer/batch` | 批量推理（地面端高吞吐） |
| `POST` | `/api/dl/model/load` | 加载/切换模型 |

### 推理请求示例

```json
{
  "imagePath": "/data/images/20260624_001.jpg",
  "modelType": "onboard",
  "confidenceThreshold": 0.40,
  "outputCoordType": "normalized",
  "extraParams": {
    "sliceSize": 1024,
    "sliceOverlap": 0.15
  }
}
```

### 推理响应示例

```json
{
  "success": true,
  "imagePath": "/data/images/20260624_001.jpg",
  "processingTimeMs": 5234.5,
  "totalSlices": 72,
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

### 错误码

| 错误码 | HTTP 状态 | 说明 |
|--------|-----------|------|
| `DL_MODEL_NOT_LOADED` | 503 | 模型未加载 |
| `DL_GPU_OOM` | 507 | GPU 显存不足 |
| `DL_IMAGE_UNREADABLE` | 400 | 图像损坏/无法读取 |
| `DL_INFERENCE_TIMEOUT` | 504 | 推理超时（> 30s） |
| `DL_INTERNAL_ERROR` | 500 | 内部异常 |

---

## 缺陷类别（16 类）

| 编码 | 中文名称 | 严重等级 | 训练状态 |
|------|----------|----------|:--------:|
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
| `CBHPM` | 腕臂底座横向销钉缺口 | serious | ★ |
| `CBVPM` | 腕臂底座垂直销钉缺口 | serious | ★ |
| `DRPS` | 吊弦不受力 | serious | |

> ★ = 已有标注训练数据。权威来源：`docs/接触网缺陷类型详解.docx`。机器可读字典：`subway_defect/deployment/defect_dict.json`。

---

## CLI 入口点

| 命令 | 说明 |
|------|------|
| `subway-server` | 启动 FastAPI 推理服务 |
| `train-defect` | 缺陷检测模型训练（Legacy 三阶段 / Modern 五阶段） |
| `train-roi` | ROI 提案器训练 |
| `synthesize-defects` | Inpainting 合成缺陷数据生成 |
| `export-tensorrt` | 导出 TensorRT FP16/INT8 引擎 |

---

## 坐标系约定

缺陷框统一采用 **归一化中心点 + 宽高**（YOLO 格式），取值 0~1：

```json
"box": { "x": 0.3421, "y": 0.5612, "w": 0.0123, "h": 0.0089 },
"coordType": "normalized"
```

前端渲染换算：`pixel_x = x * imageWidth`, `pixel_y = y * imageHeight`

---

## 模型选型

| 模型 | 用途 | 参数量 | 检测尺度 | 注意力 |
|------|------|--------|---------|--------|
| YOLO11n-ROI | Stage 1 结构区域检测 | 2.6M | P3/P4/P5 | 无 |
| YOLO11s-EMA-SimAM | 车载端主方案 | ~9.5M | P3/P4/P5 | P3:EMA+SimAM, P4:SimAM |
| YOLO11s-P2-EMA-SimAM | 车载端 P2 增强 | ~9.8M | P2/P3/P4/P5 | P2:SimAM, P3:EMA, P4:SimAM |
| YOLO11m-EMA-SimAM | 地面端 GPU 0 | 20.1M | P3/P4/P5 | P3:EMA, P4:SimAM, P5:ECA |
| YOLO11m-P2-SimAM | 地面端 GPU 1 | ~25M | P2/P3/P4/P5 | P2/P3/P4:SimAM, P5:ECA |

---

## 相关文档

| 文档 | 内容 |
|------|------|
| [SPECIFICATION.md](SPECIFICATION.md) | 完整规格说明书（需求/架构/API/数据/模型/测试/部署） |
| [docs/Train_guide.md](docs/Train_guide.md) | 详细训练指南（数据集准备 → 多阶段训练 → 推理部署全流程） |
| [docs/开发接口规范/](docs/开发接口规范/) | 前后端 ↔ DL 三方统一接口契约 |
| [docs/接触网缺陷类型详解.docx](docs/接触网缺陷类型详解.docx) | 16 类缺陷类型编码权威定义 |
| [config/README.md](config/README.md) | 配置使用说明与参数速查 |
| [subway_defect/deployment/defect_dict.json](subway_defect/deployment/defect_dict.json) | 机器可读缺陷字典 |

## License

AGPL-3.0 License
