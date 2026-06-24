# 地铁接触网缺陷检测系统 — 项目规格说明书

> **项目名称**: subway_defect  
> **版本**: 0.1.0  
> **许可证**: AGPL-3.0  
> **目标平台**: 福州地铁接触网超高清图像（1.27 亿像素）智能缺陷检测

---

## 目录

1. [项目概述](#1-项目概述)
2. [功能需求](#2-功能需求)
3. [非功能需求](#3-非功能需求)
4. [系统架构](#4-系统架构)
5. [组件规格](#5-组件规格)
6. [API 接口规格](#6-api-接口规格)
7. [数据规格](#7-数据规格)
8. [模型规格](#8-模型规格)
9. [训练规格](#9-训练规格)
10. [部署规格](#10-部署规格)
11. [测试需求](#11-测试需求)
12. [持续集成需求](#12-持续集成需求)
13. [术语表](#13-术语表)

---

## 1. 项目概述

### 1.1 项目背景

福州地铁通过车载高速相机采集接触网图像（127MP，约 13000×9800 像素），需要 AI 系统自动识别 18 类缺陷，替代人工巡检。

### 1.2 部署形态

| 维度 | 车载端 (vehicle) | 地面端 (ground) |
|------|-----------------|----------------|
| GPU 配置 | 单卡 RTX 4090 | 双卡 RTX 4090 |
| 操作系统 | Windows 10/11 Pro | Windows 10/11 Pro |
| 网络环境 | 完全离线（工控机） | 内网服务 |
| 部署形态 | 单 JAR 后端 + 本地 MySQL/Redis/MinIO | 微服务集群 + 分布式存储 |
| 数据传输 | — | 移动硬盘（主）/ 网络映射（备） |

### 1.3 验收指标

| 指标 | 车载端 | 地面端 |
|------|--------|--------|
| 检出率 Recall | ≥ 90% | ≥ 90% |
| 准确率 Precision | ≥ 90% | ≥ 90% |
| 单张推理耗时 | ≤ 10 秒 | ≤ 10 秒 |
| 模型加载延迟 | ≤ 1 秒 | ≤ 1 秒 |
| 提报率（误报率） | — | ≤ 5% |
| 地面端吞吐量 | — | ≥ 5 张/秒 |
| GPU 显存占用 | ≤ 16 GB | ≤ 24 GB/卡 |

### 1.4 关键技术栈

| 层级 | 技术 |
|------|------|
| AI 框架 | YOLO11 (vendored as `subway_yolo`) |
| 自定义注意力 | EMA (ICASSP 2023) + SimAM (ICML 2021) |
| 推理管道 | 两阶段级联 (ROI 提案 → 缺陷检测) + WBF 融合 |
| 模型导出 | TensorRT FP16 / INT8 |
| 推理服务 | FastAPI + Uvicorn |
| 语言 | Python ≥ 3.10 |

### 1.5 设计原则与约束

#### 接口规范遵循

本项目的 AI 模型（深度学习推理引擎）与前后端软件系统之间的接口规范定义在 `subway_defect/docs/` 目录中的设计文档内：

| 文档 | 内容 |
|------|------|
| `开发方案(5.30)/3-深度学习接口方案.md` | DL 接口整体方案设计 |
| `开发方案(5.30)/4-接口规范标准.md` | 请求/响应格式、错误码、坐标约定等详细规范 |
| `地铁接触网缺陷检测AI算法设计文档.md` | 核心算法设计与部署架构 |

> **重要**: 软件前后端（Spring Boot 后端、Vue 前端）的源代码暂时不会出现在本项目中。本项目仅包含 AI 推理引擎代码。AI 模块必须严格遵循上述接口规范文档中定义的 API 契约，以确保与外部前后端系统的正确集成。

#### 代码设计原则

在保证模型性能和适配性的前提下，项目代码遵循以下设计原则：

1. **操作简单明确** — CLI 命令语义清晰，参数命名直观；推理服务提供标准的 RESTful API，降低使用门槛
2. **参数集中配置** — 训练超参数统一集中在 `subway_defect/train/configs.py` 中管理；推理参数通过 FastAPI 启动参数或请求体传入，避免分散的硬编码
3. **代码可读性优先** — 模块职责单一，函数命名规范，关键流程添加注释；模型架构通过 YAML 声明式定义，降低修改门槛
4. **适配性优先于复杂性** — 在不牺牲 Recall/Precision 的前提下，优先选择简单的实现方案；自定义模块（EMA/SimAM）设计为最小化侵入式的 YOLO 插件
5. **配置优于硬编码** — 所有可调参数（切片尺寸、置信阈值、重叠率、增强概率等）均通过配置暴露，避免在代码深处修改魔法数字

---

## 2. 功能需求

### 2.1 核心检测功能

**FR-001: 超高清图像智能切片**
- 系统必须支持对 127MP（约 13000×9800）图像进行智能切片
- 默认切片尺寸：1024×1024（车载）/ 1280×1280（地面）
- 默认重叠率：15%；关键区域重叠率：25%
- 127MP 图像总切片数应在 150-250 范围内

**FR-002: 两阶段级联推理**
- Stage 1（ROI 提案）：在 1/8 降采样图像上检测结构区域，置信阈值 0.10-0.15
- ROI Recall 必须 ≥ 99%（宁可多报，绝不漏报）
- Stage 2（缺陷检测）：仅对 ROI 区域进行全分辨率切片检测，置信阈值 0.35-0.40
- 切片数从 ~180 降至 60-90（降低 50-67%）

**FR-003: 18 类缺陷识别**
- 刚性接触网 13 类：螺栓缺失/松动、开口销缺失、绝缘子破损等
- 柔性接触网 3 类：腕臂底座销钉缺失、吊弦不受力
- 通用缺陷 2 类：异物侵入、部件变形
- 严重等级分 4 级：minor、normal、serious、critical

**FR-004: 双卡异构 Ensemble（地面端专用）**
- GPU 0：YOLO11m-EMA-SimAM（ECA 通道选择变体，3 尺度）
- GPU 1：YOLO11m-P2-SimAM（P2 四尺度小目标特化）
- WBF 融合：IoU=0.55，双模型采信≥0.50，单模型采信≥0.75
- 融合后提报率从单模型 ~8-10% 降至 ~3-5%

**FR-005: TensorRT 加速推理**
- 支持 FP16 导出（车载端必备）
- 支持 INT8 导出（需 200-500 张校准图像）
- INT8 精度损失：mAP 下降 ≤ 2%（相对 FP16）

**FR-006: 合成数据生成**
- 通过 Inpainting 技术移除正常组件，自动生成"缺失"类缺陷样本
- 支持按类别、按数量限制生成
- 修复半径：5px，边界框扩展：3px

**FR-007: 场景自适应数据增强**
- 隧道模拟：暗化 + 暖色聚光灯 + 传感器噪声
- 烈日模拟：高对比度 + 渐变阴影条
- 运动模糊：随机方向/长度（模拟车辆振动）
- 天气模拟：60% 雾 + 40% 雨
- 接触网特化 CopyPaste 增强

### 2.2 训练功能

**FR-008: 四阶段训练管道**
- Stage A：COCO 预训练（使用官方权重）
- Stage B：ROI 提案器训练（YOLO11n，200 epochs，640 分辨率）
- Stage C1：缺陷检测 Head 预热（50 epochs，冻结 backbone）
- Stage C2：缺陷检测全量训练（200 epochs，强增强，Cosine LR）
- Stage C3：缺陷检测微调（50 epochs，弱增强，低 LR）

**FR-009: CLI 入口点**
- `train-roi`：ROI 提案器训练
- `train-defect`：缺陷检测三阶段训练
- `synthesize-defects`：合成缺陷数据生成
- `export-tensorrt`：TensorRT 模型导出
- `subway-server`：FastAPI 推理服务启动

### 2.3 推理服务功能

**FR-010: FastAPI 推理服务**
- `GET /health`：健康检查 + GPU 状态报告
- `POST /api/dl/infer`：单张图像缺陷检测
- `POST /api/dl/model/load`：模型加载/切换
- 启动时自动加载模型并执行热启动推理
- 关闭时清理 GPU 缓存

---

## 3. 非功能需求

### 3.1 性能需求

| 需求编号 | 描述 | 指标 | 适用端 |
|---------|------|------|--------|
| NFR-001 | 单张推理耗时（车载端） | ≤ 10s (含 JPEG 解码 2.5s) | 车载端 |
| NFR-001b | 单张推理耗时（地面端） | ≤ 10s | 地面端 |
| NFR-002 | 推理管道耗时分配 | Stage1~27ms + 映射~500ms + Stage2~2s + 融合~200ms | 车载端 |
| NFR-003 | 模型加载延迟 | ≤ 1s | 车载端 |
| NFR-004 | 模型加载延迟 | ≤ 1s | 地面端 |
| NFR-005 | DL 接口 P99 响应 | ≤ 15s | 车载端 |
| NFR-006 | DL 接口 P99 响应 | ≤ 2s | 地面端 |
| NFR-007 | 地面端批量吞吐 | ≥ 5 images/sec | 地面端 |
| NFR-008 | 单张推理超时 | 30s | 全部 |
| NFR-009 | GPU 显存 | ≤ 16GB | 车载端 |
| NFR-010 | GPU 显存 | ≤ 24GB/卡 | 地面端 |

### 3.2 可靠性需求

**NFR-011: 车载端断点续传**
- 每张图像处理后立即持久化结果（原子写入）
- 重启/断电后自动从检查点恢复
- 仅重处理 `analysis_status IN ('pending', 'analyzing')` 的图像

**NFR-012: 后端重试机制**
- DL 调用失败最多重试 3 次
- 重试退避策略：1000ms × 2^n（指数退避）
- 图像不可读时标记 `analysis_status='skipped'` 并继续

**NFR-013: 车载端数据保留**
- 本地数据保留 7 天
- 磁盘压力下自动淘汰最旧数据

### 3.3 兼容性需求

| 需求编号 | 描述 | 要求 |
|---------|------|------|
| NFR-014 | Python 版本 | ≥ 3.10 |
| NFR-015 | PyTorch 版本 | ≥ 2.0, CUDA 12.1 |
| NFR-016 | GPU | NVIDIA GPU, VRAM ≥ 8GB |
| NFR-017 | 坐标约定 | 统一使用归一化中心点+宽高 (YOLO 格式, 0-1) |

---

## 4. 系统架构

### 4.1 推理管道架构

```
127MP 原始图像 (13000×9800)
       │
       ▼
┌─────────────────────────┐
│ Stage 1: ROI 提案器      │  降采样 1/8 (~1625×1225)
│ 模型: YOLO11n-ROI        │  切片: 640×640 × 9 片
│ 4 类结构区域              │  耗时: ~27ms
│ Recall ≥ 99%（硬约束）    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ ROI 映射 & 去重          │  框映射回原始分辨率
│ 边缘扩展 10-15%          │  自适应重叠
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Stage 2: 缺陷检测        │  仅对 ROI 区域做 1024 切片
│ YOLO11s/m-EMA-SimAM     │  切片数: 60~90
│ 18 类缺陷                 │  耗时: ~2s
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ WBF 融合 / 全局 NMS      │  跨切片去重 + 坐标映射
└─────────────────────────┘

总耗时: 27ms + 500ms + 2000ms + 200ms + 2500ms(解码) ≈ 5.2s ✓
```

### 4.2 地面端双卡 Ensemble 架构

```
GPU 0: YOLO11m-EMA-SimAM        GPU 1: YOLO11m-P2-SimAM
(ECA 通道选择, 3 尺度)            (P2 四尺度小目标特化)
         │                              │
         └──────────┬───────────────────┘
                    ▼
          ┌─────────────────┐
          │  WBF 融合引擎    │
          │  IoU = 0.55     │
          │  双模型 ≥ 0.50   │
          │  单模型 ≥ 0.75   │
          │  最终 ≥ 0.60     │
          └─────────────────┘
```

### 4.3 模型注意力改造

| 位置 | 模块 | 参数量 | 延迟 | 论文 |
|------|------|--------|------|------|
| P3 检测分支 | EMA | ~200 | +0.4ms | ICASSP 2023 |
| P4 检测分支 | SimAM | 0 | +0.1ms | ICML 2021 |
| P5 检测分支 | SimAM / ECA | 0 / ~100 | +0.1ms | — |
| Backbone 末端 | C2PSA（保留） | — | — | YOLO11 原生 |

### 4.4 项目目录结构

```
Subway_defect_detection_main/
├── subway_defect/                    # 项目主包
│   ├── modules/                      # EMA、SimAM 注意力模块
│   ├── models/                       # 模型 YAML 配置（3 个变体）
│   ├── pipeline/                     # 推理管道（切片器、两阶段、WBF 融合）
│   ├── train/                        # 训练模块（超参数预设、CLI 脚本）
│   ├── augmentations/                # 数据增强（场景模拟、CopyPaste）
│   ├── deployment/                   # 部署（TensorRT 导出、FastAPI 服务）
│   ├── synthetic/                    # 合成数据生成（Inpainting）
│   └── docs/                         # 设计文档 + 开发方案（含前后端接口规范）
│       ├── 地铁接触网缺陷检测AI算法设计文档.md   # 核心算法设计
│       ├── plans/                        # 实现计划
│       └── 开发方案(5.30)/                # 系统开发方案（前端+后端+DL接口规范）
├── subway_yolo/                      # Vendored YOLO 框架（精简版）
│   ├── engine/                       # Model、Trainer、Predictor、Validator、Exporter
│   ├── nn/                           # tasks、modules、Extramodule（EMA/SimAM 桥接）
│   │   └── Extramodule/              # 自定义模块注册桥接
│   ├── models/yolo/                  # 仅 detect + classify
│   ├── data/                         # 数据加载、增强
│   ├── cfg/                          # 配置 + YOLO11 YAML
│   ├── utils/                        # 核心工具
│   └── optim/                        # 优化器
├── tests/                            # 测试套件（3 个测试文件）
├── scripts/                          # 部署脚本
│   └── setup_autodl.sh               # AutoDL 云平台环境配置
├── pyproject.toml                    # 项目配置 (setuptools)
├── README.md                         # 项目文档
├── SPECIFICATION.md                  # 本规格说明书
└── LICENSE                           # AGPL-3.0
```

---

## 5. 组件规格

### 5.1 EMA 注意力模块

**文件**: `subway_defect/modules/EMA.py`  
**类名**: `EMA(nn.Module)`  
**论文**: Efficient Multi-Scale Attention, ICASSP 2023

```python
EMA(channels: int, groups: int = 4, kernel_size: int = 3)
```

**架构**:
1. `GroupNorm(groups, channels)` → 输入归一化
2. X 方向 AvgPool → 1×1 Conv → Y 方向 AvgPool → 1×1 Conv → 展开融合
3. 3×3 Conv 空间精炼 → Sigmoid 门控
4. 加权乘法输出

**约束**: `channels % groups == 0`

**输入/输出**: `(B, C, H, W) → (B, C, H, W)`，形状和 dtype 不变，非原地操作

**验证要求**:
- 不同通道数（64, 128, 256, 512）下形状正确
- 不同空间尺寸（16×16 ~ 256×256）下形状正确
- 输出幅值 ≤ 输入最大绝对值 × 1.1
- 梯度非零流动
- train/eval 模式输出形状一致

### 5.2 SimAM 注意力模块

**文件**: `subway_defect/modules/SimAM.py`  
**类名**: `SimAM(nn.Module)`  
**论文**: Simple, Parameter-Free Attention Module, ICML 2021

```python
SimAM(channels: int = None, lambda_e: float = 1e-4)
```

> `channels` 参数接受但不使用（仅为 YOLO parse_model 兼容性），实际零可训练参数。

**能量函数**: `e_t = 4(σ² + λ) / ((t - μ)² + 2σ² + 2λ)`  
**注意力权重**: `1 / (1 + e_t)`，范围 (0, 1)

**输入/输出**: `(B, C, H, W) → (B, C, H, W)`

**验证要求**:
- 参数总数为 0（零参数模块）
- 不同 λ 值（1e-6 vs 1e-2）产生不同输出（atol=1e-4）
- 空间显著性：亮中心区域注意力后均值 > 暗边缘区域均值
- 梯度非零流动

### 5.3 SmartSlicer 智能切片器

**文件**: `subway_defect/pipeline/slicer.py`  
**类名**: `SmartSlicer`

```python
SmartSlicer(slice_size: int = 1024, overlap: float = 0.15, min_roi_overlap: float = 0.25)
```

**关键属性**:
- `stride = int(slice_size * (1 - overlap))`
- `n_cols = max(1, ceil((w - slice_size) / stride) + 1)`

**方法**:
| 方法 | 返回 | 说明 |
|------|------|------|
| `iter_tiles(img)` | `Iterator[(tile, row, col, x0, y0)]` | 遍历所有切片 |
| `tile_count(h, w)` | `int` | 总切片数 |
| `roi_tiles(img, roi_boxes)` | `Iterator[(tile, row, col, x0, y0)]` | 仅返回与 ROI 有交集的切片 |

**性能验证**:
- 2048×3072 图像 → 6-12 个切片
- 9800×13000 (127MP) 图像 → 150-250 个切片

### 5.4 TwoStagePipeline 两阶段推理管道

**文件**: `subway_defect/pipeline/two_stage.py`  
**类名**: `TwoStagePipeline`

```python
TwoStagePipeline(
    roi_model,              # YOLO 模型（Stage 1）
    defect_model,           # YOLO 模型（Stage 2）
    slice_size: int = 1024,
    overlap: float = 0.15,
    roi_conf: float = 0.15,
    defect_conf: float = 0.40,
    downsample_ratio: int = 8,
    device: str = "0",
)
```

**推理方法**:
```python
infer(image: np.ndarray) -> Dict
# Returns: {defects, total_time_ms, stage1_time_ms, stage2_time_ms,
#           num_roi_regions, image_size}
```

**管道流程**:
1. Stage 1: 降采样 8× → YOLO11n-ROI → 框缩放回原始坐标
2. ROI 映射: 边缘扩展 10-15%，合并相邻 ROI
3. Stage 2: 对 ROI 区域切片 → 缺陷检测 → 坐标映射回全图
4. NMS 合并: IoU=0.5，按类别内去重

### 5.5 WBFFusion 加权框融合

**文件**: `subway_defect/pipeline/wbf_fusion.py`  
**类名**: `WBFFusion`

```python
WBFFusion(
    iou_threshold: float = 0.55,
    dual_conf_threshold: float = 0.50,
    single_conf_threshold: float = 0.75,
    final_conf_threshold: float = 0.60,
    weights: tuple = (1.0, 1.0),
)
```

**裁决逻辑**:
```
双模型检测 AND avg(conf) ≥ 0.50 → 采纳
单模型检测 AND conf ≥ 0.75       → 采纳
其他                              → 拒绝
最终过滤: fused_conf ≥ 0.60       → 输出
```

**验证要求**:
- 双高置信重叠框 → ≥ 1 结果，含 `dual_detected=True`
- 单模型低置信（0.61 < 0.75） → 空结果
- 单模型高置信（0.85 ≥ 0.75） → 1 结果
- 空输入 → 空输出

### 5.6 场景增强函数

**文件**: `subway_defect/augmentations/scene.py`

所有函数签名：`(np.ndarray (H, W, 3) uint8 BGR) → np.ndarray (H, W, 3) uint8 BGR`

| 函数 | 操作 | 概率 |
|------|------|------|
| `tunnelize(img, p_brightness=0.5)` | 暗化 ×0.3-0.6 + 暖色聚光灯 + 高斯噪声 σ=3-10 | 训练期 0.3 |
| `sunlitize(img, p_shadow=0.4)` | 增亮 ×1.2-1.7 + 1-4 条渐变阴影 | 训练期 0.3 |
| `motion_blur(img)` | 线性运动模糊，核长 3-9，角度 0-360° | 训练期 0.15 |
| `weather_augment(img)` | 60% 雾（指数衰减白色叠加）+ 40% 雨（15-60 条短线） | 训练期 0.1 |

**验证要求**: 所有增强保持输入形状和 uint8 dtype，非原地操作

### 5.7 ContactNetCopyPaste

**文件**: `subway_defect/augmentations/contactnet_copy_paste.py`  
**类名**: `ContactNetCopyPaste(CopyPaste)`

```python
ContactNetCopyPaste(dataset=None, p: float = 0.6, mode: str = "flip")
```

继承 Ultralytics CopyPaste，默认概率 0.6，mode="flip"

### 5.8 合成缺陷生成

**文件**: `subway_defect/synthetic/defect_synthesis.py`

```python
generate_missing_defect(
    image_path: Path,
    label_path: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    target_class: int,
    suffix: str = "_synth_missing",
) -> Optional[Path]
```

**算法**: 选取目标类最大标注框 → 扩展 3px → 创建遮罩 → `cv2.INPAINT_TELEA` 修复（半径=5） → 保存修复后图像和过滤后标签

---

## 6. API 接口规格

### 6.1 FastAPI 推理服务端点

**服务地址**: `http://{host}:8001`

#### GET /health

健康检查 + GPU 状态。

**响应**:
```json
{
  "status": "healthy",
  "mode": "vehicle",
  "gpu": {
    "available": true,
    "gpu_count": 1,
    "gpu_0": {
      "name": "NVIDIA GeForce RTX 4090",
      "memory_used_mb": 3200,
      "memory_total_mb": 24564
    }
  }
}
```

#### POST /api/dl/infer

单张图像缺陷检测。

**请求**:
```json
{
  "image_path": "/data/images/20260624_001.jpg",
  "model_type": "vehicle",
  "confidence_threshold": 0.40,
  "slice_size": 1024,
  "slice_overlap": 0.15,
  "roi_regions": null
}
```

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `image_path` | string | 是 | — | 图像文件路径 |
| `model_type` | string | 否 | "vehicle" | "vehicle" 或 "ground" |
| `confidence_threshold` | float | 否 | 0.40 | 缺陷检测置信阈值 |
| `slice_size` | int | 否 | 1024 | 切片尺寸 |
| `slice_overlap` | float | 否 | 0.15 | 切片重叠率 |
| `roi_regions` | list[dict] | 否 | null | 预定义 ROI 区域 |

**响应**:
```json
{
  "success": true,
  "image_path": "/data/images/20260624_001.jpg",
  "processing_time_ms": 5234.5,
  "total_slices": 72,
  "num_roi_regions": 12,
  "defects": [
    {
      "defect_type": "rigid_base_nut_missing",
      "defect_name": "垂直悬吊安装底座螺母缺失",
      "confidence": 0.923,
      "box": {"x": 0.3421, "y": 0.5612, "w": 0.0123, "h": 0.0089},
      "coord_type": "normalized",
      "source_slice": {"row": 3, "col": 5}
    }
  ]
}
```

#### POST /api/dl/model/load

模型加载/切换。

**请求**:
```json
{
  "model_type": "vehicle",
  "model_version": "yolov11-s-v1.2",
  "force_reload": false
}
```

**响应**:
```json
{
  "success": true,
  "model_type": "vehicle",
  "model_version": "yolov11-s-v1.2",
  "load_time_ms": 850,
  "message": "Model loaded successfully"
}
```

### 6.2 DL 错误码

| 错误码 | HTTP 状态码 | 后端处理策略 |
|--------|-----------|-------------|
| `DL_MODEL_NOT_LOADED` | 503 | 自动调用 /model/load，最多重试 2 次 |
| `DL_GPU_OOM` | 507 | 减小 batch/slice 尺寸后重试 |
| `DL_IMAGE_UNREADABLE` | 400 | 标记 analysis_status='skipped'，继续 |
| `DL_INFERENCE_TIMEOUT` | 504 | 记录错误，跳过（30s 超时） |
| `DL_INTERNAL_ERROR` | 500 | 记录日志，重试 1 次 |

### 6.3 模型版本命名规范

格式: `{base_model}-{scale}-v{major}.{minor}`

示例: `yolov11-s-v1.2`, `yolov11-m-v2.0`

---

## 7. 数据规格

### 7.1 缺陷类别编码（18 类）

#### 刚性接触网缺陷（13 类）

| 编码 | 中文名称 | 严重等级 | 类权重 |
|------|---------|---------|--------|
| `rigid_base_nut_missing` | 垂直悬吊安装底座螺母缺失 | serious | 1.0 |
| `rigid_base_nut_loose` | 垂直悬吊安装底座螺母松动 | serious | 1.0 |
| `rigid_single_bracket_base_nut_missing` | 单支垂直悬吊槽钢底座螺母缺失 | serious | 1.15 |
| `rigid_single_bracket_base_nut_loose` | 单支垂直悬吊槽钢底座螺母松动 | serious | 1.0 |
| `rigid_single_bracket_upper_nut_loose` | 单支垂直悬吊槽钢上方螺母松动 | normal | 1.0 |
| `rigid_hanger_top_plate_nut_missing` | 刚性悬挂吊柱顶板底面螺母缺失 | serious | 1.0 |
| `rigid_hanger_top_plate_nut_loose` | 刚性悬挂吊柱顶板底面螺母松动 | serious | 1.0 |
| `rigid_ground_wire_clamp_nut_missing` | 地线线夹托板安装底座螺母缺失 | serious | 1.0 |
| `rigid_ground_wire_clamp_nut_loose` | 地线线夹托板安装底座螺母松动 | serious | 1.0 |
| `rigid_ground_wire_nut_missing` | 地线线夹螺母缺失 | serious | 1.2 |
| `rigid_ground_wire_nut_loose` | 地线线夹螺母松动 | serious | 1.2 |
| `rigid_busbar_joint_bolt_missing` | 汇流排中间接头螺栓缺失 | **critical** | 1.0 |
| `rigid_insulator_damage` | 绝缘子破损 | **critical** | 1.5 |

#### 柔性接触网缺陷（3 类）

| 编码 | 中文名称 | 严重等级 | 类权重 |
|------|---------|---------|--------|
| `flex_wrist_base_hori_pin_missing` | 腕臂底座横向销钉缺开口销 | serious | 1.0 |
| `flex_wrist_base_vert_pin_missing` | 腕臂底座垂直销钉缺开口销 | serious | 1.0 |
| `flex_dropper_no_force` | 吊弦不受力 | serious | 1.3 |

#### 通用缺陷（2 类）

| 编码 | 中文名称 | 严重等级 | 类权重 |
|------|---------|---------|--------|
| `foreign_object` | 异物侵入 | normal | 1.5 |
| `component_deformation` | 部件变形 | normal | 1.0 |

### 7.2 坐标约定

**统一格式**: 归一化中心点 + 宽高 (YOLO 格式)，值范围 [0, 1]

```json
{
  "box": {"x": 0.3421, "y": 0.5612, "w": 0.0123, "h": 0.0089},
  "coord_type": "normalized"
}
```

前端转换公式: `pixel_x = x * image_width`, `pixel_y = y * image_height`

### 7.3 严重等级定义

| 等级 | 英文 | 说明 |
|------|------|------|
| minor | minor | 轻微异常，不影响运行 |
| normal | normal | 一般缺陷，需关注 |
| serious | serious | 严重缺陷，影响安全 |
| critical | critical | 危急缺陷，立即停运 |

### 7.4 数据集目录结构

```
datasets/
├── roi/
│   └── roi_data.yaml              # ROI 提案器数据集配置
├── defects/
│   └── defect_data.yaml           # 缺陷检测数据集配置
├── calibration/                   # INT8 校准数据集（200-500 张）
├── synthetic/                     # 合成缺陷输出
└── latest/ -> v2.0/               # 最新稳定版本符号链接
```

---

## 8. 模型规格

### 8.1 模型变体总览

| 模型 | 用途 | 参数 | GFLOPs | 检测尺度 | 注意力 |
|------|------|------|--------|---------|--------|
| YOLO11n-ROI | 车载/地面 Stage 1 | 2.6M | 6.6 | P3/P4/P5 | 无 |
| YOLO11s-EMA-SimAM | 车载端主方案 | 9.5M | 21.7 | P3/P4/P5 | P3:EMA, P4:SimAM, P5:SimAM |
| YOLO11m-EMA-SimAM | 地面端 GPU 0 | 20.1M | 68.5 | P3/P4/P5 | P3:EMA, P4:SimAM, P5:ECA |
| YOLO11m-P2-SimAM | 地面端 GPU 1 | ~25M | ~90 | P2/P3/P4/P5 | P2:SimAM, P3:SimAM, P4:SimAM, P5:ECA |

### 8.2 YOLO11 基础架构

**Backbone**: P1(64ch) → P2(128ch) → P3(256ch) → P4(512ch) → P5(1024ch) → C2PSA

**Head (标准)**: FPN 上采样 + PAN 下采样，P3/P4/P5 三尺度检测

**缩放参数**:
| 规模 | depth | width | max_channels | 参数量 | GFLOPs |
|------|-------|-------|-------------|--------|--------|
| n | 0.50 | 0.25 | 1024 | 2.6M | 6.6 |
| s | 0.50 | 0.50 | 1024 | 9.5M | 21.7 |
| m | 0.50 | 1.00 | 512 | 20.1M | 68.5 |

### 8.3 自定义模块注册机制

`subway_yolo/nn/Extramodule/__init__.py` 导出 `CBAM`, `ECA`, `EMA`, `SimAM` 四个类。
- `CBAM`: 从 `subway_yolo.nn.modules.conv` 重新导出
- `ECA`: 本地定义（通道注意力，AdaptiveAvgPool2d + Conv1d + Sigmoid）
- `EMA`, `SimAM`: 从 `subway_defect.modules` 导入

`subway_yolo/nn/tasks.py` 通过 `from .Extramodule import *` 使这些模块在 `parse_model()` 的 `globals()` 中可用，从而实现 YAML 配置中的模块名解析。

### 8.4 损失函数

**基础 YOLO 损失**: `L = λ_box × CIoU + λ_cls × BCE + λ_dfl × DFL`

**接触网小目标改造**:
```
L_total = 1.5 × WIoU(box) + 0.5 × FocalLoss(cls, α=0.25, γ=2.0) + 1.5 × DFL(dfe)
```
- CIoU → Wise-IoU：小框梯度更稳定，IoU=0 时仍有非零梯度
- BCE → Focal Loss：处理极端正负样本不平衡（2-3 个缺陷 vs 数千个负样本）
- 框回归权重 1.5×（定位精度优先）
- 分类权重 0.5×（由 Focal Loss 平衡）

---

## 9. 训练规格

### 9.1 Stage B: ROI 提案器训练

| 参数 | 值 |
|------|-----|
| 模型 | YOLO11n |
| 类别数 | 4-5（结构区域类） |
| Epochs | 200 |
| 图像尺寸 | 640 |
| Batch size | 32 |
| 优化器 | SGD |
| 学习率 | 0.01 |
| LR 调度 | Cosine |
| Mosaic | 0.8 |
| MixUp | 0.1 |
| CopyPaste | 0（关闭） |
| 目标 | ROI Recall ≥ 99% |

### 9.2 Stage C1: 缺陷检测 Head 预热

| 参数 | 值 |
|------|-----|
| 模型 | YOLO11s-EMA-SimAM 或 YOLO11m-EMA-SimAM |
| Epochs | 50 |
| 图像尺寸 | 1024 |
| Batch size | 16 |
| 优化器 | SGD |
| 学习率 | 0.001（恒定） |
| 冻结层 | model.0 ~ model.10（backbone 全部冻结） |
| Mosaic | 0.5 |
| MixUp | 0（关闭） |
| CopyPaste | 0（关闭） |

### 9.3 Stage C2: 缺陷检测全量训练

| 参数 | 值 |
|------|-----|
| Epochs | 200 |
| 图像尺寸 | 1024 |
| Batch size | 16 |
| 优化器 | AdamW |
| 学习率 | 0.001 → 0.00001（Cosine） |
| Mosaic | 0.8 → 0.1（epoch 200 时关闭） |
| MixUp | 0.15 |
| CopyPaste | 0.6 (mode="flip") |
| 场景增强 | tunnel 0.3, sunlight 0.3, weather 0.1, motion_blur 0.15 |

### 9.4 Stage C3: 缺陷检测微调

| 参数 | 值 |
|------|-----|
| Epochs | 50 |
| 图像尺寸 | 1024 |
| Batch size | 8 |
| 优化器 | AdamW |
| 学习率 | 0.0001（恒定） |
| Mosaic/MixUp | 0（关闭） |
| CopyPaste | 0.4 |
| 几何增强 | degrees=2.0, translate=0.1, scale=0.3, shear=1.0 |

### 9.5 数据增强体系

```
第四层: 合成生成 — Inpainting 缺失合成 + CG 渲染罕见缺陷  (离线, 50-80 张/类)
第三层: 混合增强 — CopyPaste + Mosaic9                    (在线, 中后期)
第二层: 定制增强 — 隧道/烈日/运动模糊/雨雾                 (在线, 全程)
第一层: 基础扩增 — 翻转/缩放/HSV 偏移                     (在线, 全程)
```

**重要约束**: `flipud=0.0`（接触网有固定上下方向，禁止垂直翻转）

---

## 10. 部署规格

### 10.1 环境要求

| 组件 | 要求 |
|------|------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.0, CUDA 12.1 |
| GPU | NVIDIA GPU, VRAM ≥ 8 GB（推荐 RTX 4090） |
| OS（车载/地面） | Windows 10/11 Pro |
| OS（训练） | Linux |

### 10.2 依赖清单

核心依赖（`pyproject.toml`）:
```
torch>=2.0.0, torchvision>=0.15.0, numpy>=1.23.0, opencv-python>=4.8.0,
pillow>=9.4.0, pyyaml>=6.0, requests>=2.34.2, scipy>=1.10.0,
matplotlib>=3.7.0, pandas>=1.5.0, seaborn>=0.12.0, tqdm>=4.64.0,
psutil>=5.9.0, py-cpuinfo>=8.0.0, fastapi>=0.104.0, uvicorn>=0.24.0,
pydantic>=2.5.0
```

### 10.3 CLI 入口点

| 命令 | 模块 | 说明 |
|------|------|------|
| `train-defect` | `subway_defect.train.train_defect:main` | 缺陷检测三阶段训练 |
| `train-roi` | `subway_defect.train.train_roi:main` | ROI 提案器训练 |
| `synthesize-defects` | `subway_defect.synthetic.defect_synthesis:main` | 合成缺陷数据生成 |
| `export-tensorrt` | `subway_defect.deployment.export_tensorrt:main` | TensorRT 导出 |
| `subway-server` | `subway_defect.deployment.fastapi_server:main` | 推理服务启动 |

### 10.4 TensorRT 导出参数

```bash
# FP16（车载端推荐）
export-tensorrt --model best.pt --fp16 --imgsz 1024 --workspace 4

# INT8（需校准数据集）
export-tensorrt --model best.pt --int8 --calibration_data datasets/calibration/ --imgsz 1024
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | .pt 模型路径 |
| `--fp16` / `--int8` | — | 精度模式 |
| `--calibration_data` | `datasets/calibration/` | INT8 校准集（200-500 张） |
| `--imgsz` | 1024 | 导出图像尺寸 |
| `--workspace` | 4 | TensorRT workspace (GB) |
| `--output` | 自动 | 输出 .engine 路径 |

### 10.5 推理服务启动

```bash
# 车载端（单模型）
subway-server --port 8001 --model best.pt --mode vehicle

# 地面端（双 GPU + WBF）
subway-server --port 8001 --model best.pt --model_b best_p2.pt --mode ground
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 8001 | 服务端口 |
| `--host` | 0.0.0.0 | 绑定地址 |
| `--model` | 必填 | 缺陷检测模型 .pt |
| `--roi_model` | yolo11n.pt | ROI 提案器 |
| `--model_b` | None | 地面端第二模型 |
| `--mode` | vehicle | vehicle / ground |
| `--slice_size` | 1024 | 切片尺寸 |
| `--overlap` | 0.15 | 切片重叠率 |
| `--roi_conf` | 0.15 | ROI 置信阈值 |
| `--defect_conf` | 0.40 | 缺陷置信阈值 |

### 10.6 车载端断点续传

- 每张图像处理完立即持久化（原子写入）
- `t_raw_data.analysis_status` 作为自然检查点
- 重启后仅处理 `analysis_status IN ('pending', 'analyzing')` 的图像
- 无回滚语义；通过单条缺陷状态转换修正

---

## 11. 测试需求

### 11.1 测试配置

- 测试框架: pytest ≥ 7.4.0
- 测试路径: `tests/`
- 文件匹配: `test_*.py`
- 函数匹配: `test_*`
- 运行选项: `-v --tb=short -p no:asyncio`
- `--slow` 标志控制慢速集成测试

### 11.2 测试覆盖要求

| 测试文件 | 覆盖组件 | 测试数 |
|---------|---------|--------|
| `test_attention_modules.py` | EMA、SimAM 单元测试 + 模型集成测试 | 19 |
| `test_augmentations.py` | 场景增强 + CopyPaste + 训练配置 + CLI 脚本 | 10 |
| `test_pipeline.py` | SmartSlicer + WBFFusion + 部署导入 + 管道集成 | 11 |

### 11.3 关键阈值验证

| 组件 | 验证项 | 阈值 |
|------|--------|------|
| EMA | 输出幅值上界 | ≤ 输入最大绝对值 × 1.1 |
| SimAM | 参数量 | = 0（零参数） |
| SimAM | λ 敏感度 | λ=1e-6 vs 1e-2 输出不同 (atol=1e-4) |
| SmartSlicer | 2048×3072 切片数 | [6, 12] |
| SmartSlicer | 9800×13000 切片数 | [150, 250] |
| WBFFusion | 单模型低置信过滤 | conf=0.61 → 空结果 |
| WBFFusion | 单模型高置信通过 | conf=0.85 → 1 结果 |
| 模型构建 | YAML → YOLO 成功 | 3 个变体全部通过 |

---

## 12. 持续集成需求

### 12.1 持续训练管道

```
数据采集(工程车) → 主动学习筛选 → 人工标注 → 数据集版本管理 → 自动训练触发 → 模型注册 & 部署
```

**触发条件**（满足任一）:
- 新增标注数据 ≥ 200 张
- 新增缺陷类别
- 线上 Precision/Recall 低于验收线
- 定期重训练（每季度）

### 12.2 模型部署准入条件

必须**全部**满足:
1. Recall ≥ 当前线上版本
2. Precision ≥ 当前线上版本
3. 无类别退化：任何类 Recall 下降 ≤ 5%
4. 推理耗时 ≤ 当前线上版本

### 12.3 主动学习策略

1. 不确定性采样：推理未标注图像，选取 `conf ∈ [0.3, 0.6]` 的样本
2. 多样性采样：对不确定样本做特征聚类，每簇均匀采样
3. 类别平衡：优先补充稀疏类别，目标 ≥ 100 实例/类
4. 难例挖掘：自动收集模型与人工审核意见不一致的样本

### 12.4 数据集版本管理

```
datasets/
├── v1.0/   (500 张, 初始版本)
├── v1.1/   (+200 张, 开口销类别补充)
├── v1.2/   (+350 张, 隧道段补充)
├── v2.0/   (+500 张, 全类别覆盖)
└── latest/ -> v2.0/
```

每个版本记录: 日期、图像数、类别数、类别分布、验证指标 (mAP50, Recall, Precision)

---

## 13. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| 接触网 | Catenary / Overhead Contact Line | 为电力机车供电的架空线路系统 |
| 车载端 | Vehicle-side / Onboard | 安装在工程车上的边缘推理系统 |
| 地面端 | Ground-side / Server | 部署在数据中心的集中分析系统 |
| ROI | Region of Interest | 感兴趣区域（结构区域检测） |
| WBF | Weighted Boxes Fusion | 加权框融合算法 |
| 提报率 | False-report Rate | 误报率（AI 提报但人工审核确认为误报的缺陷占比） |
| EMA | Efficient Multi-Scale Attention | 高效多尺度注意力（ICASSP 2023） |
| SimAM | Simple Parameter-Free Attention Module | 无参数注意力模块（ICML 2021） |
| ECA | Efficient Channel Attention | 高效通道注意力 |
| C2PSA | Cross-Stage Partial with Spatial Attention | YOLO11 原生位置敏感注意力 |
| SmartSlicer | — | 智能切片器（重叠瓦片式大图切片） |
| 降采样 | Downsample | 降低图像分辨率（如 1/8 降采样） |
| NMS | Non-Maximum Suppression | 非极大值抑制（去重算法） |
| FP16 | Half Precision | 半精度浮点（16-bit）推理 |
| INT8 | 8-bit Integer | 8 位整数量化推理 |
| mAP | mean Average Precision | 平均精度均值 |
| DZI | Deep Zoom Image | 深度缩放图像（多分辨率瓦片金字塔） |

---

> **文档版本**: 1.0  
> **最后更新**: 2026-06-24  
> **维护团队**: Subway Defect Detection Team
