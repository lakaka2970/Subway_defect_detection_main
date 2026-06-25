# Plan 3: 推理引擎与部署实现计划

> **Goal:** 实现两级推理管道、WBF 融合、TensorRT 导出和 FastAPI 推理服务，使模型可部署到车载端（单卡）和地面端（双卡）。

**Architecture:** 推理管道封装为独立 Python 模块，通过 FastAPI 暴露为 HTTP 服务。核心路径：图像 → SmartSlicer → ROI提案器 → 精细检测 → WBF融合 → 结果。TensorRT 加速作为可选后端。

---

### Task 1: 智能切片器
- `pipeline/slicer.py` — 处理 127MP 图像切片
- `tests/test_pipeline.py`

### Task 2: 两级推理管道
- `pipeline/two_stage.py` — ROI提案 + 缺陷检测

### Task 3: WBF 融合模块  
- `pipeline/wbf_fusion.py` — 地面端双卡结果融合

### Task 4: TensorRT 导出
- `deployment/export_tensorrt.py`
- `deployment/int8_calibration.py`

### Task 5: FastAPI 推理服务
- `deployment/fastapi_server.py`
- `deployment/model_warmup.py`

### Task 6: 集成测试
- 全部管道端到端测试
