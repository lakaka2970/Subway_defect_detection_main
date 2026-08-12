#!/usr/bin/env python3
"""
FastAPI inference server for subway catenary defect detection.

Supports onboard (single model) and ground (dual ensemble) modes.

Endpoints:
    GET  /api/dl/health        — health check with loaded models + GPU status
    POST /api/dl/infer         — single image inference
    POST /api/dl/infer/batch   — batch inference
    POST /api/dl/model/load    — load/reload a model

Usage:
    # Onboard (single model)
    python -m subway_defect.deployment.fastapi_server --port 8001 \
        --model src/weights/stage5_best.pt --mode onboard

    # Ground (dual GPU WBF fusion)
    python -m subway_defect.deployment.fastapi_server --port 8001 \
        --model src/weights/stage5_best.pt \
        --model_b <gpu1_model> --mode ground
"""

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from subway_yolo import YOLO

from subway_defect.pipeline.two_stage import TwoStagePipeline
from subway_defect.pipeline.wbf_fusion import WBFFusion


# ═══════════════════════════════════════════════════════════════════════════
# Error codes (per interface specification §7)
# ═══════════════════════════════════════════════════════════════════════════

_ERROR_CODES = {
    "DL_MODEL_NOT_LOADED": (503, "模型未加载"),
    "DL_GPU_OOM": (507, "GPU 显存不足"),
    "DL_IMAGE_UNREADABLE": (400, "图像损坏/无法读取"),
    "DL_INFERENCE_TIMEOUT": (504, "推理超时（> 30s）"),
    "DL_INTERNAL_ERROR": (500, "内部异常"),
}


def dl_error_response(code: str, detail: str = "") -> JSONResponse:
    """Return a standardized error response."""
    http_status, default_msg = _ERROR_CODES.get(
        code, (500, "未知错误"))
    return JSONResponse(
        status_code=http_status,
        content={
            "success": False,
            "errorCode": code,
            "message": detail or default_msg,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Defect name resolution
# ═══════════════════════════════════════════════════════════════════════════

def _load_defect_dict() -> Dict[str, str]:
    """Load Chinese defect name mapping from defect_dict.json."""
    try:
        dict_path = Path(__file__).parent / "defect_dict.json"
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {d["code"]: d["name_cn"] for d in data.get("defects", [])}
    except Exception:
        pass
    return {}


def _resolve_defect_name(code: str) -> str:
    """Resolve a defect class code to its Chinese name.

    Checks defect_dict.json first, falls back to classes.py CN_NAME_MAP.
    """
    cn = _DEFECT_DICT.get(code)
    if cn:
        return cn
    try:
        from subway_defect.classes import CN_NAME_MAP
        return CN_NAME_MAP.get(code, code)
    except Exception:
        return code


_DEFECT_DICT: Dict[str, str] = _load_defect_dict()

# ═══════════════════════════════════════════════════════════════════════════
# Pydantic models (camelCase aliases per interface specification)
# ═══════════════════════════════════════════════════════════════════════════


class BoxCoords(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    x: float
    y: float
    w: float
    h: float


class SourceSlice(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    row: int = Field(alias="row")
    col: int = Field(alias="col")


class DefectResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    defectType: str = Field(alias="defectType")
    defectName: str = Field(alias="defectName")
    confidence: float
    box: BoxCoords
    coordType: str = Field(default="normalized", alias="coordType")
    sourceSlice: Optional[SourceSlice] = Field(
        default=None, alias="sourceSlice")


class InferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    imagePath: str = Field(alias="imagePath")
    modelType: str = Field(default="onboard", alias="modelType")
    confidenceThreshold: float = Field(
        default=0.40, alias="confidenceThreshold")
    outputCoordType: str = Field(
        default="normalized", alias="outputCoordType")
    extraParams: Optional[Dict[str, Any]] = Field(
        default=None, alias="extraParams")

    @classmethod
    def model_validate(cls, data: Any, **kwargs):
        """Flatten extraParams into top-level fields before validation."""
        if isinstance(data, dict) and "extraParams" in data:
            ep = data.pop("extraParams") or {}
            if isinstance(ep, dict):
                for k, v in ep.items():
                    if k not in data:
                        data[k] = v
        return super().model_validate(data, **kwargs)


class InferResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    imagePath: str = Field(alias="imagePath")
    processingTimeMs: float = Field(alias="processingTimeMs")
    totalSlices: int = Field(default=0, alias="totalSlices")
    defects: List[DefectResult]
    coordType: str = Field(default="normalized", alias="coordType")


class BatchInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    imagePaths: List[str] = Field(alias="imagePaths")
    modelType: str = Field(default="onboard", alias="modelType")
    confidenceThreshold: float = Field(
        default=0.40, alias="confidenceThreshold")
    outputCoordType: str = Field(
        default="normalized", alias="outputCoordType")
    extraParams: Optional[Dict[str, Any]] = Field(
        default=None, alias="extraParams")

    @classmethod
    def model_validate(cls, data: Any, **kwargs):
        """Flatten extraParams into top-level fields before validation."""
        if isinstance(data, dict) and "extraParams" in data:
            ep = data.pop("extraParams") or {}
            if isinstance(ep, dict):
                for k, v in ep.items():
                    if k not in data:
                        data[k] = v
        return super().model_validate(data, **kwargs)


class BatchInferResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    results: List[InferResponse]


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    modelType: str = Field(default="onboard", alias="modelType")
    modelVersion: str = Field(default="latest", alias="modelVersion")
    forceReload: bool = Field(default=False, alias="forceReload")


class ModelLoadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    modelType: str = Field(alias="modelType")
    modelVersion: str = Field(alias="modelVersion")
    loadTimeMs: float = Field(alias="loadTimeMs")
    message: str


# ═══════════════════════════════════════════════════════════════════════════
# Application state
# ═══════════════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self):
        self.pipeline: Optional[TwoStagePipeline] = None
        self.pipeline_b: Optional[TwoStagePipeline] = None
        self.fusion: Optional[WBFFusion] = None
        self.mode: str = "onboard"
        self.model_loaded_at: float = 0.0
        self.model_version: str = "7.25"
        self.roi_model = None
        self.defect_model = None
        self.defect_model_b = None


state = AppState()


def _normalize_model_type(mt: str) -> str:
    """Normalize model type — accept 'vehicle' as legacy alias for 'onboard'."""
    if mt == "vehicle":
        return "onboard"
    if mt in ("onboard", "ground"):
        return mt
    return "onboard"


def _resolve_roi_path(config: dict) -> str:
    """Resolve ROI model path — try configured path, then fall back to yolo11n.pt."""
    roi_path = config.get("roi_model")
    if roi_path and os.path.isfile(roi_path):
        return roi_path
    # Search in common locations
    candidates = [
        "yolo11n.pt",
        "src/weights/yolo11n.pt",
        "weights/yolo11n.pt",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return roi_path or "yolo11n.pt"


def load_models(config: dict):
    """Load models with warmup inference for fast first prediction."""
    t0 = time.time()

    roi_path = _resolve_roi_path(config)
    print(f"[load] ROI model: {roi_path}")
    state.roi_model = YOLO(roi_path)
    warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    state.roi_model(warmup, verbose=False)

    defect_path = config.get("defect_model")
    print(f"[load] Defect model: {defect_path}")
    state.defect_model = YOLO(defect_path)
    state.defect_model(warmup, verbose=False)

    state.pipeline = TwoStagePipeline(
        roi_model=state.roi_model,
        defect_model=state.defect_model,
        slice_size=config.get("slice_size", 1024),
        overlap=config.get("overlap", 0.15),
        roi_conf=config.get("roi_conf", 0.15),
        defect_conf=config.get("defect_conf", 0.40),
    )

    state.mode = _normalize_model_type(config.get("mode", "onboard"))

    if state.mode == "ground" and config.get("defect_model_b"):
        state.defect_model_b = YOLO(config["defect_model_b"])
        state.defect_model_b(warmup, verbose=False)
        state.pipeline_b = TwoStagePipeline(
            roi_model=state.roi_model,
            defect_model=state.defect_model_b,
            slice_size=config.get("slice_size", 1024),
            overlap=config.get("overlap", 0.15),
            roi_conf=config.get("roi_conf", 0.15),
            defect_conf=config.get("defect_conf", 0.40),
            downsample_ratio=config.get("downsample_ratio", 8),
            device=config.get("device", "0"),
        )
        state.fusion = WBFFusion(
            iou_threshold=config.get("wbf_iou", 0.55),
            dual_conf_threshold=config.get("wbf_dual_conf", 0.50),
            single_conf_threshold=config.get("wbf_single_conf", 0.75),
            final_conf_threshold=config.get("wbf_final_conf", 0.60),
        )

    state.model_loaded_at = time.time()
    load_ms = (state.model_loaded_at - t0) * 1000
    return load_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup GPU memory."""
    config = app.state.config
    load_ms = load_models(config)
    print(f"Models loaded in {load_ms:.0f}ms (mode={state.mode})")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {props.name}, "
                  f"memory {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB")

    yield
    state.pipeline = None
    state.roi_model = None
    state.defect_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Subway Defect Detection API", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

# -- Legacy health redirect --
@app.get("/health")
async def health_legacy():
    """Legacy health check — redirects to /api/dl/health."""
    return await health()


# -- Standard health check --
@app.get("/api/dl/health")
async def health():
    gpu_info = {"available": torch.cuda.is_available()}
    if gpu_info["available"]:
        for i in range(torch.cuda.device_count()):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memoryUsedMb": (mem_total - mem_free) // (1024 * 1024),
                "memoryTotalMb": mem_total // (1024 * 1024),
            }

    loaded_models = []
    if state.pipeline:
        loaded_models.append({
            "modelType": state.mode,
            "version": state.model_version,
            "loadedAt": state.model_loaded_at,
        })
        if state.mode == "ground" and state.pipeline_b:
            loaded_models.append({
                "modelType": "ground_aux",
                "version": state.model_version,
                "loadedAt": state.model_loaded_at,
            })

    return {
        "status": "healthy" if state.pipeline else "loading",
        "loadedModels": loaded_models,
        "gpu": gpu_info,
    }


# -- Model load --
@app.post("/api/dl/model/load")
async def load_model(req: ModelLoadRequest):
    try:
        config = app.state.config
        config["mode"] = _normalize_model_type(req.modelType)
        load_ms = load_models(config)
        return ModelLoadResponse(
            success=True,
            modelType=req.modelType,
            modelVersion=req.modelVersion,
            loadTimeMs=load_ms,
            message="Model loaded successfully",
        )
    except Exception as e:
        return ModelLoadResponse(
            success=False,
            modelType=req.modelType,
            modelVersion=req.modelVersion,
            loadTimeMs=0,
            message=str(e),
        )


# -- Single image inference --
@app.post("/api/dl/infer")
async def infer(req: InferRequest):
    if state.pipeline is None:
        return dl_error_response("DL_MODEL_NOT_LOADED")

    try:
        image_path = Path(req.imagePath).resolve()
        if not image_path.is_file():
            return dl_error_response(
                "DL_IMAGE_UNREADABLE",
                f"Image not found: {req.imagePath}")

        img = cv2.imread(str(image_path))
        if img is None:
            return dl_error_response(
                "DL_IMAGE_UNREADABLE",
                f"Cannot read image: {req.imagePath}")

        # Extract optional params from extraParams (already flattened)
        slice_size = getattr(req, "sliceSize", 1024)
        slice_overlap = getattr(req, "sliceOverlap", 0.15)

        state.pipeline.defect_conf = req.confidenceThreshold
        state.pipeline.slicer.slice_size = slice_size
        state.pipeline.slicer.overlap = slice_overlap
        state.pipeline.slicer._stride = int(
            slice_size * (1 - slice_overlap))

        result_a = state.pipeline.infer(img)

        fusion_start = time.time()
        if state.mode == "ground" and state.pipeline_b:
            state.pipeline_b.defect_conf = req.confidenceThreshold
            state.pipeline_b.slicer.slice_size = slice_size
            state.pipeline_b.slicer.overlap = slice_overlap
            state.pipeline_b.slicer._stride = int(
                slice_size * (1 - slice_overlap))
            result_b = state.pipeline_b.infer(img)
            fused_defects = state.fusion.fuse(
                result_a["defects"], result_b["defects"])
            total_ms = max(
                result_a["total_time_ms"], result_b["total_time_ms"])
            total_ms += (time.time() - fusion_start) * 1000
        else:
            fused_defects = result_a["defects"]
            total_ms = result_a["total_time_ms"]

        coord_type = req.outputCoordType

        defects = []
        for d in fused_defects:
            class_name = d.get("class_name", "")
            defects.append(DefectResult(
                defectType=class_name,
                defectName=_resolve_defect_name(class_name),
                confidence=d["confidence"],
                box=BoxCoords(**d["box"]),
                coordType=coord_type,
                sourceSlice=SourceSlice(
                    **d["source_tile"]) if d.get("source_tile") else None,
            ))

        return InferResponse(
            success=True,
            imagePath=req.imagePath,
            processingTimeMs=total_ms,
            totalSlices=result_a.get("total_slices", 0),
            defects=defects,
            coordType=coord_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        return dl_error_response("DL_INTERNAL_ERROR", str(e))


# -- Batch inference --
@app.post("/api/dl/infer/batch")
async def infer_batch(req: BatchInferRequest):
    if state.pipeline is None:
        return dl_error_response("DL_MODEL_NOT_LOADED")

    results = []
    for image_path_str in req.imagePaths:
        single_req = InferRequest(
            imagePath=image_path_str,
            modelType=req.modelType,
            confidenceThreshold=req.confidenceThreshold,
            outputCoordType=req.outputCoordType,
        )
        try:
            r = await infer(single_req)
            if isinstance(r, JSONResponse):
                # Error response — create a failed result
                error_body = json.loads(r.body)
                results.append(InferResponse(
                    success=False,
                    imagePath=image_path_str,
                    processingTimeMs=0,
                    totalSlices=0,
                    defects=[],
                    coordType=req.outputCoordType,
                ))
            else:
                results.append(r)
        except Exception:
            results.append(InferResponse(
                success=False,
                imagePath=image_path_str,
                processingTimeMs=0,
                totalSlices=0,
                defects=[],
                coordType=req.outputCoordType,
            ))

    return BatchInferResponse(success=True, results=results)


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Entry point for the inference server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Inference server")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--model", required=True)
    parser.add_argument("--roi_model", default="yolo11n.pt")
    parser.add_argument("--model_b", default=None)
    parser.add_argument("--mode", default="onboard",
                        choices=["onboard", "ground", "vehicle"])
    parser.add_argument("--slice_size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--roi_conf", type=float, default=0.15)
    parser.add_argument("--defect_conf", type=float, default=0.40)
    args = parser.parse_args()

    app.state.config = {
        "defect_model": args.model,
        "roi_model": args.roi_model,
        "defect_model_b": args.model_b,
        "mode": args.mode,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "roi_conf": args.roi_conf,
        "defect_conf": args.defect_conf,
    }

    print(f"Starting {args.mode} inference server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
