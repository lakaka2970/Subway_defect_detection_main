#!/usr/bin/env python3
"""
FastAPI inference server for subway catenary defect detection.

Supports onboard-side (single model) and ground-side (dual ensemble) modes.

Endpoints (per 接口规范标准 §4):
    GET  /api/dl/health              — health check with GPU status
    POST /api/dl/infer               — single image inference
    POST /api/dl/infer/batch         — batch inference (ground-side)
    POST /api/dl/model/load          — load/reload a model

Request/response fields use **camelCase** naming (aligned with Java backend).
All defect codes and Chinese names follow the authoritative defect_dict.json.

Usage::

    # Onboard-side
    python -m subway_defect.deployment.fastapi_server --port 8001 \\
        --model runs/defect_detector_c2_full/weights/best.pt \\
        --mode onboard

    # Ground-side (dual GPU)
    python -m subway_defect.deployment.fastapi_server --port 8001 \\
        --model runs/defect_detector_c2_full/weights/best.pt \\
        --model_b runs/defect_detector_p2/weights/best.pt \\
        --mode ground
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

from subway_defect.classes import (
    CN_NAME_MAP,
    SEVERITY_MAP,
    resolve_canonical,
)
from subway_defect.pipeline.two_stage import TwoStagePipeline
from subway_defect.pipeline.wbf_fusion import WBFFusion


# ═══════════════════════════════════════════════════════════════════════════
# Standardized DL error codes (per 接口规范标准 §4.4)
# ═══════════════════════════════════════════════════════════════════════════

DL_ERROR_CODES = {
    "DL_MODEL_NOT_LOADED":    {"http": 503, "message": "模型未加载，请先调用 /api/dl/model/load"},
    "DL_GPU_OOM":             {"http": 507, "message": "GPU 显存不足"},
    "DL_IMAGE_UNREADABLE":    {"http": 400, "message": "图像损坏或无法读取"},
    "DL_INFERENCE_TIMEOUT":   {"http": 504, "message": "推理超时（单张 > 30s）"},
    "DL_INTERNAL_ERROR":      {"http": 500, "message": "推理引擎内部异常"},
}


def dl_error_response(code: str, detail: str = "", suggestion: str = "") -> JSONResponse:
    """Build a standardized DL error response."""
    info = DL_ERROR_CODES.get(code, {"http": 500, "message": "Unknown error"})
    return JSONResponse(
        status_code=info["http"],
        content={
            "success": False,
            "errorCode": code,
            "message": f"{info['message']}. {detail}" if detail else info["message"],
            "suggestion": suggestion,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic request/response models — camelCase aliases per 接口规范标准
# ═══════════════════════════════════════════════════════════════════════════

class BoxCoords(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    x: float
    y: float
    w: float
    h: float


class SourceSlice(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    row: int
    col: int


class DefectResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    defect_type: str = Field(..., alias="defectType")
    defect_name: str = Field(..., alias="defectName")
    confidence: float
    box: BoxCoords
    coord_type: str = Field("normalized", alias="coordType")
    source_slice: Optional[SourceSlice] = Field(None, alias="sourceSlice")


class InferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image_path: str = Field(..., alias="imagePath")
    model_type: str = Field("onboard", alias="modelType")
    confidence_threshold: float = Field(0.40, alias="confidenceThreshold")
    output_coord_type: str = Field("normalized", alias="outputCoordType")
    slice_size: int = Field(1024, alias="extraParams.sliceSize",  # flattened
                            validation_alias="sliceSize")
    slice_overlap: float = Field(0.15, alias="extraParams.sliceOverlap",
                                 validation_alias="sliceOverlap")
    roi_regions: Optional[List[Dict[str, float]]] = Field(
        None, alias="extraParams.highResRegions", validation_alias="roiRegions")

    # Also accept nested extraParams (preferred by spec)
    @classmethod
    def model_validate(cls, obj, **kwargs):
        # Flatten extraParams if present
        if isinstance(obj, dict):
            extra = obj.get("extraParams", {})
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if k not in obj:
                        obj[k] = v
                # Map camelCase extraParams to snake_case internal fields
                if "sliceSize" in extra:
                    obj.setdefault("slice_size", extra["sliceSize"])
                if "sliceOverlap" in extra:
                    obj.setdefault("slice_overlap", extra["sliceOverlap"])
                if "highResRegions" in extra:
                    obj.setdefault("roi_regions", extra["highResRegions"])
        return super().model_validate(obj, **kwargs)


class InferResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    image_path: str = Field(..., alias="imagePath")
    processing_time_ms: float = Field(..., alias="processingTimeMs")
    total_slices: int = Field(0, alias="totalSlices")
    num_roi_regions: int = Field(0)
    defects: List[DefectResult]


class BatchImageItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image_id: str = Field(..., alias="imageId")
    image_path: str = Field(..., alias="imagePath")


class BatchInferRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    images: List[BatchImageItem]
    model_type: str = Field("onboard", alias="modelType")
    confidence_threshold: float = Field(0.40, alias="confidenceThreshold")
    max_batch_size: int = Field(16, alias="maxBatchSize")


class BatchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    image_id: str = Field(..., alias="imageId")
    image_path: str = Field(..., alias="imagePath")
    processing_time_ms: float = Field(..., alias="processingTimeMs")
    defects: List[DefectResult]


class BatchInferResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    results: List[BatchResult]
    total_time_ms: float = Field(..., alias="totalTimeMs")
    avg_time_per_image_ms: float = Field(..., alias="avgTimePerImageMs")
    throughput_images_per_sec: float = Field(..., alias="throughputImagesPerSec")


class ModelLoadRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_type: str = Field("onboard", alias="modelType")
    model_version: str = Field("latest", alias="modelVersion")
    force_reload: bool = Field(False, alias="forceReload")


class ModelLoadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    success: bool
    model_type: str = Field(..., alias="modelType")
    model_version: str = Field(..., alias="modelVersion")
    load_time_ms: float = Field(..., alias="loadTimeMs")
    message: str


class LoadedModelInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model_type: str = Field(..., alias="modelType")
    version: str
    loaded_at: str = Field(..., alias="loadedAt")


class GpuInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    available: bool
    gpu_count: int = Field(0, alias="gpuCount")


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    status: str
    loaded_models: List[LoadedModelInfo] = Field(default_factory=list, alias="loadedModels")
    gpu_available: bool = Field(..., alias="gpuAvailable")
    gpu_memory_used_mb: int = Field(0, alias="gpuMemoryUsedMB")
    gpu_memory_total_mb: int = Field(0, alias="gpuMemoryTotalMB")


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
        self.model_version: str = "latest"
        self.roi_model = None
        self.defect_model = None
        self.defect_model_b = None
        self.defect_dict: Dict[str, Any] = {}


state = AppState()


# ═══════════════════════════════════════════════════════════════════════════
# Defect dictionary loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_defect_dict() -> Dict[str, Any]:
    """Load defect dictionary, falling back to classes.py if JSON missing."""
    dict_path = Path(__file__).parent / "defect_dict.json"
    if dict_path.exists():
        with open(dict_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: build from classes.py
    from subway_defect.classes import DEFECT_CLASSES, CN_NAME_MAP, SEVERITY_MAP
    return {
        "_fallback": True,
        "defects": [
            {
                "code": code,
                "name_cn": CN_NAME_MAP.get(code, code),
                "severity": SEVERITY_MAP.get(code, "normal"),
            }
            for code in DEFECT_CLASSES
        ],
    }


def _resolve_defect_name(code: str) -> str:
    """Resolve a defect class code to its Chinese display name."""
    code_upper = code.upper()

    # Try the defect_dict first
    for d in state.defect_dict.get("defects", []):
        if d.get("code", "").upper() == code_upper:
            return d.get("name_cn", code_upper)

    # Fall back to classes.py mapping
    canonical = resolve_canonical(code_upper)
    return CN_NAME_MAP.get(canonical, canonical)


def _resolve_severity(code: str) -> str:
    """Resolve a defect class code to its severity level."""
    code_upper = code.upper()
    for d in state.defect_dict.get("defects", []):
        if d.get("code", "").upper() == code_upper:
            return d.get("severity", "normal")
    canonical = resolve_canonical(code_upper)
    return SEVERITY_MAP.get(canonical, "normal")


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_models(config: dict):
    """Load models with warmup inference for fast first prediction."""
    t0 = time.time()

    # Load ROI proposer
    roi_path = config.get("roi_model", "yolo11n.pt")
    state.roi_model = YOLO(roi_path)
    warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    state.roi_model(warmup, verbose=False)

    # Load defect detector
    defect_path = config.get("defect_model")
    state.defect_model = YOLO(defect_path)
    state.defect_model(warmup, verbose=False)

    # Build pipeline
    state.pipeline = TwoStagePipeline(
        roi_model=state.roi_model,
        defect_model=state.defect_model,
        slice_size=config.get("slice_size", 1024),
        overlap=config.get("overlap", 0.15),
        roi_conf=config.get("roi_conf", 0.15),
        defect_conf=config.get("defect_conf", 0.40),
    )

    # Ground-side: second model + WBF fusion
    if config.get("mode") == "ground" and config.get("defect_model_b"):
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
        state.mode = "ground"
    else:
        state.mode = config.get("mode", "onboard")
        if state.mode == "vehicle":
            state.mode = "onboard"  # legacy alias

    state.model_loaded_at = time.time()
    state.model_version = config.get("model_version", "latest")
    load_ms = (state.model_loaded_at - t0) * 1000
    return load_ms


# ═══════════════════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup GPU memory."""
    # Load defect dictionary
    state.defect_dict = _load_defect_dict()

    config = app.state.config
    load_ms = load_models(config)
    print(f"Models loaded in {load_ms:.0f}ms (mode={state.mode})")

    # Report GPU status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {props.name}, "
                  f"memory {mem_free/1024**3:.1f}/{mem_total/1024**3:.1f} GB")

    yield
    # Cleanup
    state.pipeline = None
    state.roi_model = None
    state.defect_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Subway Defect Detection API", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_legacy():
    """Legacy health check (redirects to /api/dl/health)."""
    return await health()


@app.get("/api/dl/health")
async def health():
    """Health check with loaded model info and GPU status.

    Per 接口规范标准 §4.1 / 深度学习接口方案 §2.1.
    """
    gpu_available = torch.cuda.is_available()
    gpu_used = 0
    gpu_total = 0
    if gpu_available:
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        gpu_used = (mem_total - mem_free) // (1024 * 1024)
        gpu_total = mem_total // (1024 * 1024)

    loaded_models = []
    if state.pipeline is not None:
        loaded_at_iso = ""
        if state.model_loaded_at > 0:
            import datetime
            loaded_at_iso = datetime.datetime.fromtimestamp(
                state.model_loaded_at
            ).isoformat()
        loaded_models.append({
            "modelType": state.mode,
            "version": state.model_version,
            "loadedAt": loaded_at_iso,
        })

    return {
        "status": "healthy" if state.pipeline else "loading",
        "loadedModels": loaded_models,
        "gpuAvailable": gpu_available,
        "gpuMemoryUsedMB": gpu_used,
        "gpuMemoryTotalMB": gpu_total,
    }


@app.post("/api/dl/model/load")
async def load_model(req: ModelLoadRequest):
    """Load or reload a detection model.

    Per 深度学习接口方案 §2.2.
    """
    try:
        config = app.state.config
        # Accept both "onboard" and "vehicle" for backward compat
        mode = req.model_type
        if mode == "vehicle":
            mode = "onboard"
        config["mode"] = mode
        config["model_version"] = req.model_version
        load_ms = load_models(config)
        return ModelLoadResponse(
            success=True,
            model_type=mode,
            model_version=req.model_version,
            load_time_ms=load_ms,
            message="Model loaded successfully",
        )
    except Exception as e:
        return ModelLoadResponse(
            success=False,
            model_type=req.model_type,
            model_version=req.model_version,
            load_time_ms=0,
            message=str(e),
        )


@app.post("/api/dl/infer")
async def infer(req: InferRequest):
    """Single-image defect detection.

    Per 深度学习接口方案 §2.3 / 接口规范标准 §4.2.
    """
    if state.pipeline is None:
        return dl_error_response(
            "DL_MODEL_NOT_LOADED",
            suggestion="Send POST /api/dl/model/load before inference.",
        )

    try:
        # Basic path traversal protection
        image_path = Path(req.image_path).resolve()
        if not image_path.is_file():
            return dl_error_response(
                "DL_IMAGE_UNREADABLE",
                detail=f"File not found: {req.image_path}",
            )

        img = cv2.imread(str(image_path))
        if img is None:
            return dl_error_response(
                "DL_IMAGE_UNREADABLE",
                detail=f"Cannot decode image: {req.image_path}",
            )

        # Update thresholds and slicer params for this request
        state.pipeline.defect_conf = req.confidence_threshold
        state.pipeline.slicer.slice_size = req.slice_size
        state.pipeline.slicer.overlap = req.slice_overlap
        state.pipeline.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))

        # Run pipeline A
        t0 = time.time()
        result_a = state.pipeline.infer(img)

        # Timeout guard (30s per spec)
        if (time.time() - t0) > 30.0:
            return dl_error_response("DL_INFERENCE_TIMEOUT")

        # Ground-side: run pipeline B and fuse
        fusion_start = time.time()
        total_slices = result_a.get("total_slices", 0)
        if state.mode == "ground" and state.pipeline_b:
            state.pipeline_b.defect_conf = req.confidence_threshold
            state.pipeline_b.slicer.slice_size = req.slice_size
            state.pipeline_b.slicer.overlap = req.slice_overlap
            state.pipeline_b.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))
            result_b = state.pipeline_b.infer(img)
            total_slices = max(total_slices, result_b.get("total_slices", 0))
            fused_defects = state.fusion.fuse(
                result_a["defects"], result_b["defects"])
            total_ms = max(
                result_a["total_time_ms"], result_b["total_time_ms"])
            total_ms += (time.time() - fusion_start) * 1000
        else:
            fused_defects = result_a["defects"]
            total_ms = result_a["total_time_ms"]

        # Format response with proper Chinese names
        defects = []
        for d in fused_defects:
            raw_code = d.get("class_name", "")
            code = resolve_canonical(raw_code)
            defects.append(DefectResult(
                defect_type=code,
                defect_name=_resolve_defect_name(code),
                confidence=float(d["confidence"]),
                box=BoxCoords(**d["box"]),
                coord_type="normalized",  # current pipeline always outputs normalized
                source_slice=SourceSlice(**d["source_tile"]) if d.get("source_tile") else None,
            ))

        return InferResponse(
            success=True,
            image_path=req.image_path,
            processing_time_ms=total_ms,
            total_slices=total_slices,
            num_roi_regions=result_a.get("num_roi_regions", 0),
            defects=defects,
        )
    except HTTPException:
        raise
    except Exception as e:
        return dl_error_response(
            "DL_INTERNAL_ERROR",
            detail=str(e),
            suggestion="Check server logs for full traceback.",
        )


@app.post("/api/dl/infer/batch")
async def infer_batch(req: BatchInferRequest):
    """Batch inference for ground-side high-throughput processing.

    Per 深度学习接口方案 §2.4.
    """
    if state.pipeline is None:
        return dl_error_response(
            "DL_MODEL_NOT_LOADED",
            suggestion="Send POST /api/dl/model/load before inference.",
        )

    t_batch_start = time.time()
    results: List[BatchResult] = []
    total_defects = 0

    for item in req.images:
        try:
            image_path = Path(item.image_path).resolve()
            if not image_path.is_file():
                results.append(BatchResult(
                    image_id=item.image_id,
                    image_path=item.image_path,
                    processing_time_ms=0,
                    defects=[],
                ))
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                results.append(BatchResult(
                    image_id=item.image_id,
                    image_path=item.image_path,
                    processing_time_ms=0,
                    defects=[],
                ))
                continue

            t_img_start = time.time()
            result = state.pipeline.infer(img)
            img_ms = result.get("total_time_ms", 0)

            defects = []
            for d in result.get("defects", []):
                raw_code = d.get("class_name", "")
                code = resolve_canonical(raw_code)
                defects.append(DefectResult(
                    defect_type=code,
                    defect_name=_resolve_defect_name(code),
                    confidence=float(d["confidence"]),
                    box=BoxCoords(**d["box"]),
                    coord_type="normalized",
                    source_slice=SourceSlice(**d["source_tile"]) if d.get("source_tile") else None,
                ))

            results.append(BatchResult(
                image_id=item.image_id,
                image_path=item.image_path,
                processing_time_ms=img_ms,
                defects=defects,
            ))
            total_defects += len(defects)
        except Exception as e:
            results.append(BatchResult(
                image_id=item.image_id,
                image_path=item.image_path,
                processing_time_ms=0,
                defects=[],
            ))

    total_ms = (time.time() - t_batch_start) * 1000
    n_images = len(req.images)
    avg_ms = total_ms / n_images if n_images > 0 else 0
    throughput = (n_images / total_ms * 1000) if total_ms > 0 else 0

    return BatchInferResponse(
        success=True,
        results=results,
        total_time_ms=total_ms,
        avg_time_per_image_ms=avg_ms,
        throughput_images_per_sec=round(throughput, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions → DL_INTERNAL_ERROR."""
    return dl_error_response(
        "DL_INTERNAL_ERROR",
        detail=str(exc),
        suggestion="Check server logs for full traceback.",
    )


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
                        choices=["onboard", "vehicle", "ground"])
    parser.add_argument("--slice_size", type=int, default=1024)
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--roi_conf", type=float, default=0.15)
    parser.add_argument("--defect_conf", type=float, default=0.40)
    parser.add_argument("--model_version", default="latest")
    args = parser.parse_args()

    # Normalize mode: "vehicle" → "onboard" (backward compat)
    mode = args.mode
    if mode == "vehicle":
        mode = "onboard"

    app.state.config = {
        "defect_model": args.model,
        "roi_model": args.roi_model,
        "defect_model_b": args.model_b,
        "mode": mode,
        "slice_size": args.slice_size,
        "overlap": args.overlap,
        "roi_conf": args.roi_conf,
        "defect_conf": args.defect_conf,
        "model_version": args.model_version,
    }

    print(f"Starting {mode} inference server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
