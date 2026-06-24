#!/usr/bin/env python3
"""
FastAPI inference server for subway catenary defect detection.

Supports vehicle-side (single model) and ground-side (dual ensemble) modes.

Endpoints:
    GET  /health              — health check with GPU status
    POST /api/dl/infer        — single image inference
    POST /api/dl/model/load   — load/reload a model

Usage:
    # Vehicle-side
    python -m subway_defect.deployment.fastapi_server --port 8001 \
        --model runs/defect_detector_c2_full/weights/best.pt \
        --mode vehicle

    # Ground-side (dual GPU)
    python -m subway_defect.deployment.fastapi_server --port 8001 \
        --model runs/defect_detector_c2_full/weights/best.pt \
        --model_b runs/defect_detector_p2/weights/best.pt \
        --mode ground
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from subway_yolo import YOLO

from subway_defect.pipeline.two_stage import TwoStagePipeline
from subway_defect.pipeline.wbf_fusion import WBFFusion


# -- Request/Response models --

class InferRequest(BaseModel):
    image_path: str
    model_type: str = "vehicle"
    confidence_threshold: float = 0.40
    slice_size: int = 1024
    slice_overlap: float = 0.15
    roi_regions: Optional[List[dict]] = None


class BoxCoords(BaseModel):
    x: float
    y: float
    w: float
    h: float


class DefectResult(BaseModel):
    defect_type: str
    defect_name: str
    confidence: float
    box: BoxCoords
    source_tile: Optional[dict] = None


class InferResponse(BaseModel):
    success: bool
    image_path: str
    processing_time_ms: float
    defects: List[DefectResult]
    num_roi_regions: int = 0


class ModelLoadRequest(BaseModel):
    model_type: str = "vehicle"
    model_version: str = "latest"
    force_reload: bool = False


class ModelLoadResponse(BaseModel):
    success: bool
    model_type: str
    model_version: str
    load_time_ms: float
    message: str


# -- Application state --

class AppState:
    def __init__(self):
        self.pipeline: Optional[TwoStagePipeline] = None
        self.pipeline_b: Optional[TwoStagePipeline] = None
        self.fusion: Optional[WBFFusion] = None
        self.mode: str = "vehicle"
        self.model_loaded_at: float = 0.0
        self.roi_model = None
        self.defect_model = None
        self.defect_model_b = None


state = AppState()


def load_models(config: dict):
    """Load models with warmup inference for fast first prediction."""
    t0 = time.time()

    # Load ROI proposer
    roi_path = config.get("roi_model", "yolo11n.pt")
    state.roi_model = YOLO(roi_path)
    # Warmup
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
        state.mode = "vehicle"

    state.model_loaded_at = time.time()
    load_ms = (state.model_loaded_at - t0) * 1000
    return load_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models. Shutdown: cleanup GPU memory."""
    config = app.state.config
    load_ms = load_models(config)
    print(f"Models loaded in {load_ms:.0f}ms (mode={state.mode})")

    # Report GPU status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = torch.cuda.mem_get_info(i)
            print(f"GPU {i}: {props.name}, "
                  f"memory {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB")

    yield
    # Cleanup
    state.pipeline = None
    state.roi_model = None
    state.defect_model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="Subway Defect Detection API", lifespan=lifespan)


# -- Endpoints --

@app.get("/health")
async def health():
    gpu_info = {"available": torch.cuda.is_available()}
    if gpu_info["available"]:
        for i in range(torch.cuda.device_count()):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            gpu_info[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_used_mb": (mem_total - mem_free) // (1024 * 1024),
                "memory_total_mb": mem_total // (1024 * 1024),
            }
    return {
        "status": "healthy" if state.pipeline else "loading",
        "mode": state.mode,
        "gpu": gpu_info,
    }


@app.post("/api/dl/model/load", response_model=ModelLoadResponse)
async def load_model(req: ModelLoadRequest):
    try:
        config = app.state.config
        config["mode"] = req.model_type
        load_ms = load_models(config)
        return ModelLoadResponse(
            success=True,
            model_type=req.model_type,
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


@app.post("/api/dl/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if state.pipeline is None:
        raise HTTPException(503, "Model not loaded")

    try:
        # Basic path traversal protection
        image_path = Path(req.image_path).resolve()
        if not image_path.is_file():
            raise HTTPException(400, f"Image not found: {req.image_path}")

        img = cv2.imread(str(image_path))
        if img is None:
            raise HTTPException(400, f"Cannot read image: {req.image_path}")

        # Update thresholds and slicer params for this request
        state.pipeline.defect_conf = req.confidence_threshold
        state.pipeline.slicer.slice_size = req.slice_size
        state.pipeline.slicer.overlap = req.slice_overlap
        state.pipeline.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))

        # Run pipeline A
        result_a = state.pipeline.infer(img)

        # Ground-side: run pipeline B and fuse
        fusion_start = time.time()
        if state.mode == "ground" and state.pipeline_b:
            state.pipeline_b.defect_conf = req.confidence_threshold
            state.pipeline_b.slicer.slice_size = req.slice_size
            state.pipeline_b.slicer.overlap = req.slice_overlap
            state.pipeline_b.slicer.stride = int(req.slice_size * (1 - req.slice_overlap))
            result_b = state.pipeline_b.infer(img)
            fused_defects = state.fusion.fuse(
                result_a["defects"], result_b["defects"])
            total_ms = max(
                result_a["total_time_ms"], result_b["total_time_ms"])
            total_ms += (time.time() - fusion_start) * 1000
        else:
            fused_defects = result_a["defects"]
            total_ms = result_a["total_time_ms"]

        # Format response
        defects = [
            DefectResult(
                defect_type=d.get("class_name", ""),
                defect_name=d.get("class_name", ""),
                confidence=d["confidence"],
                box=BoxCoords(**d["box"]),
                source_tile=d.get("source_tile"),
            )
            for d in fused_defects
        ]

        return InferResponse(
            success=True,
            image_path=req.image_path,
            processing_time_ms=total_ms,
            defects=defects,
            num_roi_regions=result_a.get("num_roi_regions", 0),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# -- Entry point --

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
    parser.add_argument("--mode", default="vehicle",
                        choices=["vehicle", "ground"])
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
