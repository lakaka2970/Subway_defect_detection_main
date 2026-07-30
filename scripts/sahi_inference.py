#!/usr/bin/env python3
"""
SAHI (Slicing Aided Hyper Inference) for large-image defect detection.

Slices large source images into overlapping tiles, runs YOLO detection on
each tile, merges results with NMS, and outputs full-image detections.
Critical for detecting small defects (e.g. RHTBNM @ ~80px) in source images
that are typically 4000-8000px wide.

Usage::

    # Single image
    python scripts/sahi_inference.py \\
        --model weights/stage4_best_finetune.pt \\
        --source data/raw/test_image.jpg \\
        --output output/sahi_results/

    # Batch directory
    python scripts/sahi_inference.py \\
        --model weights/stage4_best_finetune.pt \\
        --source data/raw/ \\
        --output output/sahi_results/ \\
        --tile-size 1280 --overlap 0.25

    # With cascade filtering
    python scripts/sahi_inference.py \\
        --model weights/stage4_best_finetune.pt \\
        --source data/raw/ \\
        --cascade --cascade-weights-dir weights \\
        --output output/sahi_cascade/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SlicerConfig:
    """SAHI slicing parameters."""
    tile_size: int = 1280
    overlap_ratio: float = 0.25  # overlap between adjacent tiles (0.0-1.0)
    postprocess_match_metric: str = "IOS"  # "IOS" | "IOU"
    postprocess_match_threshold: float = 0.5
    postprocess_class_agnostic: bool = True
    min_detection_size: int = 10  # pixels, filter tiny artifacts


def _overlap_pixels(tile_size: int, overlap_ratio: float) -> int:
    return int(tile_size * overlap_ratio)


def compute_tiles(
    img_w: int, img_h: int, tile_size: int, overlap_ratio: float
) -> List[Dict[str, int]]:
    """Generate overlapping tile coordinates for an image.

    Returns:
        List of tiles, each with ``x, y, w, h`` in pixel coords.
    """
    stride = tile_size - _overlap_pixels(tile_size, overlap_ratio)
    if stride <= 0:
        stride = tile_size // 2

    tiles = []
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            w = min(tile_size, img_w - x)
            h = min(tile_size, img_h - y)
            if w < 64 or h < 64:
                continue  # skip edge fragments
            tiles.append({"x": x, "y": y, "w": w, "h": h})
    return tiles


def _xyxy_to_global(
    local_xyxy: Tuple[float, float, float, float],
    tile_x: int,
    tile_y: int,
) -> Tuple[float, float, float, float]:
    """Convert tile-local (x1, y1, x2, y2) to image-global coordinates."""
    return (
        local_xyxy[0] + tile_x,
        local_xyxy[1] + tile_y,
        local_xyxy[2] + tile_x,
        local_xyxy[3] + tile_y,
    )


def sahi_nms(
    detections: List[Dict],
    match_metric: str = "IOS",
    match_threshold: float = 0.5,
    class_agnostic: bool = True,
) -> List[Dict]:
    """Merge overlapping detections from tiles using NMS.

    Uses Intersection-over-Smaller (IOS) by default, which is more
    appropriate than IOU for merging partial detections at tile boundaries.

    Args:
        detections: Flat list of detection dicts with ``x1, y1, x2, y2, conf, cls``.
        match_metric: ``"IOS"`` or ``"IOU"``.
        match_threshold: Suppression threshold.
        class_agnostic: If True, suppress across classes.

    Returns:
        Merged detection list.
    """
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    kept = []
    suppressed = set()

    for i, det_a in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det_a)
        for j, det_b in enumerate(detections[i + 1:], start=i + 1):
            if j in suppressed:
                continue
            if not class_agnostic and det_a["cls"] != det_b["cls"]:
                continue

            a_box = (det_a["x1"], det_a["y1"], det_a["x2"], det_a["y2"])
            b_box = (det_b["x1"], det_b["y1"], det_b["x2"], det_b["y2"])

            inter_x1 = max(a_box[0], b_box[0])
            inter_y1 = max(a_box[1], b_box[1])
            inter_x2 = min(a_box[2], b_box[2])
            inter_y2 = min(a_box[3], b_box[3])
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            if match_metric == "IOS":
                area_a = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
                area_b = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                metric = inter_area / min(area_a, area_b) if min(area_a, area_b) > 0 else 0
            else:  # IOU
                area_a = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
                area_b = (b_box[2] - b_box[0]) * (b_box[3] - b_box[1])
                union = area_a + area_b - inter_area
                metric = inter_area / union if union > 0 else 0

            if metric >= match_threshold:
                suppressed.add(j)

    return kept


def _filter_by_size(detections: List[Dict], min_size: int) -> List[Dict]:
    """Remove detections below minimum pixel size."""
    if min_size <= 0:
        return detections
    return [
        d for d in detections
        if (d["x2"] - d["x1"]) >= min_size and (d["y2"] - d["y1"]) >= min_size
    ]


def run_sahi(
    model_path: str,
    image_path: Path,
    config: SlicerConfig,
    device: str = "0",
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.5,
    cascade: Optional[Any] = None,
    verbose: bool = True,
) -> Tuple[List[Dict], float]:
    """Run SAHI inference on a single image.

    Returns:
        (merged_detections, elapsed_seconds)
    """
    from subway_yolo import YOLO

    t0 = time.time()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_h, img_w = img.shape[:2]

    tiles = compute_tiles(img_w, img_h, config.tile_size, config.overlap_ratio)

    if verbose:
        print(f"  Image: {image_path.name} ({img_w}x{img_h})")
        print(f"  Tiles: {len(tiles)} (size={config.tile_size}, overlap={config.overlap_ratio*100:.0f}%)")

    # ── Batch inference over tiles ──
    model = YOLO(model_path)
    all_detections: List[Dict] = []

    tile_batch_size = 4  # small batch to keep VRAM under control
    for batch_start in range(0, len(tiles), tile_batch_size):
        batch_end = min(batch_start + tile_batch_size, len(tiles))
        batch_tiles = tiles[batch_start:batch_end]

        # Extract tile images
        tile_imgs = []
        for t in batch_tiles:
            crop = img[t["y"]:t["y"] + t["h"], t["x"]:t["x"] + t["w"]]
            tile_imgs.append(crop)

        results = model.predict(
            source=tile_imgs,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=config.tile_size,
            device=device,
            stream=False,
            verbose=False,
        )

        for tile_result, tile in zip(results, batch_tiles):
            if tile_result.boxes is None or len(tile_result.boxes.cls) == 0:
                continue
            boxes = tile_result.boxes
            for i in range(len(boxes.cls)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxyn = boxes.xyxyn[i].cpu().numpy()
                # Convert normalized → pixel in tile space → global
                local_x1 = xyxyn[0] * tile["w"]
                local_y1 = xyxyn[1] * tile["h"]
                local_x2 = xyxyn[2] * tile["w"]
                local_y2 = xyxyn[3] * tile["h"]
                gx1, gy1, gx2, gy2 = _xyxy_to_global(
                    (local_x1, local_y1, local_x2, local_y2),
                    tile["x"], tile["y"],
                )
                all_detections.append({
                    "x1": float(gx1), "y1": float(gy1),
                    "x2": float(gx2), "y2": float(gy2),
                    "conf": conf, "cls": cls_id,
                })

    # ── Merge detections ──
    merged = sahi_nms(
        all_detections,
        match_metric=config.postprocess_match_metric,
        match_threshold=config.postprocess_match_threshold,
        class_agnostic=config.postprocess_class_agnostic,
    )

    # ── Size filter ──
    merged = _filter_by_size(merged, config.min_detection_size)

    # ── Cascade filter (optional) ──
    if cascade is not None:
        detections_for_filter = [
            dict(d) for d in merged
        ]  # copy for mutation
        kept, rejected = cascade.filter_detections(img, detections_for_filter)
        merged = kept
        if verbose:
            print(f"  Cascade: {len(kept)} kept, {len(rejected)} rejected")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Raw detections: {len(all_detections)}, Merged: {len(merged)}")
        print(f"  Time: {elapsed:.1f}s")
        print()

    return merged, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAHI inference for large-image defect detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=str, help="Path to YOLO model")
    parser.add_argument("--source", required=True, type=str, help="Image file or directory")
    parser.add_argument("--output", required=True, type=str, help="Output directory for results")
    parser.add_argument("--tile-size", type=int, default=1280, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.25, help="Tile overlap ratio (0-1)")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU threshold")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum detection size in pixels")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--cascade", action="store_true", help="Enable cascade classifier filtering")
    parser.add_argument("--cascade-weights-dir", type=str, default="weights")
    parser.add_argument("--cascade-conf", type=float, default=0.55,
                        help="Cascade classifier confidence threshold")
    parser.add_argument("--save-json", action="store_true", help="Save JSON results alongside images")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization images with boxes")
    args = parser.parse_args()

    config = SlicerConfig(
        tile_size=args.tile_size,
        overlap_ratio=args.overlap,
        min_detection_size=args.min_size,
    )

    # ── Resolve source ──
    source_path = Path(args.source)
    if source_path.is_file():
        images = [source_path]
    elif source_path.is_dir():
        images = sorted(
            p for p in source_path.iterdir()
            if p.suffix.lower() in IMG_SUFFIXES
        )
    else:
        print(f"ERROR: Source not found: {source_path}")
        sys.exit(1)

    if not images:
        print("No images found.")
        sys.exit(0)

    # ── Output ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # ── Cascade ──
    cascade = None
    if args.cascade:
        from subway_defect.pipeline.cascade import CascadeClassifier
        cascade = CascadeClassifier(
            weights_dir=args.cascade_weights_dir,
            device=args.device,
            confidence_threshold=args.cascade_conf,
        )
        print(cascade.summary())
        print()

    # ── Process ──
    total_time = 0.0
    all_results = {}
    for img_path in images:
        detections, elapsed = run_sahi(
            model_path=args.model,
            image_path=img_path,
            config=config,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            cascade=cascade,
        )
        total_time += elapsed
        all_results[img_path.name] = {
            "image": str(img_path),
            "width": None if not img_path.exists() else cv2.imread(str(img_path)).shape[1],
            "height": None if not img_path.exists() else cv2.imread(str(img_path)).shape[0],
            "tiles": len(compute_tiles(
                cv2.imread(str(img_path)).shape[1],
                cv2.imread(str(img_path)).shape[0],
                config.tile_size, config.overlap_ratio,
            )),
            "detections": detections,
            "elapsed_seconds": elapsed,
        }

    # ── Save ──
    if args.save_json:
        json_path = json_dir / "detections.json"
        json_output = {}
        for name, data in all_results.items():
            json_output[name] = {
                k: v for k, v in data.items()
                if k not in ("detections",)
            }
            json_output[name]["detections"] = [
                {"x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
                 "conf": d["conf"], "cls": d["cls"]}
                for d in data["detections"]
            ]
        json_path.write_text(json.dumps(json_output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved: {json_path}")

    # ── Summary ──
    total_dets = sum(len(d["detections"]) for d in all_results.values())
    print()
    print("=" * 60)
    print(f"  SAHI Inference Complete")
    print(f"  Images: {len(images)}")
    print(f"  Total detections: {total_dets}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Avg per image: {total_time/len(images):.1f}s")
    if args.cascade:
        print(f"  Cascade: enabled (conf≥{args.cascade_conf})")
    print("=" * 60)


if __name__ == "__main__":
    main()
