#!/usr/bin/env python3
"""
一键推理脚本 — 7.25 训练最优 7 类模型对 Defect_dataset 全部图片推理
====================================================================

用途:
    加载 7.25 五阶段训练产出的最优 7 类检测模型 (stage4_best_finetune.pt),
    以 tile-based 方式对 data/Defect_dataset 中 5120×5120 原图执行推理.
    每张原图被切分为 1280×1280 重叠切片后逐片推理, 结果经 NMS 合并.

    与训练尺度一致: 模型在 1280×1280 subway_crops 上训练, 本脚本用相同
    分辨率切片推理, 确保检出率与训练 mAP 匹配.

用法:
    # 默认参数 (仅输出报告)
    python scripts/run_inference.py

    # 查看帮助
    python scripts/run_inference.py --help

输出:
    output/inference/<时间戳>/
    ├── report.json          # 机器可读报告
    └── report.txt           # 文本报告

模型:
    YOLO11s-EMA-SimAM (9.58M 参数, 23.3 GFLOPs)
    7 类: VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM
    mAP50=0.622 (subway_crops val, 2,869 张 1280×1280 切片)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm as TQDM

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_yolo import YOLO

# ═══════════════════════════════════════════════════════════════════════════
# 默认配置
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL = "weights/stage4_best_finetune.pt"
DEFAULT_DATA_DIRS = [
    "data/Defect_dataset/images/train",
    "data/Defect_dataset/images/val",
]
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7
DEFAULT_TILE_SIZE = 1280    # 切片尺寸, 与训练分辨率一致
DEFAULT_OVERLAP = 0.15      # 切片重叠比例
DEFAULT_DEVICE = "0"
DEFAULT_OUTPUT_BASE = "output/inference"

# Cascade classifier defaults
DEFAULT_CASCADE_WEIGHTS_DIR = "weights"
DEFAULT_CASCADE_THRESHOLD = 0.55

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ═══════════════════════════════════════════════════════════════════════════
# 切片 & 合并
# ═══════════════════════════════════════════════════════════════════════════

def slice_image(
    img: np.ndarray,
    tile_size: int = 1280,
    overlap: float = 0.15,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    """将大图切分为重叠的方形切片.

    Args:
        img: 输入灰度图 (H, W).
        tile_size: 切片边长.
        overlap: 切片间重叠比例 (0~1).

    Returns:
        (tiles, offsets): 切片数组列表和每个切片的 (x0, y0) 偏移.
    """
    h, w = img.shape[:2]
    stride = int(tile_size * (1 - overlap))

    # 边缘对齐: 确保覆盖全部像素, 最后一个切片右/下对齐
    n_cols = max(1, math.ceil((w - tile_size) / stride) + 1)
    n_rows = max(1, math.ceil((h - tile_size) / stride) + 1)

    tiles = []
    offsets = []
    for row in range(n_rows):
        y0 = min(row * stride, h - tile_size)
        for col in range(n_cols):
            x0 = min(col * stride, w - tile_size)
            tile = img[y0:y0 + tile_size, x0:x0 + tile_size]
            # 转 BGR 3-ch 供 YOLO 使用 (模型期望 3 通道)
            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            tiles.append(tile)
            offsets.append((x0, y0))

    return tiles, offsets


def nms_merge_detections(
    detections: List[Tuple[float, float, float, float, float, int]],
    iou_thresh: float = 0.5,
) -> List[Dict]:
    """对来自同一原图多个切片的检测框做 NMS 合并.

    Args:
        detections: 列表, 每项 (x1, y1, x2, y2, conf, cls_id) 在原图坐标系.
        iou_thresh: 合并时 IoU 阈值.

    Returns:
        合并后的检测结果列表 [{x1,y1,x2,y2,conf,cls}, ...].
    """
    if not detections:
        return []

    boxes_np = np.array([d[:4] for d in detections], dtype=np.float32)
    scores_np = np.array([d[4] for d in detections], dtype=np.float32)
    cls_np = np.array([d[5] for d in detections], dtype=np.int32)

    # 转为 xyxy 格式的 tensor
    boxes_t = torch.from_numpy(boxes_np)
    scores_t = torch.from_numpy(scores_np)

    # 用 torchvision NMS (class-agnostic: 所有类一起合并, 同类相同对象不应重复)
    from torchvision.ops import nms
    keep = nms(boxes_t, scores_t, iou_thresh)
    keep = keep.numpy()

    merged = []
    for idx in keep:
        merged.append({
            "x1": float(boxes_np[idx][0]),
            "y1": float(boxes_np[idx][1]),
            "x2": float(boxes_np[idx][2]),
            "y2": float(boxes_np[idx][3]),
            "conf": float(scores_np[idx]),
            "cls": int(cls_np[idx]),
        })
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def collect_images(dirs: List[str]) -> List[Path]:
    """递归收集目录中所有唯一图片路径 (按文件名去重)."""
    seen: set = set()
    paths: List[Path] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            print(f"  ⚠ 目录不存在, 跳过: {d}")
            continue
        for ext in IMG_EXTENSIONS:
            for p in sorted(d.rglob(f"*{ext}")):
                key = p.name
                if key not in seen:
                    seen.add(key)
                    paths.append(p.resolve())
    return paths


def validate_images(paths: List[Path]) -> Tuple[List[Path], List[str]]:
    """预验证图片可读性, 剔除损坏文件."""
    good: List[Path] = []
    bad: List[str] = []
    for p in TQDM(paths, desc="Validating images", unit="file"):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            good.append(p)
        else:
            bad.append(p.name)
    if bad:
        print(f"  ⚠ 跳过 {len(bad)} 个损坏/不可读文件:")
        for name in bad[:10]:
            print(f"      {name}")
        if len(bad) > 10:
            print(f"      ... 等共 {len(bad)} 个")
    return good, bad


def format_elapsed(seconds: float) -> str:
    """格式化耗时为人可读字符串."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


# ═══════════════════════════════════════════════════════════════════════════
# 核心推理逻辑
# ═══════════════════════════════════════════════════════════════════════════

def run_inference_tiled(
    model_path: str,
    image_paths: List[Path],
    conf: float,
    iou: float,
    tile_size: int,
    overlap: float,
    device: str,
    output_dir: Path,
    source_dirs: List[str] | None = None,
    bad_images: List[str] | None = None,
    cascade=None,
) -> Dict:
    """Tile-based 推理: 每张 5120×5120 原图切分为 1280×1280 切片后逐片推理,
    然后将各切片检测结果映射回原图坐标并做 NMS 合并.

    模型在 1280×1280 subway_crops 上训练, 切片推理保持与训练一致的缺陷尺度,
    从而获得与训练 mAP 匹配的检出率.
    """
    n_images = len(image_paths)
    if n_images == 0:
        raise ValueError("未找到任何图片! 请检查 --data 参数指定的目录.")

    # ── 加载模型 ──────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  模型: {model_path}")
    model = YOLO(str(model_path))
    class_names: Dict[int, str] = model.names
    n_classes = len(class_names)
    stride_val = int(tile_size * (1 - overlap))
    print(f"  类别数: {n_classes}")
    print(f"  类别: {list(class_names.values())}")
    print(f"  数据集图片数: {n_images}")
    print(f"  切片尺寸: {tile_size}×{tile_size}px")
    print(f"  切片步长: {stride_val}px (overlap={overlap})")
    print(f"  置信度阈值: {conf}")
    print(f"  NMS IoU: {iou}")
    print(f"  设备: {device}")
    print(f"  输出目录: {output_dir}")
    if cascade is not None and cascade.enabled:
        print(f"  Cascade 分类器: 已启用 (threshold={cascade.confidence_threshold})")
        print(f"    已加载类别: {cascade.available_classes}")
        if cascade.failed_classes:
            print(f"    加载失败: {cascade.failed_classes}")
    else:
        print(f"  Cascade 分类器: 未启用")
    print(f"{'='*64}\n")

    # ── 逐图切片推理 ──────────────────────────────────────────────────
    per_class = Counter()
    det_counts: List[int] = []        # 每张原图的最终检出数
    total_dets_running = 0
    total_tiles_all = 0
    total_cascade_rejected = 0        # cascade 过滤掉的检测数
    total_cascade_input = 0           # 送入 cascade 的检测数
    t_start = time.perf_counter()

    # 复用单个临时目录, 避免每次 mkdtemp 开销
    tile_tmp = Path(tempfile.mkdtemp(prefix="tiles_"))

    pbar = TQDM(image_paths, desc="Processing images", unit="img",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for img_path in pbar:
        # 读图
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            det_counts.append(0)
            continue

        # 切片
        tiles, offsets = slice_image(img, tile_size, overlap)
        total_tiles_all += len(tiles)

        # 写入临时切片文件供 YOLO 批量推理 (比传 numpy 数组更快, YOLO 内部有高效 batch)
        for ti, tile in enumerate(tiles):
            cv2.imwrite(str(tile_tmp / f"t_{ti:04d}.jpg"), tile)

        tile_results = model.predict(
            source=str(tile_tmp),
            imgsz=tile_size,
            conf=conf,
            iou=iou,
            device=device,
            save=False,
            save_txt=False,
            verbose=False,
            stream=False,
        )

        # 清理当前批次的临时文件
        for f in tile_tmp.glob("t_*.jpg"):
            f.unlink()

        # 收集切片检测结果并映射到原图坐标
        image_dets: List[Tuple[float, float, float, float, float, int]] = []
        for ti, tr in enumerate(tile_results):
            if tr.boxes is not None and len(tr.boxes) > 0:
                x0, y0 = offsets[ti]
                for bi in range(len(tr.boxes)):
                    x1, y1, x2, y2 = tr.boxes.xyxy[bi].cpu().numpy()
                    cls_id = int(tr.boxes.cls[bi])
                    conf_sc = float(tr.boxes.conf[bi])
                    image_dets.append((x1 + x0, y1 + y0, x2 + x0, y2 + y0, conf_sc, cls_id))

        # NMS 合并重叠切片中的重复检出
        merged = nms_merge_detections(image_dets, iou_thresh=iou)

        # ── Cascade 分类器过滤 (FP 抑制) ─────────────────────────────
        if cascade is not None and cascade.enabled and merged:
            # 原图为灰度, cascade 需要 BGR 3 通道
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            total_cascade_input += len(merged)
            merged, cascade_rejected = cascade.filter_detections(img_bgr, merged)
            total_cascade_rejected += len(cascade_rejected)

        # 累计统计
        nd = len(merged)
        det_counts.append(nd)
        total_dets_running += nd
        for det in merged:
            per_class[det["cls"]] += 1

        pbar.set_postfix({
            "tiles": len(tiles),
            "dets": nd,
            "total": total_dets_running,
        })

    pbar.close()
    # 清理临时目录
    tile_tmp.rmdir()
    t_total = time.perf_counter() - t_start

    # ── 构建报告 ──────────────────────────────────────────────────────
    total_imgs = len(det_counts)
    imgs_with_dets = sum(1 for d in det_counts if d > 0)
    total_dets = sum(det_counts)

    report = {
        "model": str(Path(model_path).resolve()),
        "model_name": Path(model_path).name,
        "date": datetime.now().isoformat(),
        "mode": "tiled",
        "config": {
            "conf": conf,
            "iou": iou,
            "tile_size": tile_size,
            "overlap": overlap,
            "tile_stride": stride_val,
            "device": device,
        },
        "classes": {int(k): str(v) for k, v in class_names.items()},
        "num_classes": n_classes,
        "images_total": total_imgs,
        "images_with_detections": imgs_with_dets,
        "total_detections": total_dets,
        "detection_rate_pct": round(imgs_with_dets / total_imgs * 100, 1) if total_imgs else 0.0,
        "avg_dets_per_image": round(total_dets / total_imgs, 2) if total_imgs else 0.0,
        "total_tiles_processed": total_tiles_all,
        "avg_tiles_per_image": round(total_tiles_all / total_imgs, 1) if total_imgs else 0.0,
        "per_class": {
            class_names.get(c, f"class_{c}"): per_class[c]
            for c in sorted(per_class)
        },
        "per_class_full": {
            class_names.get(c, f"class_{c}"): {
                "count": per_class.get(c, 0),
                "pct_of_total": round(per_class.get(c, 0) / total_dets * 100, 1) if total_dets else 0.0,
            }
            for c in range(n_classes)
        },
        "timing": {
            "total_s": round(t_total, 2),
            "total_formatted": format_elapsed(t_total),
            "avg_ms_per_image": round(t_total / total_imgs * 1000, 1) if total_imgs else 0.0,
            "fps_image": round(total_imgs / t_total, 1) if t_total > 0 else 0.0,
            "fps_tile": round(total_tiles_all / t_total, 1) if t_total > 0 else 0.0,
        },
        "output_dir": str(output_dir.resolve()),
        "source_dirs": [str(Path(d).resolve()) for d in (source_dirs or [])],
        "bad_images_count": len(bad_images) if bad_images else 0,
        "bad_images": bad_images or [],
        "cascade": {
            "enabled": cascade is not None and cascade.enabled,
            "input_detections": total_cascade_input,
            "rejected": total_cascade_rejected,
            "rejection_rate_pct": (
                round(total_cascade_rejected / total_cascade_input * 100, 1)
                if total_cascade_input > 0 else 0.0
            ),
            "available_classes": (
                cascade.available_classes
                if cascade is not None else []
            ),
        },
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
# 报告输出
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: Dict) -> None:
    """在终端打印格式化的推理报告."""
    cls_list = report["classes"]
    per_class = report["per_class"]
    per_class_full = report["per_class_full"]
    timing = report["timing"]

    print()
    print("=" * 64)
    print("  推 理 结 果 报 告 (Tile-based)")
    print("=" * 64)
    print(f"  模型              : {report['model_name']}")
    print(f"  模式              : {report.get('mode', 'tiled')}")
    print(f"  时间              : {report['date'][:19]}")
    print(f"  输出目录          : {report['output_dir']}")
    print(f"  {'─'*56}")
    print(f"  处理原图数        : {report['images_total']}")
    print(f"  总切分数          : {report.get('total_tiles_processed', '?')}")
    print(f"  检测到目标的图片  : {report['images_with_detections']} "
          f"({report['detection_rate_pct']}%)")
    print(f"  总检测框数        : {report['total_detections']}")
    print(f"  平均每张检测数    : {report['avg_dets_per_image']}")
    if report.get("bad_images_count", 0) > 0:
        print(f"  跳过损坏文件      : {report['bad_images_count']}")
    print(f"  {'─'*56}")
    print(f"  处理耗时          : {timing['total_formatted']}")
    print(f"  平均每张原图耗时  : {timing.get('avg_ms_per_image', '?')} ms")
    fps_tile = timing.get("fps_tile", 0)
    print(f"  切片吞吐量        : {fps_tile} tiles/s")
    print(f"  {'─'*56}")

    # Cascade 分类器统计
    cascade_info = report.get("cascade", {})
    if cascade_info.get("enabled"):
        print(f"  Cascade FP 抑制   : 已启用")
        print(f"    送入分类器检测数: {cascade_info.get('input_detections', 0)}")
        print(f"    拒绝 (FP) 数    : {cascade_info.get('rejected', 0)}")
        print(f"    拒绝率          : {cascade_info.get('rejection_rate_pct', 0)}%")
        print(f"  {'─'*56}")

    # 按数量降序排列的逐类统计
    if per_class:
        max_count = max(per_class.values())
        print(f"\n  {'类别':12s}  {'检测数':>6s}  {'占比':>6s}  {'分布图'}")
        print(f"  {'─'*12}  {'─'*6}  {'─'*6}  {'─'*30}")
        for cls_id in sorted(per_class_full, key=lambda c: per_class_full[c]["count"], reverse=True):
            info = per_class_full[cls_id]
            name = (cls_id if isinstance(cls_id, str)
                    else cls_list.get(int(cls_id), f"class_{cls_id}"))
            count = info["count"]
            pct = info["pct_of_total"]
            bar_len = int(count / max(max_count, 1) * 30)
            bar = "█" * bar_len + ("▏" if count > 0 and bar_len == 0 else "")
            print(f"  {name:12s}  {count:6d}  {pct:5.1f}%  {bar}")
    else:
        print(f"\n  (未检测到任何目标)")

    print(f"\n  {'='*64}")
    print(f"  报告文件:")
    print(f"    {report['output_dir']}/report.json")
    print(f"    {report['output_dir']}/report.txt")
    print(f"  {'='*64}\n")


def save_report(report: Dict, output_dir: Path) -> None:
    """将报告保存为 JSON 和 TXT 文件."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON 报告 ──────────────────────────────────────────────────────
    json_path = output_dir / "report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 报告已保存: {json_path}")

    # ── TXT 报告 ───────────────────────────────────────────────────────
    txt_path = output_dir / "report.txt"
    cls_list = report["classes"]
    per_class = report["per_class"]
    per_class_full = report["per_class_full"]
    timing = report["timing"]

    lines = [
        "=" * 64,
        "  推 理 结 果 报 告 (Tile-based)",
        "=" * 64,
        "",
        f"  模型:       {report['model_name']}",
        f"  模式:       {report.get('mode', 'tiled')}",
        f"  时间:       {report['date'][:19]}",
        f"  配置:       conf={report['config']['conf']}, iou={report['config']['iou']}, "
        f"tile={report['config']['tile_size']}, overlap={report['config']['overlap']}",
        f"  类别数:     {report['num_classes']}",
        f"  类别:       {list(cls_list.values())}",
        "",
        f"  处理原图数:        {report['images_total']}",
        f"  总切分数:          {report.get('total_tiles_processed', '?')}",
        f"  检测到目标的图片:  {report['images_with_detections']} "
        f"({report['detection_rate_pct']}%)",
        f"  总检测框数:        {report['total_detections']}",
        f"  平均每张检测数:    {report['avg_dets_per_image']}",
    ]
    if report.get("bad_images_count", 0) > 0:
        lines.append(f"  跳过损坏文件:      {report['bad_images_count']}")
    lines += [
        "",
        f"  处理耗时:          {timing['total_formatted']}",
        f"  平均每张原图耗时:  {timing.get('avg_ms_per_image', '?')} ms",
        f"  切片吞吐量:        {timing.get('fps_tile', 0)} tiles/s",
    ]

    cascade_info = report.get("cascade", {})
    if cascade_info.get("enabled"):
        lines += [
            "",
            f"  Cascade FP 抑制:   已启用",
            f"    送入分类器:      {cascade_info.get('input_detections', 0)}",
            f"    拒绝 (FP):       {cascade_info.get('rejected', 0)}",
            f"    拒绝率:          {cascade_info.get('rejection_rate_pct', 0)}%",
        ]
    lines.append("")

    if per_class:
        lines += [
            f"  {'类别':12s}  {'检测数':>6s}  {'占比':>6s}",
            f"  {'─'*12}  {'─'*6}  {'─'*6}",
        ]
        for cls_id in sorted(per_class_full, key=lambda c: per_class_full[c]["count"], reverse=True):
            info = per_class_full[cls_id]
            name = (cls_id if isinstance(cls_id, str)
                    else cls_list.get(int(cls_id), f"class_{cls_id}"))
            lines.append(f"  {name:12s}  {info['count']:6d}  {info['pct_of_total']:5.1f}%")
    else:
        lines.append("  (未检测到任何目标)")

    lines += [
        "",
        f"  输出目录: {report['output_dir']}",
        "=" * 64,
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ TXT 报告已保存: {txt_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "一键推理 — 7.25 训练最优 7 类模型对 Defect_dataset 全部图片执行 Tile-based 推理\n"
            "每张 5120×5120 原图被切分为 1280×1280 重叠切片后逐片推理, NMS 合并结果.\n"
            "模型: YOLO11s-EMA-SimAM (Stage 4 best), mAP50=0.622\n"
            "类别: VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/run_inference.py\n"
            "  python scripts/run_inference.py --conf 0.4 --device 0\n"
            "  python scripts/run_inference.py --model weights/stage5_calibrated.pt --conf 0.3\n"
            "  python scripts/run_inference.py --cascade\n"
            "  python scripts/run_inference.py --cascade --cascade-threshold 0.6"
        ),
    )

    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"模型权重路径 (默认: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data", type=str, nargs="+", default=DEFAULT_DATA_DIRS,
        help="图片目录列表 (默认: train + val 目录)",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"置信度阈值 (默认: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--iou", type=float, default=DEFAULT_IOU,
        help=f"NMS IoU 阈值 (默认: {DEFAULT_IOU})",
    )
    parser.add_argument(
        "--tile-size", type=int, default=DEFAULT_TILE_SIZE,
        help=f"切片尺寸 px (默认: {DEFAULT_TILE_SIZE}, 与训练分辨率一致)",
    )
    parser.add_argument(
        "--overlap", type=float, default=DEFAULT_OVERLAP,
        help=f"切片重叠比例 (默认: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--device", type=str, default=DEFAULT_DEVICE,
        help=f"CUDA 设备号, 'cpu' 表示 CPU (默认: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_BASE,
        help=f"输出根目录 (默认: {DEFAULT_OUTPUT_BASE})",
    )

    # ── Cascade 分类器参数 ────────────────────────────────────────────
    parser.add_argument(
        "--cascade", action="store_true", default=False,
        help="启用 cascade 分类器进行 FP 抑制 (需要 weights/classifier_*.pt)",
    )
    parser.add_argument(
        "--cascade-weights-dir", type=str, default=DEFAULT_CASCADE_WEIGHTS_DIR,
        help=f"分类器权重目录 (默认: {DEFAULT_CASCADE_WEIGHTS_DIR})",
    )
    parser.add_argument(
        "--cascade-threshold", type=float, default=DEFAULT_CASCADE_THRESHOLD,
        help=f"分类器拒绝置信度阈值 (默认: {DEFAULT_CASCADE_THRESHOLD})",
    )

    args = parser.parse_args()

    # ── 参数校验 ──────────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n  ✗ 模型文件不存在: {model_path.resolve()}")
        print(f"    请确认 --model 参数正确, 可用模型见 weights/ 目录.\n")
        sys.exit(1)

    # ── 收集图片 ──────────────────────────────────────────────────────
    print("\n  ▸ 正在收集图片...")
    image_paths = collect_images(args.data)
    if not image_paths:
        print("\n  ✗ 未找到任何图片! 请检查 --data 参数.\n")
        sys.exit(1)
    print(f"  ▸ 已发现 {len(image_paths)} 张图片 (已按文件名去重)")

    # ── 预验证图片可读性 ──────────────────────────────────────────────
    print("\n  ▸ 正在验证图片可读性...")
    image_paths, bad_images = validate_images(image_paths)
    if not image_paths:
        print("\n  ✗ 所有图片均无法读取! 请检查数据集.\n")
        sys.exit(1)
    print(f"  ▸ 可读图片: {len(image_paths)}, 跳过损坏: {len(bad_images)}")

    # ── 准备输出目录 ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp

    # ── 初始化 Cascade 分类器 (可选) ─────────────────────────────────
    cascade = None
    if args.cascade:
        from subway_defect.pipeline.cascade import CascadeClassifier
        cascade = CascadeClassifier(
            weights_dir=args.cascade_weights_dir,
            device=args.device,
            confidence_threshold=args.cascade_threshold,
            enabled=True,
        )
        print(f"\n  ▸ Cascade 分类器已初始化:")
        print(f"    已加载: {cascade.available_classes}")
        if cascade.failed_classes:
            print(f"    ⚠ 加载失败: {cascade.failed_classes}")

    # ── 执行推理 ──────────────────────────────────────────────────────
    try:
        report = run_inference_tiled(
            model_path=str(model_path),
            image_paths=image_paths,
            conf=args.conf,
            iou=args.iou,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=args.device,
            output_dir=output_dir,
            source_dirs=args.data,
            bad_images=bad_images,
            cascade=cascade,
        )
    except Exception as e:
        print(f"\n  ✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ── 输出报告 ──────────────────────────────────────────────────────
    print_report(report)
    save_report(report, Path(report["output_dir"]))


if __name__ == "__main__":
    main()
