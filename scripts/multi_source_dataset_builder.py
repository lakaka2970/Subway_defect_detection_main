#!/usr/bin/env python3
"""
AutoDL Multi-Source Dataset Builder for Subway Defect Detection.

Scans ``/root/autodl-pub`` for publicly available datasets, downloads missing
ones from official sources, converts every dataset to unified YOLO format with
``generic_defect`` single-class labels, and organizes the final directory tree
ready for multi-source pretraining.

Usage (run on AutoDL instance)::

    # Phase 1 — scan only, see what's available
    python scripts/multi_source_dataset_builder.py --scan-only

    # Phase 2 — full build (scan + download + convert + merge)
    python scripts/multi_source_dataset_builder.py

    # Phase 2 with custom output root
    python scripts/multi_source_dataset_builder.py --output data/multi_datasets

    # Build only specific datasets
    python scripts/multi_source_dataset_builder.py --datasets deeppcb neu_det gc10_det

    # Skip download (use only what's in autodl-pub)
    python scripts/multi_source_dataset_builder.py --no-download

    # Dry-run: print plan without executing
    python scripts/multi_source_dataset_builder.py --dry-run

Directory structure created::

    data/multi_datasets/
    ├── public/
    │   ├── coco/                  # symlink or copy from autodl-pub
    │   ├── neu_det/
    │   │   ├── images/train/      # YOLO-format train images
    │   │   ├── images/val/
    │   │   ├── labels/train/      # YOLO-format labels (generic_defect)
    │   │   └── labels/val/
    │   ├── gc10_det/
    │   ├── deeppcb/
    │   ├── tt100k/                # optional
    │   ├── insulator_defect/      # optional
    │   └── mvtec_or_visa/         # optional
    ├── mixed_pretrain/
    │   ├── images/train/          # symlinks to all public train images
    │   ├── images/val/            # symlinks to all public val images
    │   ├── labels/train/          # symlinks to all public train labels
    │   ├── labels/val/
    │   └── data.yaml              # nc:1, names:["generic_defect"]
    ├── subway_raw/                # (populated separately)
    └── subway_crops/              # (populated separately)

Requirements:
    - Python 3.8+
    - pyyaml, tqdm, requests, Pillow, opencv-python (cv2)
    - Running on AutoDL instance with /root/autodl-pub accessible
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

# ── Optional imports with graceful fallback ──────────────────────────
try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


# ==========================================================================
# Constants
# ==========================================================================

AUTODL_PUB = Path("/root/autodl-pub")
DEFAULT_OUTPUT = Path("data/multi_datasets")
SEED = 42
MIN_BBOX_AREA_PX = 8       # drop boxes smaller than this (px²)
MIN_BBOX_SIDE_PX = 2       # drop boxes narrower/shorter than this

# All datasets merged to this single class
GENERIC_DEFECT_CLASS = "generic_defect"
TINY_OBJECT_CLASS = "tiny_object"

# ── Colour map for progress output ────────────────────────────────────
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def _colour(text: str, code: str) -> str:
    """Wrap *text* in ANSI colour codes (no-op if stdout is not a TTY)."""
    if sys.stdout.isatty():
        return f"{code}{text}{C_RESET}"
    return text


def ok(msg: str) -> str:   return _colour(f"  [OK] {msg}", C_GREEN)
def warn(msg: str) -> str: return _colour(f"  [WARN] {msg}", C_YELLOW)
def fail(msg: str) -> str: return _colour(f"  [FAIL] {msg}", C_RED)
def info(msg: str) -> str: return _colour(f"  [INFO] {msg}", C_CYAN)


# ==========================================================================
# Dataset registry — every dataset we know how to handle
# ==========================================================================

class DatasetSpec:
    """Metadata and handlers for one public dataset."""

    def __init__(
        self,
        key: str,
        name: str,
        autodl_pub_globs: List[str],
        download_urls: List[str],
        download_method: str,  # "direct", "kaggle", "roboflow", "github"
        format_desc: str,
        num_classes: int,
        class_names: List[str],
        priority: int,  # 1=highest
        enabled: bool = True,
        target_class: str = GENERIC_DEFECT_CLASS,
        notes: str = "",
    ):
        self.key = key
        self.name = name
        self.autodl_pub_globs = autodl_pub_globs
        self.download_urls = download_urls
        self.download_method = download_method
        self.format_desc = format_desc
        self.num_classes = num_classes
        self.class_names = class_names
        self.priority = priority
        self.enabled = enabled
        self.target_class = target_class
        self.notes = notes


# Every dataset the plan recommends, in priority order.
DATASET_SPECS: Dict[str, DatasetSpec] = {
    "coco": DatasetSpec(
        key="coco",
        name="COCO 2017",
        autodl_pub_globs=["coco*", "COCO*", "coco2017*", "mscoco*"],
        download_urls=["https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"],
        download_method="direct",
        format_desc="COCO JSON → YOLO (via Ultralytics converter or labels zip)",
        num_classes=80,
        class_names=[],  # COCO has 80 — we only use its pretrained weights, not labels
        priority=0,  # base — not merged into generic_defect
        enabled=True,
        target_class="",  # COCO is used for base pretrain, not merged
        notes="Only YOLO labels are needed; images referenced from autodl-pub",
    ),
    "deeppcb": DatasetSpec(
        key="deeppcb",
        name="DeepPCB",
        autodl_pub_globs=["DeepPCB*", "deeppcb*", "PCB*"],
        download_urls=[
            "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip",
        ],
        download_method="github",  # git clone https://github.com/tangsanli5201/DeepPCB.git
        format_desc="Template-test image pairs + 4-corner polygon TXT → YOLO bbox",
        num_classes=6,
        class_names=["open", "short", "mousebite", "spur", "pin_hole", "spurious_copper"],
        priority=1,
        enabled=False,  # 2026-06-27: 放弃 DeepPCB — PCB 电路板缺陷与金属件表面缺陷视觉特征差异大
        notes="[DEPRECATED] PCB regular-structure defects. Replaced by KolektorSDD2+RSDDs. "
               "1,500 pairs (1,000 train + 500 test), 640×640 px. "
               "Verified: GitHub repo active, last confirmed 2025.",
    ),
    "gc10_det": DatasetSpec(
        key="gc10_det",
        name="GC10-DET",
        autodl_pub_globs=["GC10*", "gc10*", "gc10-det*"],
        download_urls=[
            "https://www.kaggle.com/datasets/alex000kim/gc10det",  # primary — confirmed 2024-2025
            "https://github.com/lvxiaoming2019/GC10-DET-Metallic-Surface-Defect-Matasets",
        ],
        download_method="kaggle",  # kaggle CLI: kaggle datasets download -d alex000kim/gc10det
        format_desc="VOC XML annotations → YOLO (generic_defect=class 0)",
        num_classes=10,
        class_names=[
            "punching", "welding_line", "crescent_gap", "water_spot",
            "oil_spot", "silk_spot", "inclusion", "rolled_pit",
            "scratch", "crease",
        ],
        priority=1,
        enabled=True,
        notes="3,570 grayscale images, 2048×1000 px, 10 metal surface defect classes. "
               "Kaggle mirror (alex000kim) confirmed active. GitHub original (lvxiaoming2019) as backup.",
    ),
    "neu_det": DatasetSpec(
        key="neu_det",
        name="NEU-DET (NEU Surface Defect)",
        autodl_pub_globs=["NEU*", "neu*", "NEU-DET*", "NEU_DET*", "neu_det*"],
        download_urls=[
            "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
            "https://www.kaggle.com/datasets/zy12345/neudet",  # backup Kaggle mirror
        ],
        download_method="kaggle",
        format_desc="VOC XML annotations (200×200 grayscale) → YOLO (generic_defect=class 0)",
        num_classes=6,
        class_names=[
            "rolled-in_scale", "patches", "crazing", "pitted_surface",
            "inclusion", "scratches",
        ],
        priority=1,
        enabled=True,
        notes="1,800 grayscale images (300/class), 200×200 px. Auto-converted to 3-channel RGB. "
               "Official site (faculty.neu.edu.cn) may be slow; Kaggle mirrors confirmed working. "
               "IEEE DataPort DOI:10.21227/j84r-f770 as tertiary backup.",
    ),
    "tt100k": DatasetSpec(
        key="tt100k",
        name="TT100K (Tsinghua-Tencent 100K)",
        autodl_pub_globs=["TT100K*", "tt100k*", "traffic*"],
        download_urls=[
            "http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip",
        ],
        download_method="direct",  # 18 GB zip from Tsinghua official
        format_desc="Custom JSON annotations → YOLO (tiny_object=class 0)",
        num_classes=221,
        class_names=[],  # 221 traffic sign classes — all merged to tiny_object
        priority=2,
        enabled=False,  # OPTIONAL — only needed for P2 head pretraining
        target_class=TINY_OBJECT_CLASS,
        notes="Optional P2 small-object pretraining. 100K images, 30K+ instances. "
               "Ultralytics has built-in auto-download (TT100K.yaml). "
               "Direct download ~18 GB from cg.cs.tsinghua.edu.cn. "
               "License: CC BY-NC 2.0 (non-commercial).",
    ),
    "insulator_defect": DatasetSpec(
        key="insulator_defect",
        name="Insulator Defect Detection (Roboflow)",
        autodl_pub_globs=["insulator*", "Insulator*"],
        download_urls=[
            "https://universe.roboflow.com/pourya-shojaei/insatance-segmentation-insulator",
        ],
        download_method="roboflow",
        format_desc="Roboflow YOLOv8 export → relabel all classes to generic_defect",
        num_classes=1,  # merged
        class_names=["insulator"],
        priority=2,
        enabled=False,  # OPTIONAL — needs Roboflow API key
        notes="Near-domain supplement (~917 insulator images). "
               "Requires Roboflow API key. Export as YOLOv8 format. "
               "Annotation quality varies — use as supplementary only.",
    ),
    "mvtec_ad": DatasetSpec(
        key="mvtec_ad",
        name="MVTec AD (Anomaly Detection)",
        autodl_pub_globs=["MVTec*", "mvtec*", "mvtec_ad*", "mvtec-ad*"],
        download_urls=[
            "https://service.tib.eu/ldmservice/dataset/mvtec-anomaly-detection--ad--dataset",
        ],
        download_method="direct",
        format_desc="Segmentation mask → connectedComponents → bbox → YOLO (generic_defect)",
        num_classes=1,
        class_names=["anomaly"],
        priority=3,
        enabled=False,  # OPTIONAL — mask-to-bbox quality varies
        notes="15 object/texture classes, 5,000+ images. Mask→bbox conversion needed. "
               "TIB LDM (DOI:10.57702/59u9s21i) as download source. "
               "anomalib library can auto-download. Only anomaly images used.",
    ),
    "visa": DatasetSpec(
        key="visa",
        name="VisA (Visual Anomaly)",
        autodl_pub_globs=["VisA*", "visa*"],
        download_urls=[
            "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar",
        ],
        download_method="direct",  # aws s3 cp --no-sign-request (no AWS account needed)
        format_desc="Segmentation mask → bbox → YOLO (generic_defect)",
        num_classes=1,
        class_names=["anomaly"],
        priority=3,
        enabled=False,
        notes="12 subsets, 10,821 images (1,200 anomalous). Has PCB subsets. "
               "AWS Open Data — no account needed: "
               "aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar ./",
    ),
    "rsdds": DatasetSpec(
        key="rsdds",
        name="RSDDs (Rail Surface Defect Dataset)",
        autodl_pub_globs=["RSDDs*", "rsdds*", "RSDD*", "rail*"],
        download_urls=[
            "https://pan.baidu.com/s/1z62NfRTAROVYSARk41Gz-A?pwd=u7zi",  # Baidu Pan (primary)
            "https://ieee-dataport.org/documents/rsdds-rail-surface-defect-dataset",  # DOI:10.21227/qtv6-n081
        ],
        download_method="manual",  # Baidu Pan requires manual download + extraction
        format_desc="Binary mask PNG → bbox → YOLO (generic_defect). "
                     "Type-I 160×1000, Type-II 55×1250, padded to square.",
        num_classes=1,
        class_names=["rail_defect"],
        priority=2,
        enabled=True,
        notes="195 images (67 Type-I + 128 Type-II), rail surface cracks/pores/wear. "
               "Download from Baidu Pan (pwd: u7zi) or IEEE DataPort. "
               "Expected structure: data{1,2}/{train,test}/images/ + masks/. "
               "Images padded to square for YOLO training. "
               "Original: http://icn.bjtu.edu.cn/Visint/resources/RSDDs.aspx",
    ),
    "kolektor_sdd2": DatasetSpec(
        key="kolektor_sdd2",
        name="KolektorSDD2",
        autodl_pub_globs=["Kolektor*", "kolektor*", "KSDD2*", "ksdd2*"],
        download_urls=[
            "https://huggingface.co/datasets/sizhkhy/kolektor_sdd2",  # HuggingFace mirror
            "https://www.vicos.si/resources/kolektorsdd2/",           # Official (registration)
            "https://beta.hyper.ai/datasets/21545",                   # hyper.ai mirror
        ],
        download_method="manual",  # Requires registration or manual download from HuggingFace
        format_desc="Segmentation mask _GT.png → bbox → YOLO (generic_defect). "
                     "Flat train/test dirs, ~230×630 px, padded to square.",
        num_classes=1,
        class_names=["defect"],
        priority=2,
        enabled=True,
        notes="356 defective + 2,979 defect-free images, ~230×630 px. "
               "Masks use _GT.png suffix alongside original images. "
               "Expected structure: {train,test}/{id}.png + {id}_GT.png. "
               "Images padded to square for YOLO training. "
               "License: CC BY-NC-SA 4.0.",
    ),
}


# ==========================================================================
# Utility helpers
# ==========================================================================

def _requests_session() -> "requests.Session":
    """Return a requests.Session with retry logic."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "AutoDL-Dataset-Builder/1.0"})
    return session


def _download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download *url* to *dest* with a progress bar. Returns True on success."""
    if requests is None:
        print(fail("requests not installed — cannot download."))
        return False
    try:
        session = _requests_session()
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        if tqdm is not None:
            pbar = tqdm(
                total=total, unit="B", unit_scale=True,
                desc=desc or f"Downloading {dest.name}",
            )
        else:
            pbar = None

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))
        if pbar is not None:
            pbar.close()
        return True
    except Exception as exc:
        print(fail(f"Download failed: {url} → {exc}"))
        return False


def _run(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a subprocess command. Returns True on success."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(fail(f"Command failed (exit {exc.returncode}): {' '.join(cmd)}"))
        return False
    except FileNotFoundError:
        print(fail(f"Command not found: {cmd[0]}"))
        return False


def _is_html_content(file_path: Path) -> bool:
    """Check if a file looks like an HTML page rather than actual dataset content."""
    if not file_path.is_file():
        return False
    try:
        with open(file_path, "rb") as f:
            head = f.read(512).lstrip()
        return bool(head) and (
            head.startswith(b"<!DOCTYPE")
            or head.startswith(b"<html")
            or head.startswith(b"<HTML")
        )
    except (OSError, PermissionError):
        return False


def _validate_download_dir(dest_dir: Path, key: str = "") -> bool:
    """Check that a download directory contains actual dataset content, not just HTML pages.

    Returns True if the directory looks valid, False if it should be re-downloaded.
    An empty directory is considered invalid.
    """
    if not dest_dir.exists() or not dest_dir.is_dir():
        return False

    # Collect all files up to 5 levels deep.
    # (kagglehub downloads often wrap the dataset in a container directory
    #  like NEU-DET/train/images/file.jpg — 4 levels from dest_dir.)
    def _collect_files(base: Path, depth: int = 0, max_depth: int = 5) -> List[Path]:
        files: List[Path] = []
        if depth > max_depth or not base.is_dir():
            return files
        for entry in base.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_file():
                files.append(entry)
            elif entry.is_dir() and depth < max_depth:
                files.extend(_collect_files(entry, depth + 1, max_depth))
        return files

    all_files = _collect_files(dest_dir)

    if not all_files:
        return False  # empty directory

    # If all files are HTML or very small text files → invalid
    non_html: List[Path] = []
    for f in all_files:
        if not _is_html_content(f):
            non_html.append(f)

    if not non_html:
        # Check if the HTML-looking files are actually large (could be mis-detected)
        large_files = [f for f in all_files if f.stat().st_size > 100_000]  # >100KB
        if not large_files:
            tag = f"[{key}] " if key else ""
            print(fail(f"{tag}Download dir contains only small HTML/text files — not a valid dataset"))
            return False

    return True


def _autodl_pub_datasets() -> List[str]:
    """List dataset keys found in autodl-pub (for pre-download check)."""
    if not AUTODL_PUB.exists():
        return []
    found: List[str] = []
    for entry in sorted(AUTODL_PUB.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            if any(entry.iterdir()):
                found.append(entry.name.lower())
    return found


# ── KaggleHub optional import ──────────────────────────────────────
try:
    import kagglehub  # type: ignore[import-untyped]
except ImportError:
    kagglehub = None  # type: ignore[assignment]


def _find_in_autodl_pub(globs: List[str]) -> Optional[Path]:
    """Return the first path matching any glob under AUTODL_PUB."""
    if not AUTODL_PUB.exists():
        return None
    for pattern in globs:
        for match in sorted(AUTODL_PUB.glob(pattern)):
            if match.is_dir():
                return match
    # Also check top-level contents for partial name matches
    for pattern in globs:
        base = pattern.rstrip("*").rstrip("-").rstrip("_").lower()
        for child in AUTODL_PUB.iterdir():
            if child.is_dir() and base in child.name.lower():
                return child
    return None


def _make_symlink_tree(src_dir: Path, dst_dir: Path, pattern: str = "*") -> int:
    """Symlink all files matching *pattern* from *src_dir* into *dst_dir*.
    Returns the count of files linked.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            link = dst_dir / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
            count += 1
    return count


def _copy_or_symlink_files(
    src_dir: Path, dst_dir: Path, pattern: str = "*", prefer_symlink: bool = True
) -> int:
    """Copy or symlink files from *src_dir* to *dst_dir*. Returns file count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(src_dir.glob(pattern)):
        if not f.is_file():
            continue
        dst = dst_dir / f.name
        if dst.exists():
            continue
        if prefer_symlink:
            try:
                dst.symlink_to(f.resolve())
            except OSError:
                shutil.copy2(f, dst)
        else:
            shutil.copy2(f, dst)
        count += 1
    return count


# ==========================================================================
# Format converters — each dataset → YOLO txt
# ==========================================================================

def _voc_xml_to_yolo_boxes(xml_path: Path, class_map: Dict[str, int],
                           img_w: int, img_h: int) -> List[str]:
    """Parse a PASCAL VOC XML file, return YOLO-format label lines."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    lines: List[str] = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None or name.text is None:
            continue
        cls_name = name.text.strip()
        cls_id = class_map.get(cls_name, 0)  # unknown classes → generic_defect (class 0)

        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        try:
            xmin = float(bbox.find("xmin").text)   # type: ignore[union-attr]
            ymin = float(bbox.find("ymin").text)   # type: ignore[union-attr]
            xmax = float(bbox.find("xmax").text)   # type: ignore[union-attr]
            ymax = float(bbox.find("ymax").text)   # type: ignore[union-attr]
        except (AttributeError, ValueError):
            continue

        # Convert to YOLO normalized format
        w_box = xmax - xmin
        h_box = ymax - ymin
        if w_box < MIN_BBOX_SIDE_PX or h_box < MIN_BBOX_SIDE_PX:
            continue
        if w_box * h_box < MIN_BBOX_AREA_PX:
            continue

        x_center = (xmin + xmax) / 2.0 / img_w
        y_center = (ymin + ymax) / 2.0 / img_h
        w_norm = w_box / img_w
        h_norm = h_box / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        if w_norm <= 0 or h_norm <= 0:
            continue

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    return lines


def _mask_to_bboxes(mask_path: Path) -> List[Tuple[int, int, int, int]]:
    """Convert a binary segmentation mask to a list of bounding boxes.

    Uses OpenCV connectedComponentsWithStats. Returns list of (x, y, w, h).
    """
    if cv2 is None:
        print(warn("opencv-python not installed — cannot convert masks."))
        return []

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8,
    )

    bboxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):  # skip background (label 0)
        x, y, w, h, area = stats[i]
        # ---- filter tiny fragments ----
        if area < MIN_BBOX_AREA_PX:
            continue
        if w < MIN_BBOX_SIDE_PX or h < MIN_BBOX_SIDE_PX:
            continue
        bboxes.append((int(x), int(y), int(w), int(h)))

    # Merge very close boxes (within 3px) to reduce over-fragmentation
    bboxes = _merge_nearby_bboxes(bboxes, distance_threshold=3)

    return bboxes


def _merge_nearby_bboxes(
    bboxes: List[Tuple[int, int, int, int]], distance_threshold: int = 3
) -> List[Tuple[int, int, int, int]]:
    """Merge bounding boxes that are within *distance_threshold* pixels."""
    if len(bboxes) <= 1:
        return bboxes

    # Simple greedy merge: sort by x, then merge overlapping/nearby
    bboxes = sorted(bboxes, key=lambda b: (b[0], b[1]))
    merged: List[Tuple[int, int, int, int]] = []
    used = [False] * len(bboxes)

    for i, b1 in enumerate(bboxes):
        if used[i]:
            continue
        x1, y1, w1, h1 = b1
        changed = True
        while changed:
            changed = False
            for j, b2 in enumerate(bboxes):
                if used[j] or i == j:
                    continue
                x2, y2, w2, h2 = b2
                # Expand b1 by threshold and check overlap
                ex1 = (x1 - distance_threshold, y1 - distance_threshold,
                       w1 + 2 * distance_threshold, h1 + 2 * distance_threshold)
                ex2 = (x2, y2, w2, h2)
                if _rects_overlap(ex1, ex2):
                    # Merge
                    nx = min(x1, x2)
                    ny = min(y1, y2)
                    nx2 = max(x1 + w1, x2 + w2)
                    ny2 = max(y1 + h1, y2 + h2)
                    x1, y1 = nx, ny
                    w1, h1 = nx2 - nx, ny2 - ny
                    used[j] = True
                    changed = True
        merged.append((x1, y1, w1, h1))
        used[i] = True

    return merged


def _rects_overlap(r1: Tuple[int, int, int, int],
                   r2: Tuple[int, int, int, int]) -> bool:
    """Check if two axis-aligned rectangles overlap."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)


def _bboxes_to_yolo_lines(
    bboxes: List[Tuple[int, int, int, int]],
    img_w: int, img_h: int, cls_id: int = 0,
) -> List[str]:
    """Convert pixel bboxes to YOLO-format label lines (class 0 = generic_defect)."""
    lines: List[str] = []
    for x, y, w, h in bboxes:
        x_center = (x + w / 2.0) / img_w
        y_center = (y + h / 2.0) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        # Clamp
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))
        if w_norm <= 0 or h_norm <= 0:
            continue
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
    return lines


# ==========================================================================
# Per-dataset converters (public interface used by orchestrator)
# ==========================================================================

def convert_coco(autodl_pub_path: Path, output_dir: Path) -> bool:
    """Handle COCO by downloading/using YOLO-format labels.

    COCO images are left in autodl-pub; we only create YOLO labels in our tree.
    """
    spec = DATASET_SPECS["coco"]
    out_img_train = output_dir / "images" / "train2017"
    out_lbl_train = output_dir / "labels" / "train2017"
    out_img_val = output_dir / "images" / "val2017"
    out_lbl_val = output_dir / "labels" / "val2017"

    # Try to find COCO labels in the project's existing downloaded location
    project_coco_labels = Path("subway_yolo/cfg/datasets")
    coco_yaml = project_coco_labels / "coco.yaml"
    if coco_yaml.exists():
        print(ok(f"COCO YAML found at {coco_yaml} — project already has COCO config"))
        # The project already references COCO via its dataset YAML files
        # We just note its location for the mixed pretrain config
        return True

    # Download YOLO-format COCO labels if needed
    labels_zip = output_dir / "_downloads" / "coco2017labels.zip"
    if not labels_zip.exists():
        print(info("Downloading COCO YOLO-format labels..."))
        if not _download_file(spec.download_urls[0], labels_zip, "COCO labels"):
            return False

    # Unzip labels
    labels_dir = output_dir / "_downloads" / "coco_labels"
    if not labels_dir.exists():
        labels_dir.mkdir(parents=True, exist_ok=True)
        import zipfile
        with zipfile.ZipFile(labels_zip, "r") as zf:
            zf.extractall(labels_dir)
        print(ok("COCO labels extracted"))

    print(ok(f"COCO ready (images from {autodl_pub_path}, labels in {output_dir})"))
    return True


def convert_voc_dataset(
    spec: DatasetSpec,
    src_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
) -> bool:
    """Convert a VOC-XML-annotated dataset to YOLO format.

    This handles NEU-DET, GC10-DET, and similar datasets.

    Directory structure expected under *src_dir*::

        src_dir/
        ├── Annotations/   (or annotations/)
        │   ├── 001.xml
        │   └── ...
        ├── JPEGImages/    (or images/)
        │   ├── 001.jpg
        │   └── ...
        └── ImageSets/     (optional — train/val split)
            └── Main/
                ├── train.txt
                └── val.txt

    Output::

        output_dir/
        ├── images/train/
        ├── images/val/
        ├── labels/train/
        └── labels/val/
    """
    key = spec.key

    # Locate annotation directories (may be multiple in pre-split datasets like NEU-DET)
    _ANNO_CANDIDATES = ["Annotations", "annotations", "ANNOT", "xmls", "xml", "lable"]
    _IMG_CANDIDATES = ["JPEGImages", "images", "IMAGES", "imgs", "img", "JPEG"]

    anno_dirs: List[Path] = []
    for cand in _ANNO_CANDIDATES:
        p = src_dir / cand
        if p.is_dir():
            anno_dirs.append(p)
    # Search up to 2 levels deep (handles: lable/, train/annotations/, NEU-DET/train/annotations/)
    if not anno_dirs:
        for sub in sorted(src_dir.iterdir()):
            if not sub.is_dir():
                continue
            for cand in _ANNO_CANDIDATES:
                p = sub / cand
                if p.is_dir():
                    anno_dirs.append(p)
            if not anno_dirs:
                for sub2 in sorted(sub.iterdir()):
                    if sub2.is_dir():
                        for cand in _ANNO_CANDIDATES:
                            p = sub2 / cand
                            if p.is_dir():
                                anno_dirs.append(p)
    if not anno_dirs:
        print(fail(f"[{key}] Cannot find annotations directory under {src_dir}"))
        # Diagnostic: show what's actually in the source directory
        items = sorted(src_dir.iterdir()) if src_dir.exists() else []
        if items:
            listing = ", ".join(
                f"{p.name}{'/' if p.is_dir() else ''}" for p in items[:15]
            )
            print(info(f"[{key}] Source dir contents: [{listing}]"))
        else:
            print(info(f"[{key}] Source dir is empty — download may have failed. "
                       f"Check {src_dir.parent} for any downloaded archives."))
        return False

    img_dirs: List[Path] = []
    for cand in _IMG_CANDIDATES:
        p = src_dir / cand
        if p.is_dir():
            img_dirs.append(p)
    # Search up to 2 levels deep
    if not img_dirs:
        for sub in sorted(src_dir.iterdir()):
            if not sub.is_dir():
                continue
            for cand in _IMG_CANDIDATES:
                p = sub / cand
                if p.is_dir():
                    img_dirs.append(p)
            if not img_dirs:
                for sub2 in sorted(sub.iterdir()):
                    if sub2.is_dir():
                        for cand in _IMG_CANDIDATES:
                            p = sub2 / cand
                            if p.is_dir():
                                img_dirs.append(p)
    if not img_dirs:
        img_dirs = [src_dir]  # fallback: search images from src_dir

    print(info(f"[{key}] Annotations: {len(anno_dirs)} dir(s), Images: {len(img_dirs)} dir(s)"))

    # Build class map (all → generic_defect = class 0)
    class_map = {name: 0 for name in spec.class_names}

    # Find image files from all image directories
    img_files: Dict[str, Path] = {}
    for img_dir in img_dirs:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
            for f in img_dir.glob(ext):
                if f.stem not in img_files:
                    img_files[f.stem] = f
            # Also search recursively up to 2 levels (handles nested structures)
            for f in img_dir.glob(f"*/{ext}"):
                if f.stem not in img_files:
                    img_files[f.stem] = f
            for f in img_dir.glob(f"*/*/{ext}"):
                if f.stem not in img_files:
                    img_files[f.stem] = f

    # Find annotation files from all annotation directories
    # Track which split (parent dir name) each stem belongs to
    xml_files: Dict[str, Path] = {}
    stem_split: Dict[str, str] = {}  # stem → "train" / "val" / ""
    _SPLIT_NAMES = {"train", "val", "validation", "test"}
    for ad in anno_dirs:
        parent = ad.parent.name.lower()
        split_tag = ""
        if parent in _SPLIT_NAMES:
            split_tag = "val" if parent == "validation" else parent
        for f in ad.glob("*.xml"):
            if f.stem not in xml_files:
                xml_files[f.stem] = f
                stem_split[f.stem] = split_tag
        # Also search recursively
        for f in ad.glob("*/*.xml"):
            if f.stem not in xml_files:
                xml_files[f.stem] = f
                stem_split[f.stem] = split_tag

    # Match images to annotations
    matched_stems: List[str] = []
    for stem in sorted(xml_files):
        if stem in img_files:
            matched_stems.append(stem)
        else:
            # Try case-insensitive match
            img_lower = {k.lower(): k for k in img_files}
            if stem.lower() in img_lower:
                matched_stems.append(img_lower[stem.lower()])

    if not matched_stems:
        print(fail(f"[{key}] No matched image-annotation pairs found!"))
        print(f"     Image stems (first 10): {list(img_files.keys())[:10]}")
        print(f"     XML stems (first 10):   {list(xml_files.keys())[:10]}")
        return False

    print(info(f"[{key}] Matched {len(matched_stems)} image-annotation pairs"))

    # Train/val split — priority: 1) directory-based split  2) ImageSets  3) random
    train_stems: Set[str] = set()
    val_stems: Set[str] = set()

    # 1) Check if we already have a directory-based split (e.g. NEU-DET train/validation/)
    train_from_dir = {s for s in matched_stems if stem_split.get(s) == "train"}
    val_from_dir = {s for s in matched_stems if stem_split.get(s) in ("val", "validation", "test")}
    if train_from_dir and val_from_dir:
        train_stems = train_from_dir
        val_stems = val_from_dir
        print(info(f"[{key}] Using directory-based split: {len(train_stems)} train, {len(val_stems)} val"))
    else:
        # 2) Try ImageSets/Main/{train,val,test}.txt
        imageset_dir = src_dir / "ImageSets" / "Main"
        train_txt = imageset_dir / "train.txt"
        val_txt = imageset_dir / "val.txt"
        test_txt = imageset_dir / "test.txt"

        if train_txt.exists() and val_txt.exists():
            train_stems = set(train_txt.read_text().splitlines())
            val_stems = set(val_txt.read_text().splitlines())
            train_stems &= set(matched_stems)
            val_stems &= set(matched_stems)
            print(info(f"[{key}] Using ImageSets split: {len(train_stems)} train, {len(val_stems)} val"))
        elif train_txt.exists():
            all_split = set(train_txt.read_text().splitlines()) & set(matched_stems)
            all_list = sorted(all_split)
            random.seed(SEED)
            random.shuffle(all_list)
            split_idx = int(len(all_list) * (1 - val_ratio))
            train_stems = set(all_list[:split_idx])
            val_stems = set(all_list[split_idx:])
        else:
            # 3) Random split
            random.seed(SEED)
            shuffled = sorted(matched_stems)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * (1 - val_ratio))
            train_stems = set(shuffled[:split_idx])
            val_stems = set(shuffled[split_idx:])
            print(info(f"[{key}] Random split: {len(train_stems)} train, {len(val_stems)} val"))

    # Create output directories
    for split, stems in (("train", train_stems), ("val", val_stems)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        lbl_count = 0

        for stem in sorted(stems):
            img_path = img_files[stem]
            xml_path = xml_files.get(stem)
            if xml_path is None:
                # Try case-insensitive
                xml_lower = {k.lower(): k for k in xml_files}
                xml_path = xml_files.get(xml_lower.get(stem.lower(), ""))

            if xml_path is None:
                continue

            # Get image dimensions
            try:
                if Image is not None:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                elif cv2 is not None:
                    im = cv2.imread(str(img_path))
                    if im is not None:
                        img_h, img_w = im.shape[:2]
                    else:
                        continue
                else:
                    # Parse from XML
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    size = root.find("size")
                    if size is not None:
                        img_w = int(size.find("width").text)   # type: ignore[union-attr]
                        img_h = int(size.find("height").text)  # type: ignore[union-attr]
                    else:
                        img_w, img_h = 1024, 1024
            except Exception:
                continue

            # Convert labels
            lines = _voc_xml_to_yolo_boxes(xml_path, class_map, img_w, img_h)

            # Handle grayscale images (NEU-DET) — convert to 3-channel
            dst_img = out_img_dir / f"{stem}.jpg"
            if not dst_img.exists():
                try:
                    if Image is not None:
                        with Image.open(img_path) as im:
                            if im.mode == "L" or im.mode == "1":
                                im = im.convert("RGB")
                            im.save(dst_img, quality=95)
                    elif cv2 is not None:
                        im = cv2.imread(str(img_path))
                        if im is not None:
                            if len(im.shape) == 2:
                                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                            cv2.imwrite(str(dst_img), im)
                        else:
                            shutil.copy2(img_path, dst_img)
                    else:
                        shutil.copy2(img_path, dst_img)
                except Exception:
                    shutil.copy2(img_path, dst_img)
            img_count += 1

            # Write YOLO label
            dst_lbl = out_lbl_dir / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines) + "\n" if lines else "\n", encoding="utf-8")
            if lines:
                lbl_count += 1

        box_count = sum(
            1 for l in out_lbl_dir.glob("*.txt")
            for _ in l.read_text().strip().splitlines() if _.strip()
        )
        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes"))

    return True


def convert_deeppcb(src_dir: Path, output_dir: Path, val_ratio: float = 0.2) -> bool:
    """Convert DeepPCB dataset to YOLO format.

    DeepPCB structure::

        DeepPCB/
        ├── PCBData/
        │   ├── group0001/
        │   │   ├── 0001.jpg       # template (defect-free)
        │   │   ├── 0001_not.jpg   # test image (may have defects)
        │   │   └── 0001_not.txt   # defect annotations
        │   ├── group0002/
        │   ...

    Annotation format (from tangsanli5201/DeepPCB)::

        Each line: x1,y1,x2,y2,x3,y3,x4,y4,type
        (4 corner points of defect polygon + defect type index 1-6)

    Defect types: 1=open, 2=short, 3=mousebite, 4=spur, 5=pin_hole, 6=spurious_copper
    """
    spec = DATASET_SPECS["deeppcb"]
    key = spec.key

    # Find PCBData directory
    pcb_data = None
    for cand in ["PCBData", "pcbdata", "data"]:
        p = src_dir / cand
        if p.is_dir():
            pcb_data = p
            break
    if pcb_data is None:
        # Check if src_dir itself contains group* dirs
        groups = list(src_dir.glob("group*"))
        if groups:
            pcb_data = src_dir
    if pcb_data is None:
        print(fail(f"[{key}] Cannot find PCBData directory under {src_dir}"))
        return False

    # Collect all groups
    groups = sorted([d for d in pcb_data.iterdir() if d.is_dir() and d.name.startswith("group")])
    if not groups:
        # Check for flat structure: template/test pairs directly
        test_images = sorted(pcb_data.glob("*_not.jpg"))
        if test_images:
            groups = [pcb_data]  # treat as single group

    if not groups:
        print(fail(f"[{key}] No group directories or test images found in {pcb_data}"))
        return False

    print(info(f"[{key}] Found {len(groups)} groups in {pcb_data}"))

    # Collect all (image, annotation) pairs
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for grp in groups:
        # Pattern 1: *_test.jpg recursively (handles nested dirs like groupX/#####/#####_test.jpg)
        #   Annotation is in sibling *_not/ directory: "00041000_test.jpg" → "00041000.txt"
        for test_img in sorted(grp.rglob("*_test.jpg")):
            # Derive annotation stem by stripping "_test" suffix
            anno_stem = test_img.stem
            if anno_stem.endswith("_test"):
                anno_stem = anno_stem[:-5]
            txt_file = None
            # First try same-directory annotation
            candidate = test_img.with_name(f"{anno_stem}.txt")
            if candidate.exists():
                txt_file = candidate
            else:
                # Search in sibling *_not directories
                for not_dir in sorted(grp.glob("*_not")):
                    if not_dir.is_dir():
                        candidate = not_dir / f"{anno_stem}.txt"
                        if candidate.exists():
                            txt_file = candidate
                            break
            pairs.append((test_img, txt_file))
        # Pattern 2: *_not.jpg files directly in group dir (backward compat, some mirrors)
        for test_img in sorted(grp.glob("*_not.jpg")):
            txt_file = test_img.with_suffix(".txt")
            if (test_img, txt_file if txt_file.exists() else None) not in pairs:
                pairs.append((test_img, txt_file if txt_file.exists() else None))

    if not pairs:
        print(fail(f"[{key}] No test images (*_not.jpg or *_test.jpg) found"))
        return False

    print(info(f"[{key}] Found {len(pairs)} test-image/annotation pairs"))

    # Split groups (not images) to prevent leakage
    random.seed(SEED)
    group_to_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]] = defaultdict(list)
    for img_path, txt_path in pairs:
        grp_key = img_path.parent.name
        group_to_pairs[grp_key].append((img_path, txt_path))

    group_keys = sorted(group_to_pairs.keys())
    random.shuffle(group_keys)
    split_idx = int(len(group_keys) * (1 - val_ratio))
    train_groups = set(group_keys[:split_idx])
    val_groups = set(group_keys[split_idx:])

    # Write outputs
    for split, grp_set in (("train", train_groups), ("val", val_groups)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0

        for grp_key in sorted(grp_set):
            for img_path, txt_path in group_to_pairs[grp_key]:
                # Get image dimensions
                try:
                    if Image is not None:
                        with Image.open(img_path) as im:
                            img_w, img_h = im.size
                    elif cv2 is not None:
                        im = cv2.imread(str(img_path))
                        img_h, img_w = im.shape[:2] if im is not None else (0, 0)
                    else:
                        continue
                except Exception:
                    continue

                # Copy image
                dst_img = out_img_dir / f"{img_path.parent.name}_{img_path.name}"
                if not dst_img.exists():
                    shutil.copy2(img_path, dst_img)
                img_count += 1

                # Convert annotations
                lines: List[str] = []
                if txt_path is not None:
                    try:
                        raw = txt_path.read_text().strip()
                        for line in raw.splitlines():
                            # Try comma-separated (old 9-value polygon format)
                            parts = line.strip().split(",")
                            if len(parts) < 5:
                                # Try space/tab-separated (DeepPCB GitHub format: x1 y1 x2 y2 type)
                                parts = line.strip().split()
                            # Handle 5-value format: x1 y1 x2 y2 type (bounding box corners)
                            if len(parts) == 5:
                                try:
                                    pts = [float(p) for p in parts[:4]]
                                    cls_type = int(parts[4])
                                except ValueError:
                                    continue
                                xs = pts[0::2]
                                ys = pts[1::2]
                                xmin, xmax = min(xs), max(xs)
                                ymin, ymax = min(ys), max(ys)
                            # Handle 9-value format: x1,y1,x2,y2,x3,y3,x4,y4,type (polygon)
                            elif len(parts) >= 9:
                                try:
                                    pts = [float(p) for p in parts[:8]]
                                    cls_type = int(parts[8])
                                except ValueError:
                                    continue
                                xs = pts[0::2]
                                ys = pts[1::2]
                                xmin, xmax = min(xs), max(xs)
                                ymin, ymax = min(ys), max(ys)
                            else:
                                continue
                            w_box, h_box = xmax - xmin, ymax - ymin

                            if w_box < MIN_BBOX_SIDE_PX or h_box < MIN_BBOX_SIDE_PX:
                                continue
                            if w_box * h_box < MIN_BBOX_AREA_PX:
                                continue

                            x_center = (xmin + xmax) / 2.0 / img_w
                            y_center = (ymin + ymax) / 2.0 / img_h
                            w_norm = w_box / img_w
                            h_norm = h_box / img_h

                            x_center = max(0.0, min(1.0, x_center))
                            y_center = max(0.0, min(1.0, y_center))
                            w_norm = max(0.0, min(1.0, w_norm))
                            h_norm = max(0.0, min(1.0, h_norm))

                            if w_norm > 0 and h_norm > 0:
                                lines.append(
                                    f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
                                )
                    except Exception as exc:
                        print(warn(f"[{key}] Error parsing {txt_path}: {exc}"))

                # Write label file
                dst_lbl = out_lbl_dir / f"{img_path.parent.name}_{img_path.stem}.txt"
                dst_lbl.write_text(
                    "\n".join(lines) + "\n" if lines else "\n", encoding="utf-8",
                )
                box_count += len(lines)

        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes (generic_defect)"))

    return True


def convert_tt100k(src_dir: Path, output_dir: Path, val_ratio: float = 0.2) -> bool:
    """Convert TT100K to YOLO format (all → tiny_object class).

    TT100K structure::

        TT100K/
        ├── data/
        │   ├── train/
        │   │   └── *.jpg
        │   ├── test/
        │   │   └── *.jpg
        │   └── ...
        ├── annotations.json   (or annotations/)
        └── ...

    The annotation JSON contains::

        {"imgs": {"id": {"path": "...", "objects": [{"bbox": {"xmin":...,}, "category":...}]}}}
    """
    spec = DATASET_SPECS["tt100k"]
    key = spec.key

    # Find annotation JSON
    anno_json = None
    for cand in ["annotations.json", "train.json", "annotations/train.json",
                  "annotations/annotations.json"]:
        p = src_dir / cand
        if p.exists():
            anno_json = p
            break

    if anno_json is None:
        print(fail(f"[{key}] Cannot find annotation JSON under {src_dir}"))
        return False

    # Find image directories
    img_dirs: List[Path] = []
    for cand in ["data", "images", "train", "JPEGImages"]:
        p = src_dir / cand
        if p.is_dir():
            # Flatten: collect all jpg files
            img_dirs.append(p)

    if not img_dirs:
        print(warn(f"[{key}] No image directories found — will look recursively"))
        img_dirs = [src_dir]

    print(info(f"[{key}] Parsing annotation JSON: {anno_json}"))

    with open(anno_json, encoding="utf-8") as f:
        annos = json.load(f)

    imgs_data = annos.get("imgs", {})
    print(info(f"[{key}] {len(imgs_data)} annotated images in JSON"))

    # Build image path lookup
    img_lookup: Dict[str, Path] = {}
    for d in img_dirs:
        for f in d.rglob("*.jpg"):
            img_lookup[f.stem] = f
            img_lookup[f.name] = f

    # Process
    all_stems: List[str] = []
    for img_id, img_info in imgs_data.items():
        path_str = img_info.get("path", img_id)
        stem = Path(path_str).stem
        all_stems.append(stem)

    random.seed(SEED)
    shuffled = sorted(set(all_stems))
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    train_stems = set(shuffled[:split_idx])
    val_stems = set(shuffled[split_idx:])

    for split, stems in (("train", train_stems), ("val", val_stems)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0

        for stem in sorted(stems):
            # Find image and annotation
            img_info = None
            img_id = None
            for iid, iinfo in imgs_data.items():
                if Path(iinfo.get("path", iid)).stem == stem:
                    img_info = iinfo
                    img_id = iid
                    break

            if img_info is None:
                continue

            # Find image file
            img_path = None
            for d in img_dirs:
                for f in d.rglob(f"{stem}.*"):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        img_path = f
                        break
                if img_path:
                    break
            if img_path is None:
                continue

            try:
                if Image is not None:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                else:
                    continue
            except Exception:
                continue

            # Copy image
            dst_img = out_img_dir / f"{stem}.jpg"
            if not dst_img.exists():
                try:
                    if Image is not None:
                        with Image.open(img_path) as im:
                            im.convert("RGB").save(dst_img, quality=95)
                    else:
                        shutil.copy2(img_path, dst_img)
                except Exception:
                    shutil.copy2(img_path, dst_img)
            img_count += 1

            # Convert objects to YOLO bboxes (all → class 0 = tiny_object)
            lines: List[str] = []
            for obj in img_info.get("objects", []):
                bbox = obj.get("bbox", {})
                xmin = float(bbox.get("xmin", 0))
                ymin = float(bbox.get("ymin", 0))
                xmax = float(bbox.get("xmax", 0))
                ymax = float(bbox.get("ymax", 0))

                w_box = xmax - xmin
                h_box = ymax - ymin
                if w_box < MIN_BBOX_SIDE_PX or h_box < MIN_BBOX_SIDE_PX:
                    continue
                if w_box * h_box < MIN_BBOX_AREA_PX:
                    continue

                xc = (xmin + xmax) / 2.0 / img_w
                yc = (ymin + ymax) / 2.0 / img_h
                wn = w_box / img_w
                hn = h_box / img_h
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                wn = max(0.0, min(1.0, wn))
                hn = max(0.0, min(1.0, hn))

                if wn > 0 and hn > 0:
                    lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            dst_lbl = out_lbl_dir / f"{stem}.txt"
            dst_lbl.write_text("\n".join(lines) + "\n" if lines else "\n", encoding="utf-8")
            box_count += len(lines)

        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes (tiny_object)"))

    return True


def convert_mask_dataset(
    spec: DatasetSpec, src_dir: Path, output_dir: Path, val_ratio: float = 0.2,
) -> bool:
    """Convert a mask-based anomaly dataset (MVTec AD, VisA) to YOLO format.

    Expected structure::

        src_dir/
        ├── <object_class>/
        │   ├── train/good/    (defect-free images)
        │   ├── test/
        │   │   ├── <defect_type>/
        │   │   │   ├── 000.png       (defect image)
        │   │   │   └── 000_mask.png  (segmentation mask)
        │   │   └── good/            (defect-free test images — skip)
        │   └── ground_truth/
        │       └── <defect_type>/
        │           └── 000_mask.png
        └── ...
    """
    key = spec.key

    if cv2 is None:
        print(fail(f"[{key}] opencv-python required for mask conversion"))
        return False

    # Find object-class subdirectories
    obj_dirs = [d for d in src_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not obj_dirs:
        print(fail(f"[{key}] No object-class directories found under {src_dir}"))
        return False

    print(info(f"[{key}] Found {len(obj_dirs)} object classes: "
               f"{[d.name for d in obj_dirs[:10]]}"))

    # Collect (image, mask) pairs from test anomalies
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for obj_dir in obj_dirs:
        test_dir = obj_dir / "test"
        if not test_dir.is_dir():
            continue

        for defect_dir in test_dir.iterdir():
            if not defect_dir.is_dir():
                continue
            if defect_dir.name == "good":
                continue  # defect-free test images — skip for pretraining

            for img_file in sorted(defect_dir.glob("*.png")):
                if img_file.name.endswith("_mask.png"):
                    continue
                # Look for mask
                mask_file = defect_dir / f"{img_file.stem}_mask.png"
                if not mask_file.exists():
                    # Check ground_truth
                    gt_dir = obj_dir / "ground_truth" / defect_dir.name
                    mask_file = gt_dir / f"{img_file.stem}_mask.png"
                pairs.append((img_file, mask_file if mask_file.exists() else None))

    if not pairs:
        print(fail(f"[{key}] No defect image/mask pairs found"))
        return False

    print(info(f"[{key}] Found {len(pairs)} defect image/mask pairs"))

    # Split by object class to prevent leakage
    # Group pairs by object class
    cls_to_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]] = defaultdict(list)
    for img_path, mask_path in pairs:
        cls_name = img_path.parent.parent.parent.name
        cls_to_pairs[cls_name].append((img_path, mask_path))

    cls_keys = sorted(cls_to_pairs.keys())
    random.seed(SEED)
    random.shuffle(cls_keys)
    split_idx = max(1, int(len(cls_keys) * (1 - val_ratio)))
    train_classes = set(cls_keys[:split_idx])
    val_classes = set(cls_keys[split_idx:])

    for split, cls_set in (("train", train_classes), ("val", val_classes)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0

        for cls_name in sorted(cls_set):
            for img_path, mask_path in cls_to_pairs[cls_name]:
                # Get image dimensions
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                img_h, img_w = im.shape[:2]

                # Copy image
                safe_name = f"{cls_name}_{img_path.parent.name}_{img_path.name}"
                dst_img = out_img_dir / safe_name
                if not dst_img.exists():
                    cv2.imwrite(str(dst_img), im)
                img_count += 1

                # Convert mask to bboxes
                lines: List[str] = []
                if mask_path is not None:
                    bboxes = _mask_to_bboxes(mask_path)
                    lines = _bboxes_to_yolo_lines(bboxes, img_w, img_h, cls_id=0)

                # Write label
                dst_lbl = out_lbl_dir / f"{safe_name.stem}.txt"
                dst_lbl.write_text("\n".join(lines) + "\n" if lines else "\n", encoding="utf-8")
                box_count += len(lines)

        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes"))

    return True


def convert_rsdds(
    spec: DatasetSpec, src_dir: Path, output_dir: Path, val_ratio: float = 0.2,
) -> bool:
    """Convert RSDDs (Rail Surface Defect Dataset) to YOLO format.

    Supports two layout variants:

    **Format A — preprocessed (Baidu Pan / IEEE DataPort):**

        src_dir/
        ├── data1/                      # Type-I (express rail, 160×1000 px)
        │   ├── train/images/*.jpg + masks/*.png
        │   └── test/images/*.jpg + masks/*.png
        └── data2/                      # Type-II (heavy rail, 55×1250 px)
            ├── train/images/*.jpg + masks/*.png
            └── test/images/*.jpg + masks/*.png

    **Format B — original GitHub (neu-rail-rsdds/rsdds):**

        src_dir/
        ├── images/*.jpg                # colour images (113, BMP→JPG)
        ├── masks/*.png                 # binary defect masks (same stem)
        └── Ground Truth/*.png (alt)    # alternative mask location
            │   ├── images/*.jpg
            │   └── masks/*.png
            └── test/
                ├── images/*.jpg
                └── masks/*.png

    Notes:
        - Images are non-square (160×1000 or 55×1250).  They are padded to
          square with grey (114,114,114) before writing.
        - Masks are binary PNGs (white=defect, black=background).
        - The original train/test split is preserved within each subset
          (data1/data2), then all images are pooled and re-split at the dataset
          level by subset to prevent leakage.
    """
    key = spec.key

    if cv2 is None:
        print(fail(f"[{key}] opencv-python required for RSDDs conversion"))
        return False

    # ── Detect format: Format A (data{1,2}/...) vs Format B (flat images/ + masks/) ──
    data_dirs = sorted([
        d for d in src_dir.iterdir()
        if d.is_dir() and d.name.startswith("data") and not d.name.startswith(".")
    ])

    subset_pairs: Dict[str, List[Tuple[Path, Optional[Path]]]] = defaultdict(list)
    total_pairs = 0

    if data_dirs:
        # ── Format A: data{1,2}/{train,test}/images/ + masks/ ──
        print(info(f"[{key}] Format A — {len(data_dirs)} subset(s): "
                   f"{[d.name for d in data_dirs]}"))

        for data_dir in data_dirs:
            subset_name = data_dir.name
            for split_tag in ("train", "test"):
                img_dir = data_dir / split_tag / "images"
                mask_dir = data_dir / split_tag / "masks"

                if not img_dir.is_dir():
                    continue

                img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
                for ext in img_exts:
                    for img_path in sorted(img_dir.glob(ext)):
                        mask_path = mask_dir / f"{img_path.stem}.png"
                        if not mask_path.exists():
                            for m_ext in (".jpg", ".bmp"):
                                alt = mask_dir / f"{img_path.stem}{m_ext}"
                                if alt.exists():
                                    mask_path = alt
                                    break
                        subset_pairs[subset_name].append(
                            (img_path, mask_path if mask_path.exists() else None)
                        )
                        total_pairs += 1

        # Split by subset (data1/data2) to prevent leakage
        subset_names = sorted(subset_pairs.keys())
        random.seed(SEED)
        random.shuffle(subset_names)
        split_idx = max(1, int(len(subset_names) * (1 - val_ratio)))
        train_subsets = set(subset_names[:split_idx])
        val_subsets = set(subset_names[split_idx:])
        use_subset_split = True

    else:
        # ── Format B: flat images/ + masks/ (original GitHub structure) ──
        img_dir = src_dir / "images"
        mask_candidates = [
            src_dir / "masks",
            src_dir / "Ground Truth",
            src_dir / "ground_truth" / "Ground Truth",
        ]
        mask_dir = None
        for mc in mask_candidates:
            if mc.is_dir():
                mask_dir = mc
                break

        if not img_dir.is_dir():
            # Check for rsdds_flat/images/ (post-extraction reorganised)
            img_dir = src_dir / "rsdds_flat" / "images"
            for mc in [
                src_dir / "rsdds_flat" / "masks",
                src_dir / "ground_truth" / "Ground Truth",
            ]:
                if mc.is_dir():
                    mask_dir = mc
                    break

        if not img_dir.is_dir():
            print(fail(f"[{key}] No images/ directory found — "
                       f"expected images/ + masks/ or data{1,2}/..."))
            return False

        print(info(f"[{key}] Format B — flat images/ + masks/ structure"))

        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
        for ext in img_exts:
            for img_path in sorted(img_dir.glob(ext)):
                mask_path = None
                if mask_dir is not None:
                    mask_path = mask_dir / f"{img_path.stem}.png"
                    if not mask_path.exists():
                        for m_ext in (".jpg", ".bmp"):
                            alt = mask_dir / f"{img_path.stem}{m_ext}"
                            if alt.exists():
                                mask_path = alt
                                break
                subset_pairs["rsdds"].append(
                    (img_path, mask_path if (mask_path and mask_path.exists()) else None)
                )
                total_pairs += 1

        # For flat format, split individual images randomly
        shuffled = sorted(subset_pairs.get("rsdds", []),
                          key=lambda x: x[0].stem)
        random.seed(SEED)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
        subset_pairs["_train"] = shuffled[:split_idx]
        subset_pairs["_val"] = shuffled[split_idx:]
        train_subsets = {"_train"}
        val_subsets = {"_val"}
        use_subset_split = True

    if total_pairs == 0:
        print(fail(f"[{key}] No defect image/mask pairs found"))
        return False

    print(info(f"[{key}] Found {total_pairs} defect image/mask pairs"))

    for split, sub_set in (("train", train_subsets), ("val", val_subsets)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0

        for sub_name in sorted(sub_set):
            for img_path, mask_path in subset_pairs[sub_name]:
                im = cv2.imread(str(img_path))
                if im is None:
                    continue
                img_h, img_w = im.shape[:2]

                # Pad to square (RSDDs images are non-square: 160×1000 or 55×1250)
                max_side = max(img_h, img_w)
                pad_bottom = max_side - img_h
                pad_right = max_side - img_w
                im_padded = cv2.copyMakeBorder(
                    im, 0, pad_bottom, 0, pad_right,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114),
                )

                # Safe filename: subset_split_stem
                safe_name = f"{sub_name}_{img_path.parent.parent.name}_{img_path.stem}"
                dst_img = out_img_dir / f"{safe_name}.jpg"
                if not dst_img.exists():
                    cv2.imwrite(str(dst_img), im_padded,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                img_count += 1

                # Convert mask to bboxes (relative to padded image)
                lines: List[str] = []
                if mask_path is not None and mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Pad mask to match padded image
                        mask_padded = cv2.copyMakeBorder(
                            mask, 0, pad_bottom, 0, pad_right,
                            cv2.BORDER_CONSTANT, value=0,
                        )
                        bboxes = _mask_to_bboxes_from_array(mask_padded)
                        lines = _bboxes_to_yolo_lines(
                            bboxes, max_side, max_side, cls_id=0,
                        )

                dst_lbl = out_lbl_dir / f"{safe_name}.txt"
                dst_lbl.write_text(
                    "\n".join(lines) + "\n" if lines else "\n",
                    encoding="utf-8",
                )
                box_count += len(lines)

        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes"))

    return True


def _mask_to_bboxes_from_array(mask: "np.ndarray") -> List[Tuple[int, int, int, int]]:
    """Like _mask_to_bboxes but takes a numpy array instead of a file path."""
    import numpy as np
    if mask is None or mask.max() == 0:
        return []
    binary = (mask > 127).astype(np.uint8) * 255
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    bboxes: List[Tuple[int, int, int, int]] = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 4:  # skip single-pixel noise
            continue
        bboxes.append((int(x), int(y), int(w), int(h)))
    return bboxes


def convert_kolektor_sdd2(
    spec: DatasetSpec, src_dir: Path, output_dir: Path, val_ratio: float = 0.2,
) -> bool:
    """Convert KolektorSDD2 to YOLO format.

    Expected structure::

        src_dir/
        ├── train/
        │   ├── 10000.png              # defect / defect-free image
        │   ├── 10000_GT.png           # mask (only for defect images, _GT suffix)
        │   ├── 10001.png
        │   └── ...
        └── test/
            ├── {id}.png
            ├── {id}_GT.png
            └── ...

    Notes:
        - 356 defective + 2,979 defect-free images (~230×630 px).
        - Defect-free images have NO corresponding _GT mask.
        - Images are non-square; padded to square with grey (114,114,114).
        - The original train/test split is preserved.
    """
    key = spec.key

    if cv2 is None:
        print(fail(f"[{key}] opencv-python required for KolektorSDD2 conversion"))
        return False

    # Collect (image, mask) pairs
    pairs: List[Tuple[Path, Optional[Path]]] = []

    for split_tag in ("train", "test"):
        split_dir = src_dir / split_tag
        if not split_dir.is_dir():
            print(warn(f"[{key}] {split_tag}/ directory not found under {src_dir}"))
            continue

        for img_file in sorted(split_dir.glob("*.png")):
            if img_file.name.endswith("_GT.png"):
                continue  # skip mask files

            # Look for mask: {stem}_GT.png
            mask_file = split_dir / f"{img_file.stem}_GT.png"
            pairs.append((img_file, mask_file if mask_file.exists() else None))

    if not pairs:
        print(fail(f"[{key}] No images found under {src_dir}"))
        return False

    n_with_mask = sum(1 for _, m in pairs if m is not None)
    n_without = len(pairs) - n_with_mask
    print(info(f"[{key}] Found {len(pairs)} images "
               f"({n_with_mask} defective, {n_without} defect-free)"))

    # Split: shuffle all images, keep val_ratio for validation
    random.seed(SEED)
    shuffled = sorted(pairs, key=lambda x: x[0].stem)
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    train_pairs = shuffled[:split_idx]
    val_pairs = shuffled[split_idx:]

    for split, pair_list in (("train", train_pairs), ("val", val_pairs)):
        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0

        for img_path, mask_path in pair_list:
            im = cv2.imread(str(img_path))
            if im is None:
                continue
            img_h, img_w = im.shape[:2]

            # Pad to square (KolektorSDD2 images are ~230×630 px)
            max_side = max(img_h, img_w)
            pad_bottom = max_side - img_h
            pad_right = max_side - img_w
            im_padded = cv2.copyMakeBorder(
                im, 0, pad_bottom, 0, pad_right,
                cv2.BORDER_CONSTANT, value=(114, 114, 114),
            )

            safe_name = f"{img_path.parent.name}_{img_path.stem}"
            dst_img = out_img_dir / f"{safe_name}.jpg"
            if not dst_img.exists():
                cv2.imwrite(str(dst_img), im_padded,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_count += 1

            # Convert mask to bboxes
            lines: List[str] = []
            if mask_path is not None and mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_padded = cv2.copyMakeBorder(
                        mask, 0, pad_bottom, 0, pad_right,
                        cv2.BORDER_CONSTANT, value=0,
                    )
                    bboxes = _mask_to_bboxes_from_array(mask_padded)
                    lines = _bboxes_to_yolo_lines(
                        bboxes, max_side, max_side, cls_id=0,
                    )

            dst_lbl = out_lbl_dir / f"{safe_name}.txt"
            dst_lbl.write_text(
                "\n".join(lines) + "\n" if lines else "\n",
                encoding="utf-8",
            )
            box_count += len(lines)

        print(ok(f"[{key}] {split}: {img_count} images, {box_count} boxes"))

    return True


def convert_roboflow_dataset(
    spec: DatasetSpec, src_dir: Path, output_dir: Path,
) -> bool:
    """Handle a Roboflow dataset (usually already in YOLO format)."""
    key = spec.key

    # Roboflow YOLO export structure:
    #   src_dir/
    #   ├── data.yaml
    #   ├── train/images/
    #   ├── train/labels/
    #   ├── valid/images/
    #   ├── valid/labels/
    #   └── test/images/ (optional)

    # Check if already in YOLO format
    data_yaml = src_dir / "data.yaml"
    if not data_yaml.exists():
        # Try one level deeper
        for sub in src_dir.iterdir():
            if sub.is_dir() and (sub / "data.yaml").exists():
                src_dir = sub
                data_yaml = src_dir / "data.yaml"
                break

    if not data_yaml.exists():
        print(fail(f"[{key}] data.yaml not found — not a standard Roboflow YOLO export"))
        print(f"     Contents: {list(src_dir.iterdir())[:10]}")
        return False

    # Rewrite all labels to class 0
    for split in ("train", "valid", "test"):
        lbl_dir = src_dir / split / "labels"
        if not lbl_dir.is_dir():
            continue
        out_split = "val" if split == "valid" else split
        out_img = output_dir / "images" / out_split
        out_lbl = output_dir / "labels" / out_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        img_dir = src_dir / split / "images"
        if img_dir.is_dir():
            img_count = _copy_or_symlink_files(img_dir, out_img, "*.jpg")
            img_count += _copy_or_symlink_files(img_dir, out_img, "*.png")

        lbl_count = 0
        box_count = 0
        for lbl_file in sorted(lbl_dir.glob("*.txt")):
            lines: List[str] = []
            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    # Replace class_id with 0 (generic_defect)
                    parts[0] = "0"
                    lines.append(" ".join(parts))
            dst_file = out_lbl / lbl_file.name
            dst_file.write_text("\n".join(lines) + "\n" if lines else "\n", encoding="utf-8")
            lbl_count += 1
            box_count += len(lines)

        print(ok(f"[{key}] {out_split}: {lbl_count} labels, {box_count} boxes (generic_defect)"))

    return True


# ==========================================================================
# Main orchestrator
# ==========================================================================

class DatasetBuilder:
    """Orchestrates the full multi-source dataset build pipeline."""

    def __init__(
        self,
        output_root: Path = DEFAULT_OUTPUT,
        datasets: Optional[List[str]] = None,
        no_download: bool = False,
        scan_only: bool = False,
        dry_run: bool = False,
        workers: int = 8,
        val_ratio: float = 0.2,
    ):
        self.output_root = Path(output_root)
        self.datasets_filter = datasets  # None = all enabled
        self.no_download = no_download
        self.scan_only = scan_only
        self.dry_run = dry_run
        self.workers = workers
        self.val_ratio = val_ratio

        self.public_dir = self.output_root / "public"
        self.mixed_dir = self.output_root / "mixed_pretrain"
        self.download_dir = self.output_root / "_downloads"

    # ── Phase 1: Scan ──────────────────────────────────────────────
    def scan_autodl_pub(self) -> Dict[str, Optional[Path]]:
        """Scan /root/autodl-pub for each dataset. Returns found paths."""
        print(f"\n{C_BOLD}{'='*60}{C_RESET}")
        print(f"{C_BOLD}  Phase 1: Scanning /root/autodl-pub{C_RESET}")
        print(f"{'='*60}")

        results: Dict[str, Optional[Path]] = {}

        specs = self._enabled_specs()
        for key, spec in specs.items():
            found = _find_in_autodl_pub(spec.autodl_pub_globs)
            if found:
                print(ok(f"{spec.name}: FOUND → {found}"))
            else:
                print(warn(f"{spec.name}: not found in autodl-pub"))
            results[key] = found

        return results

    # ── Phase 2: Download ──────────────────────────────────────────
    def download_missing(self, found: Dict[str, Optional[Path]]) -> Dict[str, Path]:
        """Download datasets not found in autodl-pub. Returns all source paths."""
        print(f"\n{C_BOLD}{'='*60}{C_RESET}")
        print(f"{C_BOLD}  Phase 2: Downloading missing datasets{C_RESET}")
        print(f"{'='*60}")

        all_sources: Dict[str, Path] = {}

        for key, spec in self._enabled_specs().items():
            if key in found and found[key] is not None:
                all_sources[key] = found[key]
                continue

            if self.no_download:
                print(warn(f"[{key}] {spec.name}: skipping download (--no-download)"))
                continue

            print(info(f"[{key}] {spec.name}: downloading..."))

            if spec.download_method == "direct":
                src = self._download_direct(key, spec)
                if src:
                    all_sources[key] = src
            elif spec.download_method == "kaggle":
                src = self._download_kaggle(key, spec)
                if src:
                    all_sources[key] = src
            elif spec.download_method == "roboflow":
                src = self._download_roboflow(key, spec)
                if src:
                    all_sources[key] = src
            elif spec.download_method == "github":
                src = self._download_github(key, spec)
                if src:
                    all_sources[key] = src
            elif spec.download_method == "manual":
                # Manual download — check if user has placed data in _downloads/
                dest_dir = self.download_dir / key
                if dest_dir.exists() and any(dest_dir.iterdir()) and _validate_download_dir(dest_dir, key):
                    print(ok(f"[{key}] Found manually downloaded data at {dest_dir}"))
                    all_sources[key] = dest_dir
                else:
                    if dest_dir.exists() and any(dest_dir.iterdir()):
                        print(warn(f"[{key}] Data at {dest_dir} looks invalid (HTML stub?), treating as missing."))
                    print(info(
                        f"[{key}] {spec.name} requires MANUAL download.\n"
                        f"       Download from: {spec.download_urls[0]}\n"
                        f"       Extract to:    {dest_dir}\n"
                        f"       Expected structure: {spec.format_desc}"
                    ))
            else:
                print(warn(f"[{key}] Unknown download method: {spec.download_method}"))

        return all_sources

    def _download_direct(self, key: str, spec: DatasetSpec) -> Optional[Path]:
        """Download a dataset via direct URL, with content validation."""
        dest_dir = self.download_dir / key

        # ── Check existing download ────────────────────────────────
        if dest_dir.exists() and any(dest_dir.iterdir()):
            if _validate_download_dir(dest_dir, key):
                print(ok(f"[{key}] Already downloaded to {dest_dir}"))
                return dest_dir
            else:
                print(warn(f"[{key}] Previous download was invalid, cleaning up..."))
                shutil.rmtree(dest_dir)

        for url in spec.download_urls:
            fname = Path(urlparse(url).path).name or f"{key}.zip"
            archive_path = self.download_dir / fname

            if not archive_path.exists():
                print(info(f"[{key}] Downloading from {url}"))
                if not _dry_run_guard(self.dry_run):
                    if not _download_file(url, archive_path, f"{key}"):
                        continue
                    # Check if downloaded file is HTML (e.g. Kaggle page without auth)
                    if _is_html_content(archive_path):
                        print(fail(f"[{key}] Downloaded file is an HTML page, not a dataset archive. "
                                   f"The URL may require authentication. Deleting bad download..."))
                        archive_path.unlink()
                        continue
            else:
                # Check existing archive
                if _is_html_content(archive_path):
                    print(warn(f"[{key}] Cached archive is HTML (not a dataset), deleting..."))
                    archive_path.unlink()
                    continue
                print(ok(f"[{key}] Archive already at {archive_path}"))

            # Extract
            if not _dry_run_guard(self.dry_run):
                dest_dir.mkdir(parents=True, exist_ok=True)
                if self._extract_archive(archive_path, dest_dir):
                    if _validate_download_dir(dest_dir, key):
                        print(ok(f"[{key}] Extracted to {dest_dir}"))
                        return dest_dir
                    else:
                        print(fail(f"[{key}] Extraction produced no valid dataset content"))
                        # Don't delete — user may have mixed content

        return None

    def _download_kaggle(self, key: str, spec: DatasetSpec) -> Optional[Path]:
        """Download via Kaggle API, with kagglehub + GitHub fallbacks and content validation."""
        dest_dir = self.download_dir / key

        # ── Check existing download ────────────────────────────────
        if dest_dir.exists() and any(dest_dir.iterdir()):
            if _validate_download_dir(dest_dir, key):
                print(ok(f"[{key}] Already at {dest_dir}"))
                return dest_dir
            else:
                print(warn(f"[{key}] Previous download was invalid (HTML stub), cleaning up..."))
                shutil.rmtree(dest_dir)

        # ── Parse Kaggle slug ──────────────────────────────────────
        kaggle_slug = None
        for url in spec.download_urls:
            m = re.search(r"kaggle\.com/datasets/([^/]+/[^/]+)", url)
            if m:
                kaggle_slug = m.group(1)
                break

        if not kaggle_slug:
            print(fail(f"[{key}] Cannot parse Kaggle dataset slug from URLs"))
            return None

        print(info(f"[{key}] Kaggle dataset: {kaggle_slug}"))
        if _dry_run_guard(self.dry_run):
            return dest_dir

        dest_dir.mkdir(parents=True, exist_ok=True)

        # ── Strategy 1: Kaggle CLI ─────────────────────────────────
        if _run(["kaggle", "datasets", "download", "-d", kaggle_slug,
                  "-p", str(dest_dir), "--unzip"]):
            print(ok(f"[{key}] Downloaded via Kaggle CLI"))
            if _validate_download_dir(dest_dir, key):
                return dest_dir
            else:
                print(warn(f"[{key}] CLI download invalid, cleaning up..."))
                shutil.rmtree(dest_dir)
                dest_dir.mkdir(parents=True, exist_ok=True)

        # ── Strategy 2: kagglehub (no auth needed for public datasets) ──
        if kagglehub is not None:
            try:
                print(info(f"[{key}] Trying kagglehub download..."))
                dl_cache = Path(kagglehub.dataset_download(kaggle_slug))
                if dl_cache.exists():
                    # Copy from kagglehub cache to dest_dir
                    for item in dl_cache.iterdir():
                        dest_item = dest_dir / item.name
                        if item.is_dir():
                            if dest_item.exists():
                                shutil.rmtree(dest_item)
                            shutil.copytree(item, dest_item)
                        else:
                            shutil.copy2(item, dest_item)
                    if _validate_download_dir(dest_dir, key):
                        print(ok(f"[{key}] Downloaded via kagglehub"))
                        return dest_dir
                    else:
                        # Diagnostic: show what was actually downloaded
                        found_items = list(dest_dir.rglob("*"))
                        sample = [str(p.relative_to(dest_dir)) for p in found_items[:20]]
                        print(warn(f"[{key}] kagglehub download invalid. "
                                   f"Contents ({len(found_items)} items): {sample}"))
                        shutil.rmtree(dest_dir)
                        dest_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                print(warn(f"[{key}] kagglehub failed: {exc}"))
        else:
            print(info(f"[{key}] Tip: install kagglehub for simpler Kaggle downloads: pip install kagglehub"))

        # ── Strategy 3: GitHub clone (if GitHub URLs exist) ────────
        github_urls = [u for u in spec.download_urls if "github.com" in u]
        for gh_url in github_urls:
            m = re.search(r"github\.com/([^/]+/[^/]+)", gh_url)
            if not m:
                continue
            repo_path = m.group(1).rstrip("/")
            git_url = f"https://github.com/{repo_path}.git"
            print(info(f"[{key}] Trying GitHub mirror: {git_url}"))
            if _dry_run_guard(self.dry_run):
                return dest_dir
            if _run(["git", "clone", "--depth", "1", git_url, str(dest_dir)]):
                print(ok(f"[{key}] Cloned from GitHub: {git_url}"))
                if _validate_download_dir(dest_dir, key):
                    return dest_dir
                else:
                    print(warn(f"[{key}] GitHub clone invalid, cleaning up..."))
                    shutil.rmtree(dest_dir)
                    dest_dir.mkdir(parents=True, exist_ok=True)
            else:
                print(warn(f"[{key}] git clone failed: {git_url}"))

        # ── Strategy 4: Direct URL (last resort) ────────────────────
        result = self._download_direct(key, spec)
        if result and _validate_download_dir(result, key):
            return result

        # ── All strategies exhausted ───────────────────────────────
        print(fail(f"[{key}] All download methods failed."))
        print(info(f"[{key}] Manual download options:"))
        print(info(f"    1. Install Kaggle CLI and authenticate: pip install kaggle"))
        print(info(f"       Then: kaggle datasets download -d {kaggle_slug} -p {dest_dir} --unzip"))
        print(info(f"    2. Install kagglehub: pip install kagglehub"))
        print(info(f"    3. Download from Kaggle: {spec.download_urls[0]}"))
        print(info(f"       Place extracted files in: {dest_dir}"))
        return None

    def _download_roboflow(self, key: str, spec: DatasetSpec) -> Optional[Path]:
        """Download via Roboflow API."""
        dest_dir = self.download_dir / key
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print(ok(f"[{key}] Already at {dest_dir}"))
            return dest_dir

        print(warn(f"[{key}] Roboflow datasets require manual download or API key."))
        print(info(f"[{key}] Visit: {spec.download_urls[0]}"))
        print(info(f"[{key}] Export as YOLO format and place in: {dest_dir}"))

        dest_dir.mkdir(parents=True, exist_ok=True)
        # Create a README with instructions
        readme = dest_dir / "DOWNLOAD_INSTRUCTIONS.txt"
        if not readme.exists():
            readme.write_text(
                f"Download {spec.name} from Roboflow:\n"
                f"  {spec.download_urls[0]}\n\n"
                f"1. Export in YOLO format\n"
                f"2. Extract to this directory\n"
                f"3. Re-run multi_source_dataset_builder.py\n",
                encoding="utf-8",
            )

        # Check if user already placed data
        if any(dest_dir.glob("*.yaml")) or any(dest_dir.glob("data.yaml")):
            return dest_dir

        return None

    def _download_github(self, key: str, spec: DatasetSpec) -> Optional[Path]:
        """Download from GitHub repository via git clone (preferred) or zip fallback."""
        dest_dir = self.download_dir / key

        # ── Check existing download ────────────────────────────────
        if dest_dir.exists() and any(dest_dir.iterdir()):
            if _validate_download_dir(dest_dir, key):
                print(ok(f"[{key}] Already at {dest_dir}"))
                return dest_dir
            else:
                print(warn(f"[{key}] Previous download was invalid, cleaning up..."))
                shutil.rmtree(dest_dir)

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Try git clone first (preserves directory structure)
        for url in spec.download_urls:
            # Convert zip URL to git URL
            git_url = None
            m = re.search(r"github\.com/([^/]+/[^/]+)", url)
            if m:
                git_url = f"https://github.com/{m.group(1)}.git"

            if git_url and not _dry_run_guard(self.dry_run):
                print(info(f"[{key}] git clone {git_url}"))
                if _run(["git", "clone", "--depth", "1", git_url, str(dest_dir)]):
                    print(ok(f"[{key}] Cloned to {dest_dir}"))
                    if _validate_download_dir(dest_dir, key):
                        return dest_dir
                    else:
                        print(warn(f"[{key}] GitHub clone has no dataset content, cleaning up..."))
                        shutil.rmtree(dest_dir)
                        dest_dir.mkdir(parents=True, exist_ok=True)
                else:
                    print(warn(f"[{key}] git clone failed, trying zip download..."))

        # Fallback to zip download
        return self._download_direct(key, spec)

    @staticmethod
    def _extract_archive(archive_path: Path, dest_dir: Path) -> bool:
        """Extract zip/tar/tar.gz archive to dest_dir."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            if archive_path.suffix == ".zip":
                import zipfile
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extractall(dest_dir)
                return True
            elif ".tar" in archive_path.suffixes or archive_path.suffix in (".tgz", ".tar"):
                import tarfile
                with tarfile.open(archive_path, "r:*") as tf:
                    tf.extractall(dest_dir)  # type: ignore[attr-defined]
                return True
            else:
                # Single file — just copy
                shutil.copy2(archive_path, dest_dir / archive_path.name)
                return True
        except Exception as exc:
            print(fail(f"Extract failed: {archive_path} → {exc}"))
            return False

    # ── Phase 3: Convert ───────────────────────────────────────────
    def convert_all(self, sources: Dict[str, Path]) -> Dict[str, Path]:
        """Convert all datasets to unified YOLO format. Returns output dirs."""
        print(f"\n{C_BOLD}{'='*60}{C_RESET}")
        print(f"{C_BOLD}  Phase 3: Converting datasets to YOLO format{C_RESET}")
        print(f"{'='*60}")

        outputs: Dict[str, Path] = {}

        for key, spec in self._enabled_specs().items():
            if key not in sources:
                continue

            src = sources[key]
            out = self.public_dir / key
            out.mkdir(parents=True, exist_ok=True)

            print(f"\n{C_CYAN}-- Converting {spec.name} --{C_RESET}")

            if _dry_run_guard(self.dry_run):
                outputs[key] = out
                continue

            success = False

            if key == "coco":
                success = convert_coco(src, out)
            elif key in ("neu_det", "gc10_det"):
                success = convert_voc_dataset(spec, src, out, self.val_ratio)
            elif key == "deeppcb":
                success = convert_deeppcb(src, out, self.val_ratio)
            elif key == "tt100k":
                success = convert_tt100k(src, out, self.val_ratio)
            elif key in ("mvtec_ad", "visa"):
                success = convert_mask_dataset(spec, src, out, self.val_ratio)
            elif key == "rsdds":
                success = convert_rsdds(spec, src, out, self.val_ratio)
            elif key == "kolektor_sdd2":
                success = convert_kolektor_sdd2(spec, src, out, self.val_ratio)
            elif key == "insulator_defect":
                success = convert_roboflow_dataset(spec, src, out)
            else:
                # Generic fallback: try VOC first, then mask
                print(info(f"[{key}] No specific converter — trying VOC XML..."))
                success = convert_voc_dataset(spec, src, out, self.val_ratio)

            if success:
                outputs[key] = out
                print(ok(f"{spec.name} conversion complete"))
            else:
                print(fail(f"{spec.name} conversion FAILED"))

        return outputs

    # ── Phase 4: Merge ─────────────────────────────────────────────
    def create_mixed_pretrain(self, outputs: Dict[str, Path]) -> bool:
        """Create mixed_pretrain directory with symlinks and data.yaml."""
        print(f"\n{C_BOLD}{'='*60}{C_RESET}")
        print(f"{C_BOLD}  Phase 4: Creating mixed_pretrain dataset{C_RESET}")
        print(f"{'='*60}")

        mixed = self.mixed_dir

        if _dry_run_guard(self.dry_run):
            print(info(f"[DRY-RUN] Would create: {mixed / 'data.yaml'}"))
            return True
        for sp in ("train", "val"):
            (mixed / "images" / sp).mkdir(parents=True, exist_ok=True)
            (mixed / "labels" / sp).mkdir(parents=True, exist_ok=True)

        total_train_imgs = 0
        total_val_imgs = 0
        total_train_boxes = 0
        total_val_boxes = 0

        # Merge all generic_defect datasets
        merge_keys = [k for k, spec in self._enabled_specs().items()
                      if spec.target_class == GENERIC_DEFECT_CLASS and k in outputs]

        print(info(f"Merging {len(merge_keys)} datasets as generic_defect: {merge_keys}"))

        for key in merge_keys:
            out = outputs[key]
            for sp in ("train", "val"):
                img_src = out / "images" / sp
                lbl_src = out / "labels" / sp
                img_dst = mixed / "images" / sp
                lbl_dst = mixed / "labels" / sp

                if img_src.is_dir():
                    n = _make_symlink_tree(img_src, img_dst, "*.jpg")
                    n += _make_symlink_tree(img_src, img_dst, "*.png")
                    if sp == "train":
                        total_train_imgs += n
                    else:
                        total_val_imgs += n

                if lbl_src.is_dir():
                    n = _make_symlink_tree(lbl_src, lbl_dst, "*.txt")
                    # Count boxes
                    for lbl in lbl_src.glob("*.txt"):
                        lines = lbl.read_text().strip().splitlines()
                        if sp == "train":
                            total_train_boxes += len([l for l in lines if l.strip()])
                        else:
                            total_val_boxes += len([l for l in lines if l.strip()])

        # Write data.yaml
        data_yaml = mixed / "data.yaml"
        config = {
            "path": str(mixed),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": [GENERIC_DEFECT_CLASS],
        }
        if yaml is not None:
            data_yaml.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        else:
            # Manual YAML write
            lines = [
                f"path: {mixed}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                f"names: [\"{GENERIC_DEFECT_CLASS}\"]",
            ]
            data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(ok(f"mixed_pretrain created:"))
        print(f"     Train: {total_train_imgs} images, ~{total_train_boxes} boxes")
        print(f"     Val:   {total_val_imgs} images, ~{total_val_boxes} boxes")
        print(f"     Config: {data_yaml}")
        print(f"     Class:  [{GENERIC_DEFECT_CLASS}] (all defects merged)")

        return True

    # ── Phase 5: Summary ───────────────────────────────────────────
    def print_summary(
        self, found: Dict[str, Optional[Path]], sources: Dict[str, Path],
        outputs: Dict[str, Path],
    ) -> None:
        """Print a final summary of what was built."""
        print(f"\n{C_BOLD}{'='*60}{C_RESET}")
        print(f"{C_BOLD}  Summary{C_RESET}")
        print(f"{'='*60}")

        specs = self._enabled_specs()
        rows: List[Tuple[str, str, str, str]] = []
        for key in self._spec_order():
            spec = specs.get(key)
            if spec is None:
                continue
            f_status = "[+]" if (key in found and found[key]) else "[-]"
            s_status = "[+]" if key in sources else "[-]"
            o_status = "[+]" if key in outputs else "[-]"
            rows.append((spec.name, f_status, s_status, o_status))

        print(f"  {'Dataset':<30} {'Pub':>4} {'DL':>4} {'Conv':>4}")
        print(f"  {'-'*42}")
        for name, f_s, s_s, o_s in rows:
            print(f"  {name:<30} {f_s:>4} {s_s:>4} {o_s:>4}")

        print(f"\n  Output root: {self.output_root}")
        if self.mixed_dir.exists():
            train_imgs = len(list((self.mixed_dir / "images" / "train").glob("*")))
            val_imgs = len(list((self.mixed_dir / "images" / "val").glob("*")))
            print(f"  Mixed pretrain: {train_imgs} train / {val_imgs} val images")

        print(f"\n  Next steps:")
        print(f"    1. Verify: python scripts/multi_source_dataset_builder.py --scan-only")
        print(f"    2. Train with mixed_pretrain/data.yaml")

    # ── Helpers ────────────────────────────────────────────────────
    def _enabled_specs(self) -> Dict[str, DatasetSpec]:
        specs = {k: s for k, s in DATASET_SPECS.items() if s.enabled}
        if self.datasets_filter:
            # Also include explicitly requested datasets even if disabled
            for k in self.datasets_filter:
                if k in DATASET_SPECS and not DATASET_SPECS[k].enabled:
                    specs[k] = DATASET_SPECS[k]
            specs = {k: s for k, s in specs.items() if k in self.datasets_filter}
        return specs

    def _spec_order(self) -> List[str]:
        """Return dataset keys in priority order."""
        specs = self._enabled_specs()
        return sorted(specs, key=lambda k: (specs[k].priority, k))

    def run(self) -> Dict[str, Path]:
        """Execute the full pipeline."""
        print(f"{C_BOLD}{C_CYAN}")
        print("=" * 62)
        print("  AutoDL Multi-Source Dataset Builder")
        print("  Subway Defect Detection — Public Pretrain Data")
        print("=" * 62)
        print(f"{C_RESET}")

        print(info(f"Output root: {self.output_root}"))
        print(info(f"Enabled datasets: {list(self._enabled_specs().keys())}"))

        # Phase 1: Scan
        found = self.scan_autodl_pub()
        if self.scan_only:
            # Print what we found and stop
            print(f"\n{C_BOLD}Available in autodl-pub:{C_RESET}")
            for k, v in found.items():
                if v:
                    print(f"  {DATASET_SPECS[k].name}: {v}")
            return {}

        # Phase 2: Download missing
        sources = self.download_missing(found)

        # Phase 2.5: Check for manually-downloaded datasets in _downloads/
        for key, spec in self._enabled_specs().items():
            if key in sources:
                continue  # already found
            manual_dir = self.download_dir / key
            if manual_dir.exists() and any(manual_dir.iterdir()):
                print(ok(f"[{key}] Found manually downloaded data at {manual_dir}"))
                sources[key] = manual_dir

        # Phase 3: Convert
        outputs = self.convert_all(sources)

        # Phase 4: Merge
        generic_outputs = {
            k: v for k, v in outputs.items()
            if DATASET_SPECS[k].target_class == GENERIC_DEFECT_CLASS
        }
        if generic_outputs:
            self.create_mixed_pretrain(generic_outputs)

        # Also create tiny_object pretrain if TT100K is available
        tiny_outputs = {
            k: v for k, v in outputs.items()
            if DATASET_SPECS[k].target_class == TINY_OBJECT_CLASS
        }
        if tiny_outputs:
            self._create_tiny_mix(tiny_outputs)

        # Phase 5: Summary
        self.print_summary(found, sources, outputs)

        return outputs

    def _create_tiny_mix(self, outputs: Dict[str, Path]) -> None:
        """Create tiny_object pretrain dataset (for P2 head warmup)."""
        tiny_dir = self.output_root / "mixed_tiny_pretrain"
        if _dry_run_guard(self.dry_run):
            print(info(f"[DRY-RUN] Would create: {tiny_dir / 'data.yaml'}"))
            return
        for sp in ("train", "val"):
            (tiny_dir / "images" / sp).mkdir(parents=True, exist_ok=True)
            (tiny_dir / "labels" / sp).mkdir(parents=True, exist_ok=True)

        for key, out in outputs.items():
            for sp in ("train", "val"):
                img_src = out / "images" / sp
                lbl_src = out / "labels" / sp
                if img_src.is_dir():
                    _make_symlink_tree(img_src, tiny_dir / "images" / sp, "*.jpg")
                    _make_symlink_tree(img_src, tiny_dir / "images" / sp, "*.png")
                if lbl_src.is_dir():
                    _make_symlink_tree(lbl_src, tiny_dir / "labels" / sp, "*.txt")

        data_yaml = tiny_dir / "data.yaml"
        config = {
            "path": str(tiny_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": [TINY_OBJECT_CLASS],
        }
        if yaml is not None:
            data_yaml.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        else:
            lines = [
                f"path: {tiny_dir}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                f"names: [\"{TINY_OBJECT_CLASS}\"]",
            ]
            data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print(ok(f"mixed_tiny_pretrain created at {tiny_dir}"))


def _dry_run_guard(dry_run: bool) -> bool:
    """If dry_run, print a note and return True (meaning 'skip')."""
    if dry_run:
        print(info("[DRY-RUN] Would execute here"))
    return dry_run


# ==========================================================================
# CLI
# ==========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoDL Multi-Source Dataset Builder for Subway Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/multi_source_dataset_builder.py --scan-only
  python scripts/multi_source_dataset_builder.py
  python scripts/multi_source_dataset_builder.py --datasets deeppcb neu_det gc10_det
  python scripts/multi_source_dataset_builder.py --output data/multi_datasets
  python scripts/multi_source_dataset_builder.py --dry-run
  python scripts/multi_source_dataset_builder.py --no-download
  python scripts/multi_source_dataset_builder.py --enable tt100k insulator_defect
""",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--datasets", nargs="*",
        help="Specific dataset keys to process (default: all enabled)",
    )
    parser.add_argument(
        "--enable", nargs="*", default=[],
        help="Enable optional datasets (tt100k, insulator_defect, mvtec_ad, visa)",
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Only scan /root/autodl-pub, do not build",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip downloading missing datasets",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 4, 16),
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio (default: 0.2)",
    )
    args = parser.parse_args()

    # Enable any explicitly requested optional datasets
    for key in args.enable:
        if key in DATASET_SPECS:
            DATASET_SPECS[key].enabled = True
            print(info(f"Enabled optional dataset: {key}"))
        else:
            print(warn(f"Unknown dataset key: {key}. Available: {list(DATASET_SPECS)}"))

    builder = DatasetBuilder(
        output_root=args.output,
        datasets=args.datasets,
        no_download=args.no_download,
        scan_only=args.scan_only,
        dry_run=args.dry_run,
        workers=args.workers,
        val_ratio=args.val_ratio,
    )

    try:
        builder.run()
    except KeyboardInterrupt:
        print(f"\n{C_YELLOW}Interrupted by user{C_RESET}")
        sys.exit(130)
    except Exception as exc:
        print(f"\n{C_RED}Fatal error: {exc}{C_RESET}")
        raise


if __name__ == "__main__":
    main()
