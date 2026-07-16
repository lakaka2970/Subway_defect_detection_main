#!/usr/bin/env python3
"""
Setup COCO dataset from AutoDL pub directory for YOLO training.

Extracts COCO images from ``/root/autodl-pub/COCO*/`` and downloads or
generates YOLO-format labels, producing a standard Ultralytics-compatible
directory tree::

    test_fixtures/coco/
    ├── images/
    │   ├── train2014/       # (or train2017 depending on source)
    │   └── val2014/
    ├── labels/
    │   ├── train2014/
    │   └── val2014/
    ├── train2014.txt
    ├── val2014.txt
    └── coco.yaml

Usage::

    python scripts/setup_coco_from_autodl.py
    python scripts/setup_coco_from_autodl.py --output test_fixtures/coco
    python scripts/setup_coco_from_autodl.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore

# ── Constants ────────────────────────────────────────────────────────────

AUTODL_PUB = Path("/root/autodl-pub")
DEFAULT_OUTPUT = Path("test_fixtures/coco")

# Ultralytics assets for YOLO-format COCO labels
ULTRALYTICS_ASSETS = "https://github.com/ultralytics/assets/releases/download/v0.0.0"

# COCO official annotation URLs (fallback if YOLO labels unavailable)
COCO_ANNO_URLS = {
    "2014": {
        "train": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    },
    "2017": {
        "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    },
}

# COCO 80-class names (ID → name)
COCO_CLASS_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench",
    15: "bird", 16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow",
    21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack",
    26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee",
    31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite",
    35: "baseball bat", 36: "baseball glove", 37: "skateboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup",
    42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana",
    47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

MIN_BBOX_SIDE = 2
MIN_BBOX_AREA = 8


# ── Helpers ──────────────────────────────────────────────────────────────

def _find_coco_in_autodl() -> Optional[Path]:
    """Find COCO dataset directory under ``/root/autodl-pub``."""
    if not AUTODL_PUB.exists():
        return None
    for pattern in ["COCO*", "coco*", "mscoco*"]:
        matches = sorted(AUTODL_PUB.glob(pattern))
        for m in matches:
            if m.is_dir():
                return m
    return None


def _detect_coco_year(coco_dir: Path) -> Optional[str]:
    """Detect COCO year from zip filenames (e.g. train2014.zip → '2014')."""
    for f in coco_dir.iterdir():
        m = re.match(r".*?(\d{4})\.(zip|tar)", f.name, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> bool:
    """Run a subprocess, return True on success."""
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"  [CMD FAIL] {' '.join(cmd)} → exit {exc.returncode}")
        return False
    except FileNotFoundError:
        print(f"  [CMD FAIL] not found: {cmd[0]}")
        return False


def _download(url: str, dest: Path) -> bool:
    """Download a file with progress (urllib fallback if requests unavailable)."""
    try:
        import requests
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)

        if tqdm is not None:
            pbar = tqdm(total=total, unit="B", unit_scale=True,
                        desc=f"  Download {dest.name}")
        else:
            pbar = None

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))
        if pbar:
            pbar.close()
        return True
    except ImportError:
        # Fallback to urllib
        from urllib.request import urlretrieve
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading {url} ...")
        urlretrieve(url, dest)
        return dest.exists()
    except Exception as exc:
        print(f"  [FAIL] Download: {url} → {exc}")
        return False


# ── Phase 1: Extract images ──────────────────────────────────────────────

def extract_images(coco_dir: Path, output_dir: Path, year: str) -> Dict[str, Path]:
    """Extract train/val zip archives to ``output_dir/images/``.

    Returns ``{"train": img_dir, "val": img_dir}``.
    """
    img_base = output_dir / "images"
    result: Dict[str, Path] = {}

    for split, prefix in [("train", "train"), ("val", "val")]:
        # Find the zip file
        zip_path = None
        for candidate in [
            coco_dir / f"{prefix}{year}.zip",
            coco_dir / f"{prefix}{year}.tar",
        ]:
            if candidate.exists():
                zip_path = candidate
                break

        if zip_path is None:
            # Try glob
            for f in coco_dir.glob(f"{prefix}*.zip"):
                zip_path = f
                break

        if zip_path is None:
            print(f"  [WARN] No {prefix} archive found in {coco_dir}")
            continue

        dst_dir = img_base / f"{prefix}{year}"
        if dst_dir.exists() and any(dst_dir.iterdir()):
            print(f"  [OK] {split} images already extracted → {dst_dir}")
            result[split] = dst_dir
            continue

        print(f"  Extracting {zip_path.name} → {dst_dir} ...")
        dst_dir.mkdir(parents=True, exist_ok=True)

        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.infolist()
                if tqdm is not None:
                    members = tqdm(members, desc=f"  {split}", unit="file")
                for member in members:
                    zf.extract(member, dst_dir)
        elif ".tar" in zip_path.suffixes:
            import tarfile
            with tarfile.open(zip_path, "r:*") as tf:
                tf.extractall(dst_dir)  # type: ignore[attr-defined]

        result[split] = dst_dir

    return result


# ── Phase 2: Labels ─────────────────────────────────────────────────────

def _try_download_yolo_labels(output_dir: Path, year: str) -> bool:
    """Try to download pre-made YOLO labels from Ultralytics assets.

    Returns True if successful.
    """
    url = f"{ULTRALYTICS_ASSETS}/coco{year}labels.zip"
    zip_path = output_dir / "_coco_labels.zip"

    print(f"  Trying YOLO labels: {url}")
    if zip_path.exists():
        print(f"  [OK] Labels archive already downloaded")
    else:
        if not _download(url, zip_path):
            return False

    # Extract labels
    lbl_dir = output_dir / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(lbl_dir)
        print(f"  [OK] YOLO labels extracted to {lbl_dir}")
        return True
    except zipfile.BadZipFile:
        print(f"  [WARN] Corrupted labels zip, will regenerate")
        zip_path.unlink()
        return False


def _generate_labels_from_coco_json(output_dir: Path, year: str) -> bool:
    """Generate YOLO labels from official COCO JSON annotations.

    Downloads annotation JSON and converts to YOLO .txt files.
    """
    print(f"  Generating YOLO labels from COCO {year} JSON annotations ...")

    # Download annotation JSON
    anno_url = COCO_ANNO_URLS.get(year, {}).get("train")
    if not anno_url:
        print(f"  [FAIL] No annotation URL known for COCO {year}")
        return False

    anno_zip = output_dir / f"_coco_anno_{year}.zip"
    if not anno_zip.exists():
        if not _download(anno_url, anno_zip):
            return False

    # Extract JSON
    anno_dir = output_dir / f"_anno_{year}"
    if not anno_dir.exists():
        anno_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(anno_zip, "r") as zf:
            zf.extractall(anno_dir)
        print(f"  Extracted annotations to {anno_dir}")

    # Find the instances JSON file
    instances_json = None
    for f in anno_dir.rglob("instances_*.json"):
        if "train" in f.name or "val" in f.name:
            instances_json = f
            break
    if instances_json is None:
        # Look in annotations/ subdir
        for f in anno_dir.rglob("annotations/instances_*.json"):
            instances_json = f
            break
    if instances_json is None:
        print(f"  [FAIL] Cannot find instances JSON in {anno_dir}")
        print(f"  Contents: {list(anno_dir.rglob('*'))[:20]}")
        return False

    print(f"  Parsing {instances_json.name} ...")
    with open(instances_json, encoding="utf-8") as f:
        data = json.load(f)

    # Build image_id → (filename, width, height)
    images: Dict[int, Tuple[str, int, int]] = {}
    for img in data.get("images", []):
        images[img["id"]] = (img.get("file_name", f"{img['id']:012d}.jpg"),
                             img["width"], img["height"])

    # Collect annotations per image
    img_annos: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    for anno in data.get("annotations", []):
        img_id = anno["image_id"]
        cat_id = anno["category_id"]
        bbox = anno["bbox"]  # COCO format: [x, y, width, height]
        x, y, w, h = bbox

        if w < MIN_BBOX_SIDE or h < MIN_BBOX_SIDE:
            continue
        if w * h < MIN_BBOX_AREA:
            continue

        # COCO cat_id starts at 1, YOLO cls_id at 0
        cls_id = cat_id - 1

        # Convert to YOLO normalized [xc, yc, w, h]
        img_info = images.get(img_id)
        if img_info is None:
            continue
        _, iw, ih = img_info
        xc = (x + w / 2) / iw
        yc = (y + h / 2) / ih
        wn = w / iw
        hn = h / ih

        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        wn = max(0.0, min(1.0, wn))
        hn = max(0.0, min(1.0, hn))

        if wn <= 0 or hn <= 0:
            continue

        img_annos[img_id].append((cls_id, xc, yc, wn, hn))

    # Determine which split each image belongs to
    # COCO 2014: train2014 and val2014 from separate JSONs
    # For simplicity, all images from this JSON → train
    split_name = "train" if "train" in instances_json.name else "val"

    lbl_dir = output_dir / "labels" / f"{split_name}{year}"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id, annos in (tqdm(list(img_annos.items()), desc="  Writing labels",
                               unit="img") if tqdm else img_annos.items()):
        img_name = Path(images[img_id][0]).stem
        lines = [f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                 for cid, xc, yc, wn, hn in annos]
        (lbl_dir / f"{img_name}.txt").write_text(
            "\n".join(lines) + "\n" if lines else "\n", encoding="utf-8")
        count += 1

    print(f"  [OK] Generated {count} label files → {lbl_dir}")
    return True


def setup_labels(output_dir: Path, year: str, img_dirs: Dict[str, Path]) -> bool:
    """Get YOLO labels — try download first, fall back to generating from COCO JSON."""
    # Try pre-made YOLO labels
    if _try_download_yolo_labels(output_dir, year):
        # Verify labels match images
        for split, img_dir in img_dirs.items():
            lbl_dir = output_dir / "labels" / f"{split}{year}"
            if not lbl_dir.exists():
                print(f"  [WARN] Labels for {split} not found, may need generation")
        return True

    # Fallback: generate from COCO JSON
    print("  Pre-made labels unavailable, generating from COCO JSON ...")
    if not _generate_labels_from_coco_json(output_dir, year):
        return False

    # COCO JSON only covers one split; try to handle both
    # For val split, we might need a different JSON
    return True


# ── Phase 3: Create config files ─────────────────────────────────────────

def create_config(output_dir: Path, year: str, img_dirs: Dict[str, Path]) -> None:
    """Create train/val .txt path lists and coco.yaml."""
    # Create path list files
    for split, img_dir in img_dirs.items():
        txt_file = output_dir / f"{split}{year}.txt"
        image_files = sorted(
            list(img_dir.glob("*.jpg")) +
            list(img_dir.glob("*.jpeg")) +
            list(img_dir.glob("*.png"))
        )
        if not image_files:
            # Check one level deeper (zip might have a subdirectory)
            for sub in img_dir.iterdir():
                if sub.is_dir():
                    image_files += sorted(
                        list(sub.glob("*.jpg")) +
                        list(sub.glob("*.jpeg")) +
                        list(sub.glob("*.png"))
                    )

        if image_files:
            content = "\n".join(
                str(f.relative_to(output_dir).as_posix()) for f in image_files
            )
            txt_file.write_text(content + "\n", encoding="utf-8")
            print(f"  [OK] {txt_file.name}: {len(image_files)} paths")

    # Create coco.yaml
    import yaml as yaml_lib
    config = {
        "path": str(output_dir.resolve()),
        "train": f"train{year}.txt",
        "val": f"val{year}.txt",
        "nc": 80,
        "names": {i: name for i, name in COCO_CLASS_NAMES.items()},
    }
    yaml_path = output_dir / "coco.yaml"
    yaml_path.write_text(
        yaml_lib.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"  [OK] {yaml_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup COCO dataset from AutoDL pub directory",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output dataset directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--coco-dir", type=Path,
        help="Manual COCO source directory (auto-detect from /root/autodl-pub if omitted)",
    )
    parser.add_argument(
        "--year", type=str,
        help="COCO year (e.g. 2014, 2017). Auto-detected from filenames if omitted.",
    )
    parser.add_argument(
        "--skip-labels", action="store_true",
        help="Skip label generation (images only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without executing",
    )
    args = parser.parse_args()

    # ── Discover COCO source ──────────────────────────────────────────
    coco_dir = args.coco_dir
    if coco_dir is None:
        coco_dir = _find_coco_in_autodl()
    if coco_dir is None:
        print("ERROR: No COCO directory found in /root/autodl-pub.")
        print("       Specify manually with --coco-dir")
        sys.exit(1)

    if not coco_dir.exists():
        print(f"ERROR: COCO directory not found: {coco_dir}")
        sys.exit(1)

    year = args.year or _detect_coco_year(coco_dir)
    if year is None:
        print("ERROR: Cannot detect COCO year from filenames.")
        print("       Specify with --year (e.g. --year 2014)")
        sys.exit(1)

    output_dir = args.output

    print("=" * 60)
    print("  COCO Dataset Setup (from AutoDL pub)")
    print("=" * 60)
    print(f"  Source : {coco_dir}")
    print(f"  Year   : {year}")
    print(f"  Output : {output_dir}")
    print(f"  Dry-run: {args.dry_run}")
    print()

    if args.dry_run:
        print("[DRY-RUN] Would extract and set up COCO dataset")
        return

    # Phase 1: Extract images
    print("─ Phase 1: Extract images ─")
    img_dirs = extract_images(coco_dir, output_dir, year)
    if not img_dirs:
        print("ERROR: No images extracted!")
        sys.exit(1)
    for split, d in img_dirs.items():
        n = len(list(d.rglob("*.*")))
        print(f"  {split}: {n} files in {d}")

    # Phase 2: Labels
    if not args.skip_labels:
        print("\n─ Phase 2: Labels ─")
        setup_labels(output_dir, year, img_dirs)

    # Phase 3: Config
    print("\n─ Phase 3: Config files ─")
    create_config(output_dir, year, img_dirs)

    print(f"\n{'=' * 60}")
    print("  Setup complete!")
    print(f"  Dataset: {output_dir.resolve()}")
    print(f"  Config:  {output_dir / 'coco.yaml'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
