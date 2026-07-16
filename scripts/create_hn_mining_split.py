#!/usr/bin/env python3
"""Reserve a source-grouped HN mining subset from the frozen train sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def crop_source(stem: str) -> str:
    return re.sub(r"_[pn]\d+$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/subway_crops"))
    parser.add_argument("--manifest", type=Path, default=Path("data/eval_v20260714/manifest.json"))
    parser.add_argument("--ratio", type=float, default=.10)
    parser.add_argument("--output-sources", type=Path,
                        default=Path("data/eval_v20260714/hn_mining_sources.txt"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_sources = list(manifest["splits"]["train"])
    ranked = sorted(train_sources, key=lambda s: hashlib.sha256(f"714:{s}".encode()).hexdigest())
    count = max(1, round(len(ranked) * args.ratio))
    reserved = set(ranked[:count])
    args.output_sources.write_text("\n".join(sorted(reserved)) + "\n", encoding="utf-8")

    source_img = args.dataset_root / "train" / "images"
    source_lbl = args.dataset_root / "train" / "labels"
    target_img = args.dataset_root / "hn_mining" / "images"
    target_lbl = args.dataset_root / "hn_mining" / "labels"
    target_img.mkdir(parents=True, exist_ok=True)
    target_lbl.mkdir(parents=True, exist_ok=True)

    moved_images = moved_labels = 0
    for image in sorted(source_img.glob("*")):
        if not image.is_file() or crop_source(image.stem) not in reserved:
            continue
        label = source_lbl / f"{image.stem}.txt"
        shutil.move(str(image), target_img / image.name)
        moved_images += 1
        if label.exists():
            shutil.move(str(label), target_lbl / label.name)
            moved_labels += 1

    manifest.setdefault("hn_mining", {})
    manifest["hn_mining"] = {
        "source": "frozen train split only", "ratio_within_train": args.ratio,
        "sources": sorted(reserved), "images": moved_images, "labels": moved_labels,
        "calibration_or_test_sources": 0,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["hn_mining"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
