#!/usr/bin/env python3
"""Step 2: Generate defect_data.yaml for Ultralytics YOLO training.

channels: 3 — OpenCV IMREAD_COLOR auto-converts grayscale to 3-channel BGR
so COCO-pretrained weights (yolo11s.pt) remain compatible.

Usage:
    python scripts/create_defect_data_yaml.py
"""

import yaml
from pathlib import Path

CLASSES_PATH = Path("data/Defect_dataset/labels/classes.txt")
YAML_PATH = Path("data/Defect_dataset/defect_data.yaml")


def main() -> None:
    # Read class names from classes.txt
    names: dict[int, str] = {}
    lines = CLASSES_PATH.read_text(encoding="utf-8").strip().splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            names[i] = line.strip()

    config = {
        "path": "data/Defect_dataset",
        "train": "images/train",
        "val": "images/val",
        "test": None,
        "channels": 3,
        "names": names,
    }

    YAML_PATH.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[Step 2] Dataset YAML written: {YAML_PATH}")
    print(f"         {len(names)} classes, channels={config['channels']}")
    for cid, name in names.items():
        print(f"           {cid}: {name}")


if __name__ == "__main__":
    main()
