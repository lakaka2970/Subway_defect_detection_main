#!/usr/bin/env python3
"""Step 1: Strip trailing empty line from classes.txt so exactly 7 classes remain.

Usage:
    python tool/fix_classes_txt.py
"""

from pathlib import Path

CLASSES_PATH = Path("data/Defect_dataset/labels/classes.txt")


def main() -> None:
    text = CLASSES_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    CLASSES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[Step 1] Fixed {CLASSES_PATH}")
    print(f"         {len(lines)} classes: {lines}")


if __name__ == "__main__":
    main()
