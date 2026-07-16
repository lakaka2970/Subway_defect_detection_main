#!/usr/bin/env python3
"""
One-click dataset preparation for subway defect detection.

Runs the following steps in order:
  1. fix_classes_txt          — strip trailing blank line
  2. create_defect_data_yaml   — write defect_data.yaml
  3. split_dataset             — source-grouped 80/20 train/val split
  4. generate_scene_augmentations — tunnel/sunlit/blur/weather variants
  5. generate_synthetic_defects   — inpainting-based missing-defect samples
  6. validate_dataset          — integrity and statistics check

Usage:
    python scripts/prepare_dataset.py              # run all steps
    python scripts/prepare_dataset.py --step 3     # run only step 3
    python scripts/prepare_dataset.py --skip 4 5   # skip augmentations
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent

# Steps that support --workers parallel processing
_PARALLEL_STEPS = {3, 4, 5, 6}

STEPS: dict[int, tuple[str, str]] = {
    1: ("Fix classes.txt",             "fix_classes_txt.py"),
    2: ("Create defect_data.yaml",      "create_defect_data_yaml.py"),
    3: ("Train/Val split",              "split_dataset.py"),
    4: ("Scene augmentations",          "generate_scene_augmentations.py"),
    5: ("Synthetic defect generation",  "generate_synthetic_defects.py"),
    6: ("Validate dataset",             "validate_dataset.py"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click subway defect dataset preparation",
    )
    parser.add_argument(
        "--step", type=int, choices=range(1, 7),
        help="Run only a single step (1-6)",
    )
    parser.add_argument(
        "--skip", type=int, nargs="*", default=[],
        help="Step numbers to skip",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print steps without executing",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help=f"Parallel workers for steps 3-6 (default: {os.cpu_count()})",
    )
    args = parser.parse_args()

    steps_to_run = (
        [args.step] if args.step
        else [n for n in sorted(STEPS) if n not in (args.skip or [])]
    )

    print("=" * 60)
    print("  Subway Defect Dataset Preparation")
    print("=" * 60)
    print(f"  Steps to run: {steps_to_run}")
    if args.skip:
        print(f"  Skipped: {args.skip}")
    print(f"  Tool directory: {TOOL_DIR}")
    print()

    if args.dry_run:
        for step_num in steps_to_run:
            name, script = STEPS[step_num]
            print(f"  [DRY-RUN] Step {step_num}: {name} → {script}")
        return

    for step_num in steps_to_run:
        name, script = STEPS[step_num]
        script_path = TOOL_DIR / script

        print(f"\n{'=' * 60}")
        print(f"  Step {step_num}/{len(STEPS)}: {name}")
        print(f"  Script: {script}")
        print(f"{'=' * 60}")

        cmd = [sys.executable, str(script_path)]
        if step_num in _PARALLEL_STEPS:
            cmd.extend(["--workers", str(args.workers)])

        result = subprocess.run(
            cmd,
            cwd=TOOL_DIR.parent,
            check=False,
        )
        if result.returncode != 0:
            print(f"\n  ERROR: Step {step_num} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print(f"\n{'=' * 60}")
    print("  Dataset preparation complete!")
    print(f"  Config: data/Defect_dataset/defect_data.yaml")
    print(f"{'=' * 60}")
    print()
    print("  Next step — training:")
    print("    python -m subway_defect.train.train_defect \\")
    print("        --data data/Defect_dataset/defect_data.yaml \\")
    print("        --coco_pretrain --device 0")


if __name__ == "__main__":
    main()
