#!/usr/bin/env python3
"""
Train state classifiers for two-stage defect verification.

Trains lightweight MobileNetV3-small classifiers to verify YOLO proposals:
  - CBHPM: binary (normal vs missing)
  - CBVPM: binary (normal vs missing)
  - SVHBNM: binary (normal vs missing)
  - SVHBNL: binary (normal vs loose)
  - SVHTNL: binary (normal vs loose)
  - VHBNM/VHBNL: 4-class (normal / missing / loose / ambiguous)

Usage::

    # Train CBHPM binary classifier
    python scripts/train_state_classifier.py --task cbhpm

    # Train all binary classifiers
    python scripts/train_state_classifier.py --task cbvpm
    python scripts/train_state_classifier.py --task svhbnm
    python scripts/train_state_classifier.py --task svhbnl
    python scripts/train_state_classifier.py --task svhtnl

    # Train VHBNM/VHBNL 4-class classifier
    python scripts/train_state_classifier.py --task vhbnm_vhbnl

    # Custom data path and epochs
    python scripts/train_state_classifier.py --task cbhpm --data data/classifier/cbhpm --epochs 50

    # Evaluate on test set after training
    python scripts/train_state_classifier.py --task cbhpm --evaluate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


TASK_CONFIGS = {
    "cbhpm": {
        "class_names": ["normal", "missing"],
        "num_classes": 2,
        "description": "CBHPM binary: normal vs missing (腕臂底座横向销钉缺口)",
        "default_data": "data/classifier/cbhpm",
        "output_weight": "weights/classifier_cbhpm.pt",
    },
    "cbvpm": {
        "class_names": ["normal", "missing"],
        "num_classes": 2,
        "description": "CBVPM binary: normal vs missing (腕臂底座垂直销钉缺口)",
        "default_data": "data/classifier/cbvpm",
        "output_weight": "weights/classifier_cbvpm.pt",
    },
    "svhbnm": {
        "class_names": ["normal", "missing"],
        "num_classes": 2,
        "description": "SVHBNM binary: normal vs missing (单支垂直悬吊槽钢底座螺母缺失)",
        "default_data": "data/classifier/svhbnm",
        "output_weight": "weights/classifier_svhbnm.pt",
    },
    "svhbnl": {
        "class_names": ["normal", "loose"],
        "num_classes": 2,
        "description": "SVHBNL binary: normal vs loose (单支垂直悬吊槽钢底座螺母松动)",
        "default_data": "data/classifier/svhbnl",
        "output_weight": "weights/classifier_svhbnl.pt",
    },
    "svhtnl": {
        "class_names": ["normal", "loose"],
        "num_classes": 2,
        "description": "SVHTNL binary: normal vs loose (单支垂直悬吊槽钢上方螺母松动)",
        "default_data": "data/classifier/svhtnl",
        "output_weight": "weights/classifier_svhtnl.pt",
    },
    "vhbnm_vhbnl": {
        "class_names": ["normal", "missing", "loose", "ambiguous"],
        "num_classes": 4,
        "description": "VHBNM/VHBNL 4-class: normal/missing/loose/ambiguous",
        "default_data": "data/classifier/vhbnm_vhbnl",
        "output_weight": "weights/classifier_vhbnm_vhbnl.pt",
    },
    "vhb_level1": {
        "class_names": ["normal", "defective"],
        "num_classes": 2,
        "description": "VHB hierarchical L1: normal vs defective (is there any problem?)",
        "default_data": "data/classifier/vhb_level1",
        "output_weight": "weights/classifier_vhb_level1.pt",
    },
    "vhb_level2": {
        "class_names": ["missing", "loose"],
        "num_classes": 2,
        "description": "VHB hierarchical L2: missing vs loose (what type of defect?)",
        "default_data": "data/classifier/vhb_level2",
        "output_weight": "weights/classifier_vhb_level2.pt",
    },
    "insd": {
        "class_names": ["normal", "damage"],
        "num_classes": 2,
        "description": "INSD binary: normal vs damage (绝缘子破损) — P1 high-FP reduction target",
        "default_data": "data/classifier/insd",
        "output_weight": "weights/classifier_insd.pt",
    },
    "bsbm": {
        "class_names": ["normal", "missing"],
        "num_classes": 2,
        "description": "BSBM binary: normal vs missing (汇流排中间接头螺栓缺失) — critical defect",
        "default_data": "data/classifier/bsbm",
        "output_weight": "weights/classifier_bsbm.pt",
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Train state classifier for two-stage defect verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_state_classifier.py --task cbhpm
  python scripts/train_state_classifier.py --task vhbnm_vhbnl --epochs 50
  python scripts/train_state_classifier.py --task cbhpm --evaluate
""",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(TASK_CONFIGS.keys()),
        help="Classifier task to train",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Data directory (default: task-specific)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output weight path (default: task-specific)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate on test set after training",
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone for first few epochs (warmup head only)",
    )
    args = parser.parse_args()

    config = TASK_CONFIGS[args.task]
    data_dir = args.data or Path(config["default_data"])
    output_path = args.output or Path(config["output_weight"])

    print("=" * 60)
    print("  State Classifier Training")
    print("=" * 60)
    print(f"  Task:        {args.task}")
    print(f"  Description: {config['description']}")
    print(f"  Classes:     {config['class_names']}")
    print(f"  Data:        {data_dir}")
    print(f"  Output:      {output_path}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.lr}")
    print(f"  Patience:    {args.patience}")
    print(f"  Device:      {args.device}")
    print()

    # Check data exists
    if not data_dir.is_dir():
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"  Run: python scripts/prepare_classifier_data.py")
        sys.exit(1)

    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        print(f"ERROR: Train directory not found: {train_dir}")
        sys.exit(1)

    # Import after path setup
    from subway_defect.classifier.model import StateClassifier
    from subway_defect.classifier.dataset import build_dataloaders
    from subway_defect.classifier.train import train_classifier

    # Build dataloaders
    print("  Loading data...")
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root=data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        class_names=config["class_names"],
    )

    # Print dataset stats
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    if test_loader:
        print(f"  Test:  {len(test_loader.dataset)} samples")

    train_counts = train_dataset.get_class_counts()
    print(f"  Train class distribution:")
    for name, count in train_counts.items():
        print(f"    {name}: {count}")
    print()

    # Create model
    model = StateClassifier(
        num_classes=config["num_classes"],
        pretrained=True,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    )
    print(f"  Model: MobileNetV3-small ({model.num_parameters:,} params)")
    print()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Train
    t0 = time.time()
    best_metrics = train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
        save_path=output_path,
        class_names=config["class_names"],
    )
    elapsed = time.time() - t0

    print(f"\n  Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best model saved to: {output_path}")
    print(f"  Best metrics:")
    for key, value in sorted(best_metrics.items()):
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")

    # Evaluate on test set
    if args.evaluate and test_loader:
        print(f"\n  {'='*50}")
        print(f"  Test Set Evaluation")
        print(f"  {'='*50}")

        import torch
        import numpy as np
        from subway_defect.classifier.train import compute_metrics

        # Load best model
        best_model = StateClassifier.load(output_path, device=args.device)
        device = torch.device(f"cuda:{args.device}" if args.device not in ("", "cpu") else "cpu")
        best_model.to(device)
        best_model.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = best_model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        test_metrics = compute_metrics(all_preds, all_labels, config["num_classes"])

        print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Test macro-F1: {test_metrics['macro_f1']:.4f}")
        for c, name in enumerate(config["class_names"]):
            p = test_metrics.get(f"class_{c}_precision", 0)
            r = test_metrics.get(f"class_{c}_recall", 0)
            f1 = test_metrics.get(f"class_{c}_f1", 0)
            print(f"    {name:12s}: P={p:.3f} R={r:.3f} F1={f1:.3f}")

        # Gate check
        print(f"\n  Gate Check:")
        if test_metrics["macro_f1"] >= 0.70:
            print(f"    ✅ macro-F1 {test_metrics['macro_f1']:.4f} >= 0.70")
        else:
            print(f"    ❌ macro-F1 {test_metrics['macro_f1']:.4f} < 0.70 — needs more data")

    # Usage hint
    print(f"\n  Integration:")
    print(f"    from subway_defect.classifier import ClassifierReasoner")
    print(f"    reasoner = ClassifierReasoner('{output_path}', class_names={config['class_names']})")
    print(f"    pipeline = TwoStagePipeline(..., state_reasoner=reasoner)")


if __name__ == "__main__":
    main()
