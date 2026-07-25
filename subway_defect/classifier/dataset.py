"""
Dataset for state classifier training.

Loads 128x128 crops extracted from YOLO proposals with state labels.
Supports binary (CBHPM: normal/missing) and multi-class (VHBNM/VHBNL:
normal/missing/loose/ambiguous) classification.

Directory structure expected::

    data/classifier/cbhpm/
    ├── train/
    │   ├── normal/       # *.jpg crops
    │   └── missing/      # *.jpg crops
    ├── val/
    │   ├── normal/
    │   └── missing/
    └── test/
        ├── normal/
        └── missing/

Or for 4-class::

    data/classifier/vhbnm_vhbnl/
    ├── train/
    │   ├── normal/
    │   ├── missing/
    │   ├── loose/
    │   └── ambiguous/
    ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


# Standard ImageNet normalization for MobileNetV3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INPUT_SIZE = 128


def get_train_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Training transforms with mild augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """Validation/test transforms (deterministic)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class StateDataset(Dataset):
    """Dataset for state classification from folder structure.

    Args:
        root: Root directory containing class subdirectories.
        transform: Image transforms to apply.
        class_names: Optional explicit class name list (sorted alphabetically
            from subdirectory names if not provided).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[transforms.Compose] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        if class_names:
            self.class_names = class_names
        else:
            self.class_names = sorted(
                d.name for d in self.root.iterdir() if d.is_dir()
            )

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        # Collect all samples
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = self.root / class_name
            if not class_dir.is_dir():
                continue
            class_idx = self.class_to_idx[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, class_idx))
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, class_idx))

        if not self.samples:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        if cv2 is not None:
            img = cv2.imread(str(img_path))
            if img is None:
                # Fallback: return a black image
                img = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            img = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label

    def get_class_counts(self) -> Dict[str, int]:
        """Return per-class sample counts."""
        counts = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            counts[self.class_names[label]] += 1
        return counts

    def get_sample_weights(self) -> torch.Tensor:
        """Compute per-sample weights for balanced sampling."""
        counts = self.get_class_counts()
        total = len(self.samples)
        weights = torch.zeros(total)
        for idx, (_, label) in enumerate(self.samples):
            class_name = self.class_names[label]
            weights[idx] = total / (self.num_classes * counts[class_name])
        return weights


def create_weighted_sampler(dataset: StateDataset) -> WeightedRandomSampler:
    """Create a weighted random sampler for class-balanced training."""
    weights = dataset.get_sample_weights()
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    class_names: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train/val/test dataloaders from directory structure.

    Args:
        data_root: Root directory with train/val/test subdirectories.
        batch_size: Batch size for training.
        num_workers: DataLoader workers.
        class_names: Optional explicit class names.

    Returns:
        (train_loader, val_loader, test_loader or None)
    """
    data_root = Path(data_root)

    # Infer class names from train directory if not provided
    if class_names is None:
        train_dir = data_root / "train"
        if train_dir.is_dir():
            class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    train_dataset = StateDataset(
        data_root / "train",
        transform=get_train_transforms(),
        class_names=class_names,
    )
    val_dataset = StateDataset(
        data_root / "val",
        transform=get_val_transforms(),
        class_names=class_names,
    )

    # Use weighted sampling for training
    sampler = create_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    test_dir = data_root / "test"
    if test_dir.is_dir():
        test_dataset = StateDataset(
            test_dir,
            transform=get_val_transforms(),
            class_names=class_names,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
