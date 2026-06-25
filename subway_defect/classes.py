"""
Central defect class registry — single source of truth for class names.

Every module that references defect categories should import from here
rather than hard-coding class name lists.  This ensures that adding or
renaming a class is a one-line change that propagates everywhere.

Usage::

    from subway_defect.classes import DEFECT_CLASSES, NC, get_class_id, get_class_name

    # Iterate over classes
    for idx, name in enumerate(DEFECT_CLASSES):
        ...

    # Look up by name or index
    cls_id = get_class_id("VHBNM")     # → 0
    name  = get_class_name(3)          # → "SVHBNL"

    # Use NC in model YAML or dataset config
    nc = NC  # 18 (full taxonomy)

Classes
-------
The full 18-class taxonomy covers rigid catenary (13 classes) and
flexible catenary / general defects (5 classes).  The standard 7-class
subset used for most training runs is::

    VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM

Keep this file in sync with ``data/*/classes.txt`` and
``data/*/defect_data.yaml``.
"""

from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Full 18-class defect taxonomy
# ═══════════════════════════════════════════════════════════════════════════

# Rigid catenary (13 classes)
# ├─ VHBNM, VJBNM are the most common — >60% of instances
# ├─ SVHBN*, SVHTN* are relatively rare — ~10-15% of instances
# └─ CBHPM, CBVPM are moderate frequency — ~15-20% of instances
#
# Flexible catenary / General (5 classes)
# ├─ CWZJ, JXP are rare specific defects
# └─ BYLZ is catch-all for other defects

DEFECT_CLASSES: list[str] = [
    # ── Rigid catenary (刚性接触网) ──────────────────────────────────
    "VHBNM",    #  0 — Vertical Hex Bolt Normal Missing      (垂直主螺栓正常缺失)
    "VHBNL",    #  1 — Vertical Hex Bolt Normal Loose        (垂直主螺栓正常松动)
    "SVHBNM",   #  2 — Support Vertical Hex Bolt Normal Missing  (支撑垂直主螺栓正常缺失)
    "SVHBNL",   #  3 — Support Vertical Hex Bolt Normal Loose    (支撑垂直主螺栓正常松动)
    "SVHTNL",   #  4 — Support Vertical Hex Bolt Top Nut Loose   (支撑垂直主螺栓上螺母松)
    "CBHPM",    #  5 — Clevis Bolt Horizontal Pin Missing    (U型抱箍螺栓平销缺失)
    "CBVPM",    #  6 — Clevis Bolt Vertical Pin Missing      (U型抱箍螺栓竖销缺失)
    "CBHSL",    #  7 — Clevis Bolt Horizontal Shaft Loose    (U型抱箍螺栓横轴松动)
    "CBHNM",    #  8 — Clevis Bolt Horizontal Nut Missing    (U型抱箍螺栓平头螺母缺失)
    "CBHNL",    #  9 — Clevis Bolt Horizontal Nut Loose      (U型抱箍螺栓平头螺母松动)
    "VJBNM",    # 10 — Vertical J-Bolt Normal Missing        (垂直J型螺栓正常缺失)
    "VJBNL",    # 11 — Vertical J-Bolt Normal Loose          (垂直J型螺栓正常松动)
    "VJBSL",    # 12 — Vertical J-Bolt Shaft Loose           (垂直J型螺栓杆松动)

    # ── Flexible catenary / General (柔性接触网 / 通用) ──────────────
    "CWZJ",     # 13 — Catenary Wire ZJ-type defect          (接触线ZJ型缺陷)
    "JXP",      # 14 — JXP-type insulator defect             (JXP型绝缘子缺陷)
    "GJJ",      # 15 — Steel structure corrosion/deformation (钢结构锈蚀/变形)
    "DDL",      # 16 — Dropper loose/break                   (吊弦松动/断裂)
    "BYLZ",     # 17 — Other anomaly (不属于以上类别的缺陷)   (其他异常)
]

NC: int = len(DEFECT_CLASSES)  # 18

# ═══════════════════════════════════════════════════════════════════════════
# Standard 7-class subset (used for most training / inference runs)
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_CLASSES: list[str] = DEFECT_CLASSES[:7]  # indices 0–6
TRAIN_NC: int = len(TRAIN_CLASSES)             # 7

# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DEFECT_CLASSES)}
_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(DEFECT_CLASSES)}


def get_class_id(name: str) -> Optional[int]:
    """Return the integer class ID for a defect class name, or ``None``."""
    return _NAME_TO_ID.get(name.upper())


def get_class_name(idx: int) -> Optional[str]:
    """Return the defect class name for an integer ID, or ``None``."""
    return _ID_TO_NAME.get(idx)


def validate_class_id(cls_id: int) -> bool:
    """Return ``True`` if *cls_id* is a valid defect class index."""
    return 0 <= cls_id < NC


def validate_class_name(name: str) -> bool:
    """Return ``True`` if *name* is a valid defect class name (case-insensitive)."""
    return name.upper() in _NAME_TO_ID
