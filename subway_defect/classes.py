"""
Central defect class registry — single source of truth for class names.

Every module that references defect categories should import from here
rather than hard-coding class name lists.  This ensures that adding or
renaming a class is a one-line change that propagates everywhere.

The canonical class list is aligned with the authoritative document
``docs/接触网缺陷类型详解.docx`` (16 types).  The standard 7-class
subset used for training runs is::

    VHBNM, VHBNL, SVHBNM, SVHBNL, SVHTNL, CBHPM, CBVPM

Usage::

    from subway_defect.classes import (
        DEFECT_CLASSES, NC, SEVERITY_MAP, CN_NAME_MAP,
        get_class_id, get_class_name, get_cn_name, get_severity,
    )

    # Iterate over classes
    for idx, (code, cn_name) in enumerate(zip(DEFECT_CLASSES, ...)):
        ...

    # Look up by name or index
    cls_id = get_class_id("VHBNM")     # → 0
    name  = get_class_name(3)          # → "SVHBNL"
    cn    = get_cn_name("VHBNM")       # → "垂直悬吊安装底座螺母缺失"
    sev   = get_severity("VHBNM")      # → "serious"

    # Use NC in model YAML or dataset config
    nc = NC  # 16 (full taxonomy per authoritative document)
"""

from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════
# Full 16-class defect taxonomy (canonical — per 接触网缺陷类型详解.docx)
# ═══════════════════════════════════════════════════════════════════════════

DEFECT_CLASSES: list[str] = [
    # ── 刚性接触网 (Rigid Catenary) ──────────────────────────────────
    "VHBNM",    #  0 — 垂直悬吊安装底座螺母缺失
    "VHBNL",    #  1 — 垂直悬吊安装底座螺母松动
    "SVHBNM",   #  2 — 单支垂直悬吊槽钢底座螺母缺失
    "SVHBNL",   #  3 — 单支垂直悬吊槽钢底座螺母松动
    "SVHTNL",   #  4 — 单支垂直悬吊槽钢上方螺母松动
    "RHTBNM",   #  5 — 刚性悬挂吊柱顶板底面螺母缺失
    "RHTBNL",   #  6 — 刚性悬挂吊柱顶板底面螺母松动
    "GWCSBNM",  #  7 — 地线线夹托板安装底座螺母缺失
    "GWCSBNL",  #  8 — 地线线夹托板安装底座螺母松动
    "GWCNM",    #  9 — 地线线夹螺母缺失
    "GWCNL",    # 10 — 地线线夹螺母松动
    "BSBM",     # 11 — 汇流排中间接头螺栓缺失
    "INSD",     # 12 — 绝缘子破损
    # ── 柔性接触网 (Flexible Catenary) ──────────────────────────────
    "CBHPM",    # 13 — 腕臂底座横向销钉缺口
    "CBVPM",    # 14 — 腕臂底座垂直销钉缺口
    "DRPS",     # 15 — 吊弦不受力
]

NC: int = len(DEFECT_CLASSES)  # 16

# ═══════════════════════════════════════════════════════════════════════════
# Model/dataset class order (the order actually used by trained weights)
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: the trained YOLO11s-v2 weights and all dataset label files
# (data/train_data_2/classes.txt, data/Defect_dataset_16_rebuilt, etc.)
# use a DIFFERENT class order from the canonical DEFECT_CLASSES above.
# Positions 5-14 are permuted.  Never mix the two orders — convert
# explicitly with the helpers below.  See docs/plans/8.31泛化性阶段1报告.md §2.4.

MODEL_CLASS_ORDER: list[str] = [
    "VHBNM",    #  0
    "VHBNL",    #  1
    "SVHBNM",   #  2
    "SVHBNL",   #  3
    "SVHTNL",   #  4
    "CBHPM",    #  5
    "CBVPM",    #  6
    "RHTBNM",   #  7
    "RHTBNL",   #  8
    "GWCSBNM",  #  9
    "GWCSBNL",  # 10
    "GWCNM",    # 11
    "GWCNL",    # 12
    "BSBM",     # 13
    "INSD",     # 14
    "DRPS",     # 15
]


def model_id_to_canonical(model_id: int) -> int:
    """Map a trained-model class index to the canonical DEFECT_CLASSES index."""
    return get_class_id(MODEL_CLASS_ORDER[model_id])


def canonical_to_model_id(canonical_id: int) -> int:
    """Map a canonical DEFECT_CLASSES index to the trained-model class index."""
    return MODEL_CLASS_ORDER.index(get_class_name(canonical_id))

# ═══════════════════════════════════════════════════════════════════════════
# Chinese name mapping (per 接触网缺陷类型详解.docx)
# ═══════════════════════════════════════════════════════════════════════════

CN_NAME_MAP: dict[str, str] = {
    "VHBNM":   "垂直悬吊安装底座螺母缺失",
    "VHBNL":   "垂直悬吊安装底座螺母松动",
    "SVHBNM":  "单支垂直悬吊槽钢底座螺母缺失",
    "SVHBNL":  "单支垂直悬吊槽钢底座螺母松动",
    "SVHTNL":  "单支垂直悬吊槽钢上方螺母松动",
    "RHTBNM":  "刚性悬挂吊柱顶板底面螺母缺失",
    "RHTBNL":  "刚性悬挂吊柱顶板底面螺母松动",
    "GWCSBNM": "地线线夹托板安装底座螺母缺失",
    "GWCSBNL": "地线线夹托板安装底座螺母松动",
    "GWCNM":   "地线线夹螺母缺失",
    "GWCNL":   "地线线夹螺母松动",
    "BSBM":    "汇流排中间接头螺栓缺失",
    "INSD":    "绝缘子破损",
    "CBHPM":   "腕臂底座横向销钉缺口",
    "CBVPM":   "腕臂底座垂直销钉缺口",
    "DRPS":    "吊弦不受力",
}

# ═══════════════════════════════════════════════════════════════════════════
# Severity level mapping (per 接口规范标准 §5.2)
# ═══════════════════════════════════════════════════════════════════════════

SEVERITY_MAP: dict[str, str] = {
    "VHBNM":   "serious",
    "VHBNL":   "serious",
    "SVHBNM":  "serious",
    "SVHBNL":  "serious",
    "SVHTNL":  "normal",
    "RHTBNM":  "serious",
    "RHTBNL":  "serious",
    "GWCSBNM": "serious",
    "GWCSBNL": "serious",
    "GWCNM":   "serious",
    "GWCNL":   "serious",
    "BSBM":    "critical",
    "INSD":    "critical",
    "CBHPM":   "serious",
    "CBVPM":   "serious",
    "DRPS":    "serious",
}

SEVERITY_LEVELS: list[str] = ["minor", "normal", "serious", "critical"]

# ═══════════════════════════════════════════════════════════════════════════
# Standard 7-class training subset
# ═══════════════════════════════════════════════════════════════════════════
#
# These are the 7 classes for which annotated training data exists.
# Existing trained models use *name-based* class matching (the model's
# internal .names dict), so changing indices in DEFECT_CLASSES does NOT
# affect trained models — they map by name at inference time.
#
# The TRAIN_CLASSES list uses name-based lookup so it self-adjusts to
# any index changes in DEFECT_CLASSES above.

TRAIN_CLASSES: list[str] = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
]
TRAIN_NC: int = len(TRAIN_CLASSES)  # 7

# ═══════════════════════════════════════════════════════════════════════════
# 12-class training subset (expanded — all classes with annotated data)
# ═══════════════════════════════════════════════════════════════════════════
#
# Matches the 12-class indexing in data/Defect_dataset/defect_data.yaml.
# Classes 7-11 (RHTBNM, RHTBNL, BSBM, INSD, DRPS) were added based on
# available annotations in Defect_dataset (767 train instances total).
# The 4 ground-wire clamp classes (GWCSBNM/GWCSBNL/GWCNM/GWCNL) remain
# excluded — zero annotations exist.
#
# IMPORTANT: This list uses the Defect_dataset indexing (0-11), NOT the
# canonical 16-class indexing. CBHPM=5, CBVPM=6 in the dataset.

TRAIN_CLASSES_12: list[str] = [
    "VHBNM",   #  0
    "VHBNL",   #  1
    "SVHBNM",  #  2
    "SVHBNL",  #  3
    "SVHTNL",  #  4
    "CBHPM",   #  5
    "CBVPM",   #  6
    "RHTBNM",  #  7
    "RHTBNL",  #  8
    "BSBM",    #  9
    "INSD",    # 10
    "DRPS",    # 11
]
TRAIN_NC_12: int = len(TRAIN_CLASSES_12)  # 12

# ═══════════════════════════════════════════════════════════════════════════
# Component-type grouping (for multi-task auxiliary head)
# ═══════════════════════════════════════════════════════════════════════════
#
# Maps each defect class to its physical component type.  The auxiliary
# head predicts which component types are present in the image (multi-label
# BCE), providing a structural learning signal to the backbone.

COMPONENT_TYPES: list[str] = [
    "VHB",      # 0 — 垂直悬吊安装底座 (VHBNM, VHBNL)
    "SVHB",     # 1 — 单支垂直悬吊槽钢底座 (SVHBNM, SVHBNL)
    "SVHTN",    # 2 — 单支垂直悬吊槽钢上方 (SVHTNL)
    "CBH",      # 3 — 腕臂底座横向 (CBHPM)
    "CBV",      # 4 — 腕臂底座垂直 (CBVPM)
    "RHTBN",    # 5 — 刚性悬挂吊柱顶板 (RHTBNM, RHTBNL)
    "BSB",      # 6 — 汇流排中间接头 (BSBM)
    "INSD",     # 7 — 绝缘子 (INSD)
    "DRPS",     # 8 — 吊弦 (DRPS)
]
NUM_COMPONENT_TYPES: int = len(COMPONENT_TYPES)  # 9

# defect class name → component type index
_DEFECT_TO_COMPONENT: dict[str, int] = {
    "VHBNM": 0, "VHBNL": 0,
    "SVHBNM": 1, "SVHBNL": 1,
    "SVHTNL": 2,
    "CBHPM": 3,
    "CBVPM": 4,
    "RHTBNM": 5, "RHTBNL": 5,
    "BSBM": 6,
    "INSD": 7,
    "DRPS": 8,
}


def defect_cls_to_component_type(cls_id: int) -> int:
    """Map a 12-class defect index to its component-type index (0-8)."""
    name = TRAIN_CLASSES_12[cls_id]
    return _DEFECT_TO_COMPONENT[name]


def build_component_type_matrix() -> list[list[int]]:
    """Return a (12, 9) binary matrix: row i has 1 at component type of class i."""
    mat = [[0] * NUM_COMPONENT_TYPES for _ in range(TRAIN_NC_12)]
    for cls_id in range(TRAIN_NC_12):
        comp_id = defect_cls_to_component_type(cls_id)
        mat[cls_id][comp_id] = 1
    return mat

# ═══════════════════════════════════════════════════════════════════════════
# Legacy aliases — for backward compatibility with older checkpoint names
# ═══════════════════════════════════════════════════════════════════════════

LEGACY_ALIASES: dict[str, str] = {
    # Old code names that appear in older models → canonical code
    "CBHSL": "CBHPM",   # 旧: U型抱箍螺栓横轴松动 → 腕臂底座横向销钉缺口
    "CBHNM": "CBHPM",   # 旧: U型抱箍螺栓平头螺母缺失 → 腕臂底座横向销钉缺口
    "CBHNL": "CBHPM",   # 旧: U型抱箍螺栓平头螺母松动 → 腕臂底座横向销钉缺口
    "VJBNM": "VHBNM",   # 旧: 垂直J型螺栓正常缺失 → 垂直悬吊安装底座螺母缺失
    "VJBNL": "VHBNL",   # 旧: 垂直J型螺栓正常松动 → 垂直悬吊安装底座螺母松动
    "VJBSL": "SVHTNL",  # 旧: 垂直J型螺栓杆松动 → 单支垂直悬吊槽钢上方螺母松动
    "CWZJ":  "DRPS",    # 旧: 接触线ZJ型缺陷 → 吊弦不受力
    "JXP":   "INSD",    # 旧: JXP型绝缘子缺陷 → 绝缘子破损
    "GJJ":   "BSBM",    # 旧: 钢结构锈蚀/变形 → 汇流排中间接头螺栓缺失
    "DDL":   "DRPS",    # 旧: 吊弦松动/断裂 → 吊弦不受力
    "BYLZ":  "INSD",    # 旧: 其他异常 → 绝缘子破损 (保守归并)
}

# ═══════════════════════════════════════════════════════════════════════════
# Lookup helpers
# ═══════════════════════════════════════════════════════════════════════════

_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DEFECT_CLASSES)}
_ID_TO_NAME: dict[int, str] = {idx: name for idx, name in enumerate(DEFECT_CLASSES)}


def get_class_id(name: str) -> Optional[int]:
    """Return the integer class ID for a defect class code, or ``None``."""
    return _NAME_TO_ID.get(name.upper())


def get_class_name(idx: int) -> Optional[str]:
    """Return the defect class code for an integer ID, or ``None``."""
    return _ID_TO_NAME.get(idx)


def get_cn_name(code: str) -> str:
    """Return the Chinese name for a defect class code.

    Falls back to the code itself if no mapping exists.
    """
    return CN_NAME_MAP.get(code.upper(), code.upper())


def get_severity(code: str) -> str:
    """Return the severity level for a defect class code.

    Falls back to ``"normal"`` if no mapping exists.
    """
    return SEVERITY_MAP.get(code.upper(), "normal")


def validate_class_id(cls_id: int) -> bool:
    """Return ``True`` if *cls_id* is a valid defect class index."""
    return 0 <= cls_id < NC


def validate_class_name(name: str) -> bool:
    """Return ``True`` if *name* is a valid defect class code (case-insensitive)."""
    return name.upper() in _NAME_TO_ID


def resolve_canonical(code: str) -> str:
    """Resolve a legacy class code to its canonical equivalent.

    Returns the canonical code if a mapping exists, otherwise returns
    the input code unchanged.
    """
    return LEGACY_ALIASES.get(code.upper(), code.upper())
