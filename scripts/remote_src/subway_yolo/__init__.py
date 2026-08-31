# -*- coding: utf-8 -*-
"""兼容 shim 包：旧权重 pickle 引用 subway_yolo.* 与 subway_yolo.nn.*。"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from subway_defect.modules import CoordAtt, DeformConv2d, EMA, LSK, SimAM
from subway_defect.modules.AuxHead import AuxClassifyHead

AuxHead = AuxClassifyHead
__all__ = ["CoordAtt", "DeformConv2d", "EMA", "LSK", "SimAM",
           "AuxClassifyHead", "AuxHead"]
