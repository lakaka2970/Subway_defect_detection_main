# -*- coding: utf-8 -*-
"""兼容 shim：旧训练代码把自定义模块 pickle 为 subway_yolo.*，
此模块将其重导出，使 stage4_best.pt 可在当前仓库结构下加载。
需与本目录下的 subway_defect/ 包一同置于 PYTHONPATH。
"""
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
