# -*- coding: utf-8 -*-
import ultralytics.nn.modules as _u

from subway_yolo import (AuxClassifyHead, AuxHead, CoordAtt, DeformConv2d,
                         EMA, LSK, SimAM)


def __getattr__(name):
    return getattr(_u, name)
