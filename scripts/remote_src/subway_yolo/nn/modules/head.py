# -*- coding: utf-8 -*-
import ultralytics.nn.modules.head as _u


def __getattr__(name):
    return getattr(_u, name)
