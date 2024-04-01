# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 19:31
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from __future__ import annotations
from typing import Dict, Tuple, Sequence, Optional, Union

import numpy as np

from .lib import court


__all__ = [
    "detect_court"
]


def detect_court(frame: np.ndarray, pmatrix: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取当前图像的球场线，根据是否传入透视变换矩阵可以采用不同的方法
    :param frame: 当前帧图像
    :param pmatrix: 参考球场到图像中的球场的透视变换矩阵
    :return: 当前帧图像中构成球场线的线条，参考球场到图像中的球场的透视变换矩阵
            线条的shape为(40, )，其中连续的每4个数(x1, y1, x2, y2)表示一条直线，一共10条直线
            透视变换矩阵的shape为(3, 3)
    """
    # 判断检测球场线还是跟踪微调球场线
    if pmatrix is None:
        # 检测球场线
        lines, pmatrix = court.detect_court(frame)
    else:
        # 跟踪微调球场线
        lines, pmatrix = court.track_court(frame, pmatrix, cdist=None)

    return lines, pmatrix
