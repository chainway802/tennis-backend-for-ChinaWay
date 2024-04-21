# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/19 17:46
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class VideoAnalyzeItemEntity:
    id: Optional[int] = None
    analyzeId: Optional[int] = None
    orderNum: Optional[int] = None
    unitAction: Optional[str] = None
    actionDesc: Optional[str] = None
    actionOrder: Optional[int] = None
    shotType: Optional[str] = None
    gesRet: Optional[str] = None
    veloRet: Optional[str] = None
    isGroundPoint: Optional[str] = None
    groundPointPos: Optional[str] = None
    isInsideOut: Optional[int] = None
    movLen: Optional[int] = None
    similarValue: Optional[str] = None
    retPicData: Optional[str] = None
    inVideoTime: Optional[str] = None
    person: Optional[int] = None
    roundNum: Optional[int] = None

