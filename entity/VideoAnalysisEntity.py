# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/19 17:45
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Optional

from entity.VideoAnalyzeItemEntity import VideoAnalyzeItemEntity


@dataclass
class VideoAnalysisEntity:
    id: Optional[int] = None
    userId: Optional[str] = None
    videoName: Optional[str] = None
    videoUrl: Optional[str] = None
    processUrl: Optional[str] = None
    funcValue: Optional[str] = None
    analyzeTime: Optional[datetime] = None
    status: Optional[int] = None
    pinNum: Optional[int] = None
    scoreRet: Optional[str] = None
    serveNum: Optional[int] = None
    analyzePic: Optional[str] = None
    totalRounds: Optional[int] = None
    movLen: Optional[str] = None
    createdTime: Optional[datetime] = None
    videoType: Optional[str] = None
    analyzeItems: Optional[List[VideoAnalyzeItemEntity]] = None

    def __init__(self,
                 id: Optional[int] = None,
                 userId: Optional[str] = None,
                 videoName: Optional[str] = None,
                 videoUrl: Optional[str] = None,
                 processUrl: Optional[str] = None,
                 funcValue: Optional[str] = None,
                 analyzeTime: Optional[str] = None,
                 status: Optional[int] = None,
                 pinNum: Optional[int] = None,
                 scoreRet: Optional[str] = None,
                 serveNum: Optional[int] = None,
                 analyzePic: Optional[str] = None,
                 totalRounds: Optional[int] = None,
                 movLen: Optional[str] = None,
                 createdTime: Optional[str] = None,
                 videoType: Optional[str] = None,
                 analyzeItems: Optional[List[Dict]] = None
                 ):
        self.id = id
        self.userId = userId
        self.videoName = videoName
        self.videoUrl = videoUrl
        self.processUrl = processUrl
        self.funcValue = funcValue
        self.analyzeTime = (datetime.strptime(analyzeTime, '%Y-%m-%d %H:%M:%S') if analyzeTime is not None else analyzeTime)
        self.status = status
        self.pinNum = pinNum
        self.scoreRet = scoreRet
        self.serveNum = serveNum
        self.analyzePic = analyzePic
        self.totalRounds = totalRounds
        self.movLen = movLen
        self.createdTime = (datetime.strptime(createdTime, '%Y-%m-%d %H:%M:%S') if createdTime is not None else createdTime)
        self.videoType = videoType
        self.analyzeItems = ([VideoAnalyzeItemEntity(**item) for item in analyzeItems] if analyzeItems is not None else analyzeItems)

    @staticmethod
    def serialize_complex_types(obj):
        """自定义JSON编码器，用于处理不能直接序列化的数据类型"""
        if isinstance(obj, datetime) and obj is not None:
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, VideoAnalyzeItemEntity) and obj is not None:
            return asdict(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
