# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/19 12:32
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import json
import threading
from kafka import KafkaConsumer
from abc import ABC, abstractmethod

from entity.VideoAnalysisEntity import VideoAnalysisEntity

__all__ = [
    "AbstractProcessor"
]


class AbstractProcessor(ABC):
    """
    Processor的抽象类
    """
    _logger = None  # 日志器

    _thread_pool = None  # 处理器全局线程池
    _export_func = None  # 下一步处理函数

    @abstractmethod
    def _process(self, *args, **kwargs):
        pass

    def process(self, message):
        self._thread_pool.submit(self._process, message)
