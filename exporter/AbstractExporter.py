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
from dataclasses import asdict
from kafka import KafkaProducer
from abc import ABC, abstractmethod


from entity.VideoAnalysisEntity import VideoAnalysisEntity

__all__ = [
    "AbstractExporter"
]


class AbstractExporter(ABC):
    """
    Exporter的抽象类
    """
    _logger = None  # 日志器

    _producer_conf = None  # 消息队列生产者配置
    _producer = None  # 消息队列生产者
    _topic = ""  # 消息队列生产者的topic

    _thread_pool = None  # 输出器全局线程池

    @classmethod
    def build_producer(cls, conf):
        # 生产者配置
        producer_config = {
            'bootstrap_servers': conf["bootstrap_servers"],
            'value_serializer': lambda m: json.dumps(asdict(m), default=VideoAnalysisEntity.serialize_complex_types).encode('utf-8')
        }
        # 初始化生产者
        producer = KafkaProducer(**producer_config)

        # 返回生产者
        return producer

    @abstractmethod
    def _export(self, value):
        pass

    def export(self, value):
        self._thread_pool.submit(self._export, value)

    def stop(self):
        self._producer.close()
