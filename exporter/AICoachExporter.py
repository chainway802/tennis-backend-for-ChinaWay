# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:45
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from exporter.AbstractExporter import AbstractExporter

__all__ = [
    "AICoachExporter"
]


class AICoachExporter(AbstractExporter):
    """
    AI教练Exporter
    """

    def __init__(self, logger, producer_conf, thread_pool, oss):
        """
        初始化AI教练Exporter

        :param logger: 日志器
        :param producer_conf: 消息队列生产者配置
        :param thread_pool: 输出器全局线程池
        :param oss: oss服务
        """
        # 初始化参数
        self._logger = logger  # 日志器
        self._producer_conf = producer_conf  # 消息队列生产者配置
        self._producer = super().build_producer(self._producer_conf)  # 消息队列生产者
        self._topic = self._producer_conf["exporter_topic"]  # 消息队列生产者的topic
        self._thread_pool = thread_pool  # 输出器全局线程池
        self._oss = oss  # oss服务

    def _export(self, value, *args, **kwargs):
        pass

