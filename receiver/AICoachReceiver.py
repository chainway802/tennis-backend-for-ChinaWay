# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:42
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import threading

from receiver.AbstractReceiver import AbstractReceiver

__all__ = [
    "AICoachReceiver"
]


class AICoachReceiver(AbstractReceiver):
    """
    AI教练Receiver
    """

    def __init__(self, logger, consumer_conf, num_workers=1, process_func=None):
        """
        初始化AI教练Receiver

        :param logger: 日志器
        :param consumer_conf: 消息队列消费者配置
        :param num_workers: 线程数
        :param process_func: 下一步处理函数
        :return: 视频剪辑Receiver实例
        """
        self._logger = logger  # 日志器
        self._consumer_conf = consumer_conf  # 消息队列消费者配置
        self._consumer = super().build_consumer(self._consumer_conf)  # 消息队列消费者
        self._num_workers = num_workers  # 线程数
        self._process_func = process_func  # 下一步处理函数
        self._threads = []  # 线程列表
        self._is_running = False  # 运行状态
