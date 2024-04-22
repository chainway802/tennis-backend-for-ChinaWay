# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:44
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from processor.AbstractProcessor import AbstractProcessor

__all__ = [
    "AICoachProcessor"
]


class AICoachProcessor(AbstractProcessor):
    """
    AI教练Processor
    """

    def __init__(self, logger, thread_pool, export_func=None):
        """
        初始化AI教练Processor

        :param logger: 日志器
        :param thread_pool: 处理器全局线程池
        :param export_func: 输出器处理函数
        """
        # 初始化参数
        self._logger = logger  # 日志器
        self._thread_pool = thread_pool  # 处理器全局线程池
        self._export_func = export_func  # 输出器处理函数

    def _process(self, message, *args, **kwargs):
        pass
