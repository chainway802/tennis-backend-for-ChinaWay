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
    "AbstractReceiver"
]


class AbstractReceiver(ABC):
    """
    Receiver的抽象类
    """
    _logger = None  # 日志器

    _consumer_conf = None  # 消息队列消费者配置
    _consumer = None  # 消息队列消费者

    _num_workers = 1  # 线程数
    _threads = []  # 线程列表

    _process_func = None  # 下一步处理函数
    _is_running = False  # 运行状态

    @classmethod
    def build_consumer(cls, conf):
        # 消费者配置
        consumer_config = {
            'bootstrap_servers': conf["bootstrap_servers"],
            'group_id': conf["group_id"],
            'auto_offset_reset': conf["auto_offset_reset"],
            'value_deserializer': lambda m: VideoAnalysisEntity(**json.loads(m.decode('utf-8')))
        }
        # 初始化消费者
        consumer = KafkaConsumer(conf["receiver_topic"], **consumer_config)

        # 返回消费者
        return consumer

    def _receive_data(self):
        # 不断地从消息队列中获取数据
        while True:
            # 从消息队列中获取某个topic和分区的多条消息
            records = self._consumer.poll(timeout_ms=self._consumer_conf["timeout_ms"], max_records=self._consumer_conf["max_records"])
            # 如果消息队列不为空
            if records:
                for messages in records.values():
                    for message in messages:
                        # 将数据发送到下一步的处理函数
                        self._process_func(message)
                        # print(message)

            # 判断是否满足结束条件
            if not self._is_running:
                break

    def start(self):
        # 改变运行状态
        self._is_running = True
        # 依次启动多个线程
        for i in range(self._num_workers):
            t = threading.Thread(target=self._receive_data, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        # 改变运行状态
        self._is_running = False
        self.join()
        self._consumer.close()

    def join(self):
        for t in self._threads:
            t.join()
