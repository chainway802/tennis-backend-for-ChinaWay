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
    "VideoClipExporter"
]


class VideoClipExporter(AbstractExporter):
    """
    视频剪辑Exporter
    """

    def __init__(self, logger, producer_conf, thread_pool, oss):
        """
        初始化视频剪辑Exporter

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

    def _export(self, value, url, *args, **kwargs):
        # 上传视频
        try:
            processUrl = self._oss.upload_file(value.id, r"./static/video/video_input1.mp4")
        except:
            print("upload error")
        # 保存剪辑视频地址
        value.processUrl = processUrl
        print(value, '\n')
        # 将结果发送到消息队列
        self._producer.send(self._topic, key=str(value.id).encode('utf-8'), value=value)
        self._producer.flush()
