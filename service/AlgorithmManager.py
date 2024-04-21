# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 15:47
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import importlib
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    "AlgorithmManager"
]


class AlgorithmManager(object):
    """
    算法管理器
    """

    def __init__(self, logger, conf, oss):
        """
        初始化算法管理器

        :param logger: 日志器
        :param conf: 算法服务配置
        :param oss: oss服务
        """
        # 初始化参数
        self.logger = logger
        self.conf = conf
        self.oss = oss

        # 构建所有功能模块的服务，全局处理器线程池，全局输出器线程池
        self.services, self.processor_thread_pool, self.exporter_thread_pool = self._build_services()

    def _build_services(self):
        """
        构建所有功能模块的服务

        :return: 所有功能模块服务的列表, 全局处理器线程池, 全局输出器线程池
            所有功能模块服务的列表:其中每一个功能模块服务是一个字典，字典包括：receiver,processor,exporter
        """
        # 初始化全局处理器线程池
        processor_thread_pool = ThreadPoolExecutor(max_workers=self.conf["processor_max_workers"])

        # 初始化全局输出器线程池
        exporter_thread_pool = ThreadPoolExecutor(max_workers=self.conf["exporter_max_workers"])

        # 初始化服务列表
        services = []
        # 遍历服务配置
        for service_conf in self.conf["services"]:
            # 初始化exporter
            exporter_module = importlib.import_module(service_conf["exporter_module"])  # 动态导入模块
            exporter_class = getattr(exporter_module, service_conf["exporter_class"])  # 获取类
            exporter = exporter_class(  # 实例化对象
                self.logger,
                {
                    "bootstrap_servers": self.conf["bootstrap_servers"],
                    "exporter_topic": service_conf["exporter_topic"]
                },
                exporter_thread_pool,
                self.oss
            )
            # 初始化processor
            processor_module = importlib.import_module(service_conf["processor_module"])  # 动态导入模块
            processor_class = getattr(processor_module, service_conf["processor_class"])  # 获取类
            processor = processor_class(  # 实例化对象
                self.logger,
                processor_thread_pool,
                export_func=exporter.export
            )
            # 初始化receiver
            receiver_module = importlib.import_module(service_conf["receiver_module"])  # 动态导入模块
            receiver_class = getattr(receiver_module, service_conf["receiver_class"])  # 获取类
            receiver = receiver_class(  # 实例化对象
                self.logger,
                {
                    "bootstrap_servers": self.conf["bootstrap_servers"],
                    "group_id": self.conf["group_id"],
                    "auto_offset_reset": self.conf["auto_offset_reset"],
                    "receiver_topic": service_conf["receiver_topic"],
                    "timeout_ms": service_conf["timeout_ms"],
                    "max_records": service_conf["max_records"]
                },
                num_workers=service_conf["num_workers"],
                process_func=processor.process
            )
            # 将服务添加到列表
            services.append({
                "receiver": receiver,
                "processor": processor,
                "exporter": exporter
            })

        return services, processor_thread_pool, exporter_thread_pool

    def start(self):
        # 依次启动所有服务的receiver
        for service in self.services:
            service["receiver"].start()

    def join(self):
        # 依次阻塞所有服务的receiver
        for service in self.services:
            service["receiver"].join()
        # 关闭全局处理器线程池
        self.processor_thread_pool.shutdown(wait=True)
        # 关闭全局输出器线程池
        self.exporter_thread_pool.shutdown(wait=True)
        # 依次关闭所有服务的exporter
        for service in self.services:
            service["exporter"].stop()

    def stop(self):
        # 依次关闭所有服务的receiver
        for service in self.services:
            service["receiver"].stop()
