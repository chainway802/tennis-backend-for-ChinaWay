# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 15:47
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import logging
import argparse

import util
from service.AlgorithmManager import AlgorithmManager
from oss.OSSHelper import OSSHelper

__all__ = [
    "AlgorithmService"
]


class AlgorithmService(object):
    """
    算法服务
    """

    def __init__(self, config_file_path):
        # 读取配置文件
        self.conf = util.load_yaml_config(config_file_path)

        # 解析命令行参数并更新配置
        self._parse_argument()

        # 初始化日志系统
        self.logger = self._init_logger()

        # 初始化oss服务
        self.oss = OSSHelper(**self.conf["oss"])

        # 初始化算法管理器
        self.algorithm_manager = AlgorithmManager(self.logger, self.conf, self.oss)

    def start(self):
        # 启动算法管理器
        self.algorithm_manager.start()
        # 阻塞算法服务主程序
        self.algorithm_manager.join()

    def stop(self):
        # 终止算法管理器
        self.algorithm_manager.stop()

    def _parse_argument(self):
        # 解析命令行参数
        parser = argparse.ArgumentParser()
        parser.add_argument("-nw1", type=int, help="读取video_clip订阅的消息队列的num_workers")
        parser.add_argument("-nw2", type=int, help="读取AI_coach订阅的消息队列的num_workers")
        parser.add_argument("-mw1", type=int, help="processor线程池的max_workers")
        parser.add_argument("-mw2", type=int, help="exporter线程池的max_workers")
        args = parser.parse_args()

        # 依次更新配置
        if args.nw1 is not None:
            self.conf["services"][0]["num_workers"] = args.nw1
        if args.nw2 is not None:
            self.conf["services"][1]["num_workers"] = args.nw2
        if args.mw1 is not None:
            self.conf["processor_max_workers"] = args.mw1
        if args.mw2 is not None:
            self.conf["exporter_max_workers"] = args.mw2

    def _init_logger(self):
        # 初始化日期器
        logger = logging.getLogger('logger')
        logger.setLevel(logging.WARNING)
        # 定义日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%m/%d/%Y %H:%M:%S'
        formatter = logging.Formatter(log_format, datefmt=date_format)
        # 创建并设置处理器
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        # 文件处理器
        file_handler = logging.FileHandler(self.conf["log_file_path"])
        file_handler.setFormatter(formatter)
        # 将处理器添加到日志器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger
