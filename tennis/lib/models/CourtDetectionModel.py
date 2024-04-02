# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:45
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import onnxruntime

from .AbstractModel import AbstractModel


class CourtDetectionModel(AbstractModel):
    """
    球场检测模型
    """

    def __init__(self):
        print("初始化球场检测模型对象")

    def init_model(self, model_path):
        print("初始化球场检测模型")

    def _pre_process(self, data):
        print("球场检测模型中的预处理")

    def inference(self, data):
        self._pre_process(data)
        print("球场检测模型中的推理")
        self._post_process(data)

    def _post_process(self, data):
        print("球场检测模型中的后处理")
