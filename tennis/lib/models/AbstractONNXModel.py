# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 15:45
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from abc import ABC, abstractmethod

__all__ = [
    "AbstractONNXModel"
]


class AbstractONNXModel(object):
    """
    ONNX模型的抽象类
    """
    @abstractmethod
    def init_model(self, model_path):
        pass

    @abstractmethod
    def _pre_process(self, data):
        pass

    @abstractmethod
    def inference(self, data):
        pass

    @abstractmethod
    def _post_process(self, data):
        pass
