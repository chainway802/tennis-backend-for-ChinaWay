# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:22
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

from .TRTModel import TRTModel


class PlayerDetectionModel(TRTModel):

    def __init__(self):
        print("初始化球员检测模型")

    def _pre_process(self, data):
        print("球员检测模型预处理")

    def _post_process(self, data):
        print("球员检测模型后处理")

