# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

from .lib.models.PlayerPoseEstimationModel import PlayerPoseEstimationModel
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerPoser"
]

player_pose_model = PlayerPoseEstimationModel()

class PlayerPoser(object):
    def __init__(self, engine_path):
        self.channel_convert = True
        player_pose_model.init_model(engine_path)
        
        
    def detect(self, frame, bbox):
        kpts_post = player_pose_model.inference(frame, bbox, self.channel_convert)
        return kpts_post