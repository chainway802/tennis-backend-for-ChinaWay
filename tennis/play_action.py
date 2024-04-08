# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

from .lib.models.PlayerActionModel import PlayerActionModel
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerAction"
]

player_action_model = PlayerActionModel()

class PlayerAction(object):
    def __init__(self, engine_path):
        player_action_model.init_model(engine_path)
        
        
    def detect(self, keypoints, primary_id, frame_id):
        actioncounter_with_id, action_timestamps_with_id = player_action_model.inference(keypoints, primary_id, frame_id)
        return actioncounter_with_id, action_timestamps_with_id