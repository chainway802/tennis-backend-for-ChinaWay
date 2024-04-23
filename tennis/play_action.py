# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
from .lib.models.PlayerActionModel import PlayerActionModel
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerAction"
]
weights_dir = os.path.join(os.path.dirname(__file__), "lib", "weights")

player_action_model = PlayerActionModel(
    engine_file_path=os.path.join(weights_dir, "action_classify.engine"),
    onnx_file_path=os.path.join(weights_dir, "action_classify.onnx"),
    use_onnx=False,
    precision_flop='FP16',
    dynamic_max_batch_size=1,
    dynamic_shapes={},
    window_size=30
)
player_action_model.init_model()

class PlayerAction(object):
    def __init__(self):
        pass
        
    def detect(self, keypoints, primary_id, frame_id):
        actioncounter_with_id, action_timestamps_with_id = player_action_model.inference(keypoints, primary_id, frame_id)
        return actioncounter_with_id, action_timestamps_with_id