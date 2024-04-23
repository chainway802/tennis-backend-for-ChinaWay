# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
from .lib.models.PlayerPoseEstimationModel import PlayerPoseEstimationModel

# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerPoser"
]
weights_dir = os.path.join(os.path.dirname(__file__), "lib", "weights")

player_pose_model = PlayerPoseEstimationModel(
    engine_file_path=os.path.join(weights_dir, "vitpose_small.engine"),
    onnx_file_path=os.path.join(weights_dir, "vitpose_small.onnx"),
    use_onnx=False,
    precision_flop='FP16',
    img_size=(256, 192),
    dynamic_shapes={},
    dynamic_max_batch_size=1
)
player_pose_model.init_model()


class PlayerPoser(object):
    '''
    channel_convert: 是否将channel维度的顺序调换为(2,1,0)
    '''

    def __init__(self, channel_convert=True):
        self.channel_convert = channel_convert

    def detect(self, frame, bbox):
        kpts_post = player_pose_model.inference(frame, bbox, self.channel_convert)
        return kpts_post
