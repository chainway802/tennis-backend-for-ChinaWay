# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
from .lib.models.PlayerDetectionModel import PlayerDetectionModel
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerDetector"
]
weights_dir = os.path.join(os.path.dirname(__file__), "lib", "weights")

player_det_model = PlayerDetectionModel(
    engine_file_path=os.path.join(weights_dir, "yolov8s_FP16.trt"),
    onnx_file_path=os.path.join(weights_dir, "yolov8s.onnx"),
    use_onnx=False,
    precision_flop='FP16',
    img_size=(640, 640),
    dynamic_shapes={},
    dynamic_max_batch_size=1,
    model_type='YOLO'
)
player_det_model.init_model()

class PlayerDetector(object):
    '''
    human_area_sort: 是否先按人目标框面积降序筛选
    human_max_numbers_by_area: 按目标框面积降序筛选的最大人数
    human_max_numbers: 最终筛选的最大人数
    human_thr: 人的置信度阈值
    racket_area_sort: 是否先按拍子的目标框面积降序筛选
    racket_max_numbers_by_area: 按拍子目标框面积降序筛选的最大拍子数
    racker_max_numbers: 最终筛选的最大拍子数
    racket_thr: 拍子的置信度阈值
    '''
    def __init__(self, human_area_sort=False, human_max_numbers_by_area=10, human_max_numbers=4, human_thr=0.4,
                 racket_area_sort=False, racket_max_numbers_by_area=5, racket_max_numbers=2, racket_thr=0.3, channel_convert=False):
        self.channel_convert = channel_convert
        self.human_max_numbers_by_area = human_max_numbers_by_area
        self.racket_max_numbers_by_area = racket_max_numbers_by_area
        self.human_max_numbers = human_max_numbers
        self.racket_max_numbers = racket_max_numbers
        self.human_thr = human_thr
        self.racket_thr = racket_thr
        self.human_area_sort = human_area_sort
        self.racket_area_sort = racket_area_sort

        
        
    def detect(self, frame):
        '''
        return: 人的目标框列表，拍子的目标框列表
        '''
        human_bboxes_post, racket_bboxes_post, ball_bboxes_post = player_det_model.inference(frame, self.channel_convert, self.human_max_numbers_by_area, 
                                                                           self.racket_max_numbers_by_area, self.human_max_numbers, self.racket_max_numbers, 
                                                                           self.human_thr, self.racket_thr, self.human_area_sort, self.racket_area_sort)
        return human_bboxes_post, racket_bboxes_post, ball_bboxes_post