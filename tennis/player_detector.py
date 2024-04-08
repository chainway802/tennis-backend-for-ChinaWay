# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

from .lib.models.PlayerDetectionModel import PlayerDetectionModel
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "PlayerDetector"
]

player_det_model = PlayerDetectionModel()

class PlayerDetector(object):
    def __init__(self, engine_path, human_thr=0.4, racket_thr=0.3, human_area_sort=True, racket_area_sort=True):
        self.channel_convert = False
        self.human_max_numbers_by_area = 3
        self.racket_max_numbers_by_area = 2
        self.human_max_numbers = 2
        self.racket_max_numbers = 1
        self.human_thr = human_thr
        self.racket_thr = racket_thr
        self.human_area_sort = human_area_sort
        self.racket_area_sort = racket_area_sort
        player_det_model.init_model(engine_path)
        
        
    def detect(self, frame):
        human_bboxes_post, racket_bboxes_post = player_det_model.inference(frame, self.channel_convert, self.human_max_numbers_by_area, 
                                                                           self.racket_max_numbers_by_area, self.human_max_numbers, self.racket_max_numbers, 
                                                                           self.human_thr, self.racket_thr, self.human_area_sort, self.racket_area_sort)
        return human_bboxes_post, racket_bboxes_post