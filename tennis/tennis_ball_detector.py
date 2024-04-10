# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/10 17:56
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import queue

from .lib.models.TennisBallDetectionModel import TennisBallDetectionModel

__all__ = [
    "TennisBallDetector"
]

weights_dir = os.path.join(os.path.dirname(__file__), "lib", "weights")

tennis_ball_detection_model = TennisBallDetectionModel(
    engine_file_path=os.path.join(weights_dir, "tracknet.trt"),
    onnx_file_path=os.path.join(weights_dir, "tracknet.onnx"),
    use_onnx=True,
    precision_flop="FP16",
    img_size=(360, 640),
    dynamic_shapes={},
    dynamic_max_batch_size=1
)

tennis_ball_detection_model.init_model()


class TennisBallDetector(object):
    """
    网球位置检测
    """

    def __init__(self, cache_len=8):
        """
        初始化网球检测器

        :param cache_len: 缓存队列长度
        """
        self.cache_len = cache_len
        # 初始化缓存队列
        self.q = queue.deque()
        for i in range(cache_len):
            self.q.appendleft(None)

    def detect_tennis_ball(self, frame):
        """
        检测网球位置
        :param frame: 当前帧图像
        :return: 返回网球位置(x, y)，如果没有检测到或者检测到多个网球则返回None
        """
        # 检测球的位置
        position = tennis_ball_detection_model.inference(frame)
        # 更新队列
        self.q.appendleft(position)
        self.q.pop()

        return position

    def get_cached_tennis_ball(self):
        """
        获取缓存的网球位置列表

        :return: 返回缓存的网球位置列表
        """
        ret = []
        for i in range(self.cache_len):
            ret.append(self.q[i])

        return ret
