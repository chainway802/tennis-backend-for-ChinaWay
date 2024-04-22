# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:44
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from processor.AbstractProcessor import AbstractProcessor

__all__ = [
    "VideoClipProcessor"
]


class VideoClipProcessor(AbstractProcessor):
    """
    视频剪辑Processor
    """

    def __init__(self, logger, thread_pool, export_func=None):
        """
        初始化视频剪辑Processor

        :param logger: 日志器
        :param thread_pool: 处理器全局线程池
        :param export_func: 输出器处理函数
        """
        # 初始化参数
        self._logger = logger  # 日志器
        self._thread_pool = thread_pool  # 处理器全局线程池
        self._export_func = export_func  # 输出器处理函数


    def _process(self, message, *args, **kwargs):
        print(message)
        self._export_func(message.value)


        # 加载视频
        video = cv2.VideoCapture(message.value.videoUrl)
        # 获取视频属性
        fps, total_frame_length, w, h = util.get_video_properties(video)
        # 初始化球场检测器
        court_detector = tennis.CourtDetector(max_age=40)

        # 初始化一些数据
        frame_ind = 0
        new_frames = []
        # 遍历所有视频帧
        while True:
            # 读取一帧
            ret, frame = video.read()
            frame_ind += 1  # 帧数累计

            # 成功读取帧
            if ret:
                # 检测第一帧的场地线
                if frame_ind == 1:
                    lines = court_detector.detect_court(frame)
                else:  # 其他帧跟踪场地线
                    lines = court_detector.detect_court(frame)
                # 在当前帧画出场地线
                for i in range(0, len(lines), 4):
                    x1, y1, x2, y2 = lines[i:i + 4]
                    new_frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                # 缩放图像尺寸
                new_frame = cv2.resize(new_frame, (w, h))
                # 将处理后的一帧添加到列表
                new_frames.append(new_frame)
            else:  # 视频结尾跳出循环
                break
            # 释放打开的视频
        video.release()
