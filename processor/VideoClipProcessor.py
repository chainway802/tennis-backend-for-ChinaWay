# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:44
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from processor.AbstractProcessor import AbstractProcessor
from util.video_process import VideoLoader
from oss.OSSHelper import OSSHelper
from util.io import load_yaml_config
from tennis.player_detector import PlayerDetector
from tennis.player_tracker import SortTracker
from tennis.player_poser import PlayerPoser
from tennis.play_action import PlayerAction
from tennis.auto_editor import AutoEditor
import time
import os
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
        self.conf = load_yaml_config(r"./config/config.yaml")
        self._oss = OSSHelper(**self.conf["oss"])
        self.local_file_path = None

    def _process(self, message, *args, **kwargs):
        print(message)
        # 读取视频, 获得视频信息
        # self.local_file_path = self._oss.download_file(message.value.videoUrl)
        videof = VideoLoader('/aidata/mmfuck/tennis-backend-new/tennis-backend/temp/65b808b6-8015-4c79-bd4e-d7302c909502.mp4')
        fps, w, h, video_duration_frames = videof.get_video_info()
        # 初始化算法模型
        player_detector = PlayerDetector(human_thr=0.5, racket_thr=0.3, human_area_sort=True)
        player_tracker = SortTracker(max_age=30, min_hits=3, resolution=(w, h))
        player_poser = PlayerPoser(channel_convert=True)
        player_action = PlayerAction()
        editor = AutoEditor(video_duration_frames, pre_serve_filter = True, pre_serve_window=120, 
                 hit_labels=[1,2], serve_label=3, hit_filter=True, hit_minimum_distance=45, hit_isolated_distance=210, 
                 rally_threshold=180, rally_action_count=3, pre_rally_window=30, post_rally_window=60)
        # 初始化相关id和序列
        primary_id = None
        frame_ind = 0
        start = time.time()
        actioncounter_with_id = None
        action_timestamps_with_id = None
        temp_shot_count = None
        # cap = cv2.VideoCapture(message.value.videoUrl)
        # 遍历检测
        while True:
            t1 = time.time()
            ret, frame = videof.read_frame()
            print('read time:', time.time() - t1)
            if not ret:
                break
            frame_ind += 1
            print(frame_ind)
            human_bboxes, racket_bboxes, ball_bboxes = player_detector.detect(frame)
            trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)
            print(human_bboxes)
            t2 = time.time()
            if trackers is not None and primary_id is not None:
                player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
                kpts = player_poser.detect(frame, player_bbox)
                print('pose time:', time.time() - t2)
                t3 = time.time()
                if kpts is not None:
                    actioncounter_with_id, action_timestamps_with_id = player_action.detect(kpts, int(primary_id), frame_ind)
                print('action time:', time.time() - t3)
            if frame_ind % 300 == 1 and primary_id is not None and primary_id in actioncounter_with_id:
                if temp_shot_count == actioncounter_with_id[primary_id]:
                    primary_id = None
                else:
                    temp_shot_count = actioncounter_with_id[primary_id].copy()

        # 剪辑和输出视频
        output_folder = os.path.join(os.path.dirname(__file__), "temp")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, "temp.mp4")
        
        # 获得第一个目标的动作时间戳
        action_timestamps = action_timestamps_with_id[list(action_timestamps_with_id.keys())[0]]
        # 获得高质量片段的区间
        rally_intervals = editor.get_rallys(action_timestamps)

        # 创建视频片段并拼接
        videof.edit_video(rally_intervals, output_path=output_path)     
        used_time = time.time() - start
        message.value.analyzeTime = used_time
        self._export_func(message.value, output_path)

