# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:22
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
import numpy as np
from .TRTModel import TRTModel
from ..utils.process import keypoint_normalization_within_frame, update_action_counter
class PlayerActionModel(TRTModel):

    def __init__(self, window_size=30):
        super().__init__(max_batch_size=1)
        self.kptseq_with_id = {}
        self.actionseq_with_id = {}
        self.actioncounter_with_id = {}
        self.action_prob_sequences_with_id = {}
        self.action_timestamps_with_id = {}
        self.window_size = window_size

    def _pre_process(self):
        pass

    def inference(self, keypoints, primary_id, frame_id):
        # 初始化字典
        if primary_id not in self.kptseq_with_id:
            self.kptseq_with_id[primary_id] = []
        if primary_id not in self.actionseq_with_id:
            self.actionseq_with_id[primary_id] = [0 for _ in range(self.window_size-1)]
        if primary_id not in self.actioncounter_with_id:
            # 将其他id的动作计数累加到primary_id中
            self.actioncounter_with_id[primary_id] = {0: 0, 1: 0, 2: 0, 3: 0}
            for id, action_counter in self.actioncounter_with_id.items():
                if id != primary_id:
                    for action_type, count in action_counter.items():
                        self.actioncounter_with_id[primary_id][action_type] += count
        if primary_id not in self.action_prob_sequences_with_id:
            self.action_prob_sequences_with_id[primary_id] = {0: [0 for _ in range(self.window_size-1)], 1: [0 for _ in range(self.window_size-1)], 
                                                    2: [0 for _ in range(self.window_size-1)], 3: [0 for _ in range(self.window_size-1)]}
        if primary_id not in self.action_timestamps_with_id:
            self.action_timestamps_with_id[primary_id] = {0: [], 1: [], 2: [], 3: []}
            for id, action_timestamps in self.action_timestamps_with_id.items():
                if id != primary_id:
                    for action_type, timestamps in action_timestamps.items():
                        self.action_timestamps_with_id[primary_id][action_type].extend(timestamps)
        self.kptseq_with_id[primary_id].append(keypoints)
        
        # 处理窗口
        if len(self.kptseq_with_id[primary_id]) >= self.window_size:
            self.kptseq_with_id[primary_id] = self.kptseq_with_id[primary_id][-self.window_size:]
            current_window = np.array(self.kptseq_with_id[primary_id])
            current_window = keypoint_normalization_within_frame(current_window).reshape(1, self.window_size, -1)
            # clasify_result = action_model(current_window)['output']
            clasify_result = super().inference(current_window)
            action_probs = clasify_result['output'].squeeze().tolist()
            for action_type, prob in enumerate(action_probs):
                self.action_prob_sequences_with_id[primary_id][action_type].append(prob)
                # action_prob_sequences_with_id_static[primary_id][action_type].append(prob)
            shot_type1 = np.argmax(clasify_result)
            self.actionseq_with_id[primary_id].append(shot_type1)
        
        update_action_counter(self.action_prob_sequences_with_id, self.actioncounter_with_id, 20, self.action_timestamps_with_id, frame_id)
        self._post_process(primary_id)
        return self.actioncounter_with_id, self.action_timestamps_with_id
        
    def _post_process(self, primary_id):
        self.actioncounter_with_id = {primary_id:self.actioncounter_with_id[primary_id]}
        self.action_prob_sequences_with_id = {primary_id:self.action_prob_sequences_with_id[primary_id]}
        self.actionseq_with_id = {primary_id:self.actionseq_with_id[primary_id][-2:]}
        self.kptseq_with_id = {primary_id:self.kptseq_with_id[primary_id]}
        self.action_timestamps_with_id = {primary_id:self.action_timestamps_with_id[primary_id]}
