# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/02 15:20
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
# __all__ 的作用是定义了在使用 from xxx import * 时，可以被导入的模块名
__all__ = [
    "AutoEditor"
]


class AutoEditor(object):
    '''
    video_duration_frames: 视频总帧数
    action_timestamps: 动作时间戳
    pre_serve_filter: 是否过滤掉发球前的帧数窗口
    pre_serve_window: 过滤掉发球前的帧数窗口(原因是，有时候发球的前摇会被识别成某个击球)
    hit_labels: 除发球外的击球标签
    serve_label: 发球标签
    hit_filter: 是否过滤掉相邻过近的击球和孤立的击球
    hit_minimum_distance: 击球的最小帧数跨度
    hit_isolated_distance: 如果一个击球发生在某个稀疏独立的时间区间（距离上一个击球和下一个击球的帧数跨度都超过阈值），则过滤掉这次击球。用于筛选出目标在自己随意挥拍时产生的干扰击球
    rally_threshold: 一个回合内击球的最大帧数跨度
    rally_action_count: 一个回合内最少的击球次数
    pre_rally_window: 在一个回合的开始往前移动的帧数窗口
    post_rally_window: 在一个回合的结束往后移动的帧数窗口
    filtered_intervals: 被过滤掉的击球时间区间
    '''
    
    def __init__(self, video_duration_frames, pre_serve_filter = True, pre_serve_window=120, 
                 hit_labels=['1','2'], serve_label='3', hit_filter=True, hit_minimum_distance=45, hit_isolated_distance=210, 
                 rally_threshold=180, rally_action_count=3, pre_rally_window=30, post_rally_window=60):
        self.video_duration_frames = video_duration_frames
        self.pre_serve_filter = pre_serve_filter
        self.pre_serve_window = pre_serve_window
        self.hit_labels = hit_labels
        self.serve_label = serve_label
        self.hit_filter = hit_filter
        self.hit_minimum_distance = hit_minimum_distance
        self.hit_isolated_distance = hit_isolated_distance
        self.rally_threshold = rally_threshold
        self.rally_action_count = rally_action_count
        self.pre_rally_window = pre_rally_window
        self.post_rally_window = post_rally_window
        self.filtered_intervals = []
    
    def serve_filt(self, action_timestamps):
        '''
        过滤掉发球前的帧数窗口
        '''    
        action_timestamps_sf = action_timestamps.copy()
        # 遍历每个发球动作的起始时间
        for serve_start, serve_end in action_timestamps[self.serve_label]:
            serve_window_start = serve_start - self.pre_serve_window
            
            # 检查每个正手和反手动作是否在发球前x帧内完整包含
            for action_type in self.hit_labels:
                for action_start, action_end in action_timestamps[action_type]:
                    # 检查动作是否完整地在发球前x帧内
                    if action_start >= serve_window_start and action_end <= serve_start:
                        self.filtered_intervals.append([action_start, action_end])

        # 去除被过滤的动作
        for action_type in self.hit_labels:
            new_intervals = [interval for interval in action_timestamps[action_type] if interval not in self.filtered_intervals]
            action_timestamps_sf[action_type] = new_intervals

        return action_timestamps_sf
    
    def hit_filt(self, action_timestamps):
        '''
        过滤掉相邻过近的击球和孤立的击球
        '''
        action_timestamps_hf = action_timestamps.copy()
        action_intervals = []
        action_labels = self.hit_labels + [self.serve_label]
        for label in action_labels:
            action_intervals += action_timestamps[label]
        action_intervals = sorted(action_intervals, key = lambda x:x[0])
        # 找出与前后两次动作起始时间间隔45-300帧的动作时间区间
        for i, interval in enumerate(action_intervals):
            # 检查与前一个动作的间隔
            if i > 0 and self.hit_minimum_distance > (interval[0] - action_intervals[i-1][0]):
                if (interval[1] - interval[0]) > (action_intervals[i-1][1] - action_intervals[i-1][0]):
                    self.filtered_intervals.append(action_intervals[i-1])
                else:
                    self.filtered_intervals.append(interval)

        for action_type in self.hit_labels:
            new_intervals = [interval for interval in action_timestamps[action_type] if interval not in self.filtered_intervals]
            action_timestamps_hf[action_type] = new_intervals

        for i, interval in enumerate(action_intervals):
            # 检查与前一个动作的间隔
            if i > 0 and self.hit_isolated_distance < (interval[0] - action_intervals[i-1][0]):
                # 如果是列表中的第一个元素，或者不是第一个但满足与前一个动作的间隔超过300
                if i < (len(action_intervals) - 1) and self.hit_isolated_distance < (action_intervals[i+1][0] - interval[0]):
                    # 如果是列表中的最后一个元素，或者不是最后一个但满足与后一个动作的间隔超过300
                    self.filtered_intervals.append(interval)

        # 去除被过滤的动作
        for action_type in self.hit_labels:
            new_intervals = [interval for interval in action_timestamps_hf[action_type] if interval not in self.filtered_intervals]
            action_timestamps_hf[action_type] = new_intervals
            
        return action_timestamps_hf
    def get_rallys(self, action_timestamps):
        '''
        获取回合时间区间
        '''
        if self.pre_serve_filter:
            action_timestamps = self.serve_filt(action_timestamps)
        if self.hit_filter:
            action_timestamps = self.hit_filt(action_timestamps)

        # 合并所有区间
        action_intervals = []
        action_labels = self.hit_labels + [self.serve_label]
        for label in action_labels:
            action_intervals += action_timestamps[label]
        # 按开始时间排序
        action_intervals = sorted(action_intervals, key = lambda x:x[0])

        # 初始化回合列表
        rounds = []
        current_round = [action_intervals[0]]

        for i in range(1, len(action_intervals)):
            previous_interval = current_round[-1]
            current_interval = action_intervals[i]
            
            # 检查当前击球与前一个击球的时间间隔
            if current_interval[0] - previous_interval[1] < self.rally_threshold:
                # 如果间隔小于阈值，将当前击球加入到当前回合
                current_round.append(current_interval)
            else:
                # 否则，当前回合结束，开始新的回合
                rounds.append(current_round)
                current_round = [current_interval]

        # 添加最后一个回合
        if current_round:
            rounds.append(current_round)

        # 去除过短的回合
        rounds = [round for round in rounds if len(round) > self.rally_action_count]

        # 扩展回合时间区间
        rally_intervals = [[max(0,round[0][0]-self.pre_rally_window), min(round[-1][1]+self.post_rally_window, self.video_duration_frames)] for round in rounds]
        return rally_intervals
        
    