# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 20:05
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import utils
import tennis
import time
import numpy as np
ACTION_TYPE = {0:'idle', 1:'forehand', 2:'backhand', 3:'serve'}

def test_detect_court(video_path):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
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

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(video_path.replace("input", "output"), fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()


def test_detect_player(video_path, engine_path):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    # 初始化球员检测器
    player_detector = tennis.PlayerDetector(engine_path, human_thr=0.4, racket_thr=0.1)
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
            # 检测球员
            human_bboxes, racket_bboxes = player_detector.detect(frame)
            # 在当前帧画出球员框
            if human_bboxes is not None:
                for bbox in human_bboxes:
                    x1, y1, x2, y2, _ = bbox
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
            # 在当前帧画出球拍框
            if racket_bboxes is not None:
                for bbox in racket_bboxes:
                    x1, y1, x2, y2, _ = bbox
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)

            new_frames.append(frame)
        else:  # 视频结尾跳出循环
            break
    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/aidata/mmfuck/output_backend_test.mp4', fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)

    # 释放打开的视频
    video.release()
    # 释放输出的视频
    output_video.release()


def test_track_player(video_path, engine_path):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    # 初始化球员检测器
    player_detector = tennis.PlayerDetector(engine_path, human_thr=0.4, racket_thr=0.1)
    # 初始化球员跟踪器
    player_tracker = tennis.SortTracker(max_age=30, min_hits=3, resolution=(w, h))
    # 球员id
    primary_id = None
    frame_ind = 0
    new_frames = []
    start = time.time()
    while True:
        # 读取一帧
        ret, frame = video.read()
        print(frame.shape)
        frame_ind += 1  # 帧数累计
        # 成功读取帧
        if ret:
            # 检测球员
            human_bboxes, racket_bboxes = player_detector.detect(frame)
            # sort跟踪器更新      
            trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)
            # 在当前帧画出球员框
            if trackers is not None and primary_id is not None:
                player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
                x1, y1, x2, y2, _ = player_bbox
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                frame = cv2.putText(frame, f"Player ID: {primary_id}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            new_frames.append(frame)
        else:  # 视频结尾跳出循环
            break
    print(f"FPS: {frame_ind / (time.time() - start)}")
    # 释放打开的视频
    video.release()

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/aidata/mmfuck/output_backend_test.mp4', fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()


def test_pose_player(video_path, det_engine, pose_engine):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    # 初始化球员检测器
    player_detector = tennis.PlayerDetector(det_engine, human_thr=0.3, racket_thr=0.3, human_area_sort=True)
    # 初始化球员跟踪器
    player_tracker = tennis.SortTracker(max_age=30, min_hits=3, resolution=(w, h))
    # 初始化球员姿态器
    player_poser = tennis.PlayerPoser(pose_engine)
    # 球员id
    primary_id = None
    frame_ind = 0
    new_frames = []
    start = time.time()
    while True:
        # 读取一帧
        ret, frame = video.read()
        frame_ind += 1  # 帧数累计
        # 成功读取帧
        if ret:
            # 检测球员
            human_bboxes, racket_bboxes = player_detector.detect(frame)
            # sort跟踪器更新      
            trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)

            if trackers is not None and primary_id is not None:
                player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
                kpts = player_poser.detect(frame, player_bbox)
                if kpts is not None:
                    for kpt in kpts:
                        x, y = kpt
                        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

            cv2.imwrite('frame.jpg', frame)
            new_frames.append(frame)
        else:  # 视频结尾跳出循环
            break
    print(f"FPS: {frame_ind / (time.time() - start)}")
    # 释放打开的视频
    video.release()

    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/aidata/mmfuck/output_backend_test.mp4', fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()


def test_action_player(video_path, det_engine, pose_engine, action_engine):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)
    # 初始化球员检测器
    player_detector = tennis.PlayerDetector(det_engine, human_thr=0.2, racket_thr=0.2, human_max_numbers=6, racket_area_sort=True)
    # 初始化球员跟踪器
    player_tracker = tennis.SortTracker(max_age=5, min_hits=3, resolution=(w, h))
    # 初始化球员姿态器
    player_poser = tennis.PlayerPoser(pose_engine)
    # 初始化球员动作识别器
    player_action = tennis.PlayerAction(action_engine)
    # 球员id
    primary_id = None
    frame_ind = 0
    temp_shot_count = None
    new_frames = []
    start = time.time()
    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/aidata/mmfuck/test_video/output/output_backend_test_full.mp4', fourcc, fps, (w, h))
    while True:
        # 读取一帧
        ret, frame = video.read()
        frame_ind += 1  # 帧数累计
        print(frame_ind)
        # 成功读取帧
        if ret:
            # 检测球员
            human_bboxes, racket_bboxes = player_detector.detect(frame)
            # if human_bboxes is not None:
            #     for bbox in human_bboxes:
            #         x1, y1, x2, y2, score = bbox
            #         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            #         cv2.putText(frame, str(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     print('没有目标')
            # sort跟踪器更新      
            trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes) 
            
            if trackers is not None and primary_id is not None:
                player_bbox = trackers[trackers[:,4] == primary_id].squeeze()
                kpts = player_poser.detect(frame, player_bbox)
                if kpts is not None:
                    for kpt in kpts:
                        x, y = kpt
                        frame = cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                    actioncounter_with_id, action_timestamps_with_id = player_action.detect(kpts, int(primary_id), frame_ind)
                    for id in actioncounter_with_id:
                        count_text = f'player_{id}:' + " | ".join([f"{ACTION_TYPE[action]}: {count}" for action, count in actioncounter_with_id[id].items() if action != 0])
                        # 首先计算文本框大小
                        (text_width, text_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                        # 在文本下方绘制填充矩形作为背景,每个id的文本框高度为40，宽度为文本宽度+20,竖直间距为10 
                        cv2.rectangle(frame, (10, 30 - text_height - 10 + 50), (10 + text_width + 20, 40 + 50), (186,196,206), -1)
                        cv2.putText(frame, count_text, (10, 30 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1,31,32), 2)
                trackers = trackers[trackers[:,4] == primary_id]
            # 每隔300帧，检查primary_id对应的action_counter是否有更新：
            if frame_ind % 300 == 1 and primary_id is not None and primary_id in actioncounter_with_id:
                if temp_shot_count == actioncounter_with_id[primary_id]:
                    primary_id = None
                else:
                    temp_shot_count = actioncounter_with_id[primary_id].copy()
            
            # 添加动作统计文本以及对应id

                
            # cv2.imwrite('frame.jpg', frame)
            # new_frames.append(frame)
        else:  # 视频结尾跳出循环
            break
    print(f"FPS: {frame_ind / (time.time() - start)}")
    print(action_timestamps_with_id)
    # 保存动作时间戳
    with open('action_timestamps_full.json', 'w') as f:
        json.dump(action_timestamps_with_id, f)
        
    with open('action_counter_full.json', 'w') as f:
        json.dump(actioncounter_with_id, f)
    # 释放打开的视频
    video.release()


    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()
if __name__ == '__main__':
    # 测试检测球场线
    # test_detect_court(r"./static/video/video_input1.mp4")

    # test_detect_player('/aidata/tronevan/Dataset/clips_new/bh-vl-0/IMG_0564(1)_clip_00000_offset-0.2.mp4', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine')

    # test_track_player('/aidata/mmfuck/short3.mp4', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine')

    # test_pose_player('/aidata/mmfuck/short3.mp4', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmpose/td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192_fp16/end2end.engine')
    test_action_player('/aidata/mmfuck/short3.mp4', 
                       '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine', 
                       '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmpose/td-hm_ViTPose-small-simple_8xb64-210e_coco-256x192_fp16/end2end.engine',
                       '/aidata/mmfuck/action_classify.engine')
