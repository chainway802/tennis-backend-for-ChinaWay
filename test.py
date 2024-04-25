# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 20:05
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
import util
from tennis.player_detector import PlayerDetector
from tennis.player_tracker import SortTracker
from tennis.player_poser import PlayerPoser
from tennis.play_action import PlayerAction
from tennis.auto_editor import AutoEditor
from tennis.lib.utils.process import VISUALIZATION_CFG
import time
import datetime
import numpy as np
import json
import uuid

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from dataclasses import asdict

from entity.VideoAnalysisEntity import VideoAnalysisEntity
from entity.VideoAnalyzeItemEntity import VideoAnalyzeItemEntity
from oss import OSSHelper

ACTION_TYPE = {0: 'idle', 1: 'forehand', 2: 'backhand', 3: 'serve'}

def draw_keypoints_with_trails(frame, all_keypoints, trails_dict, point_color, max_trails=10, fade_step=25):
    palette = VISUALIZATION_CFG['coco']['palette']

    for person_id, keypoints in all_keypoints.items():
        if person_id not in trails_dict:
            trails_dict[person_id] = []
        trails = trails_dict[person_id]
        
        # 将当前关键点添加到轨迹列表中
        trails.append(keypoints)
        if len(trails) > max_trails:
            trails.pop(0)
        
        # 绘制关键点尾影
        for i, trail in enumerate(trails):
            alpha = max(255 - fade_step * (len(trails) - i), 0)
            for point, color in zip(trail, point_color):
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (palette[color][0], palette[color][1], palette[color][2], alpha), -1, cv2.LINE_AA)
    
    return frame

def test_detect_court(video_path):
    pass
    # # 加载视频
    # video = cv2.VideoCapture(video_path)
    # # 获取视频属性
    # fps, total_frame_length, w, h = util.get_video_properties(video)
    # # 初始化球场检测器
    # court_detector = tennis.CourtDetector(max_age=40)

    # # 初始化一些数据
    # frame_ind = 0
    # new_frames = []
    # # 遍历所有视频帧
    # while True:
    #     # 读取一帧
    #     ret, frame = video.read()
    #     frame_ind += 1  # 帧数累计

    #     # 成功读取帧
    #     if ret:
    #         # 检测第一帧的场地线
    #         if frame_ind == 1:
    #             lines = court_detector.detect_court(frame)
    #         else:  # 其他帧跟踪场地线
    #             lines = court_detector.detect_court(frame)
    #         # 在当前帧画出场地线
    #         for i in range(0, len(lines), 4):
    #             x1, y1, x2, y2 = lines[i:i + 4]
    #             new_frame = cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    #         # 缩放图像尺寸
    #         new_frame = cv2.resize(new_frame, (w, h))
    #         # 将处理后的一帧添加到列表
    #         new_frames.append(new_frame)
    #     else:  # 视频结尾跳出循环
    #         break
    # # 释放打开的视频
    # video.release()

    # # 初始化输出视频
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter(video_path.replace("input", "output"), fourcc, fps, (w, h))
    # # 遍历写入视频
    # for frame in new_frames:
    #     output_video.write(frame)
    # # 释放输出的视频
    # output_video.release()

def test_detect_player(video_path, engine_path):
    pass
    # # 加载视频
    # video = cv2.VideoCapture(video_path)
    # # 获取视频属性
    # fps, total_frame_length, w, h = util.get_video_properties(video)
    # # 初始化球员检测器
    # player_detector = tennis.PlayerDetector(engine_path, human_thr=0.4, racket_thr=0.1)
    # # 初始化一些数据
    # frame_ind = 0
    # new_frames = []
    # # 遍历所有视频帧
    # while True:
    #     # 读取一帧
    #     ret, frame = video.read()
    #     frame_ind += 1  # 帧数累计
    #     # 成功读取帧
    #     if ret:
    #         # 检测球员
    #         human_bboxes, racket_bboxes = player_detector.detect(frame)
    #         # 在当前帧画出球员框
    #         if human_bboxes is not None:
    #             for bbox in human_bboxes:
    #                 x1, y1, x2, y2, _ = bbox
    #                 frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    #         # 在当前帧画出球拍框
    #         if racket_bboxes is not None:
    #             for bbox in racket_bboxes:
    #                 x1, y1, x2, y2, _ = bbox
    #                 frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)

    #         new_frames.append(frame)
    #     else:  # 视频结尾跳出循环
    #         break
    # # 初始化输出视频
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter('/aidata/mmfuck/output_backend_test.mp4', fourcc, fps, (w, h))
    # # 遍历写入视频
    # for frame in new_frames:
    #     output_video.write(frame)

    # # 释放打开的视频
    # video.release()
    # # 释放输出的视频
    # output_video.release()

def test_track_player(video_path, engine_path):
    pass
    # # 加载视频
    # video = cv2.VideoCapture(video_path)
    # # 获取视频属性
    # fps, total_frame_length, w, h = util.get_video_properties(video)
    # # 初始化球员检测器
    # player_detector = tennis.PlayerDetector(engine_path, human_thr=0.4, racket_thr=0.1)
    # # 初始化球员跟踪器
    # player_tracker = tennis.SortTracker(max_age=30, min_hits=3, resolution=(w, h))
    # # 球员id
    # primary_id = None
    # frame_ind = 0
    # new_frames = []
    # start = time.time()
    # while True:
    #     # 读取一帧
    #     ret, frame = video.read()
    #     print(frame.shape)
    #     frame_ind += 1  # 帧数累计
    #     # 成功读取帧
    #     if ret:
    #         # 检测球员
    #         human_bboxes, racket_bboxes = player_detector.detect(frame)
    #         # sort跟踪器更新      
    #         trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)
    #         # 在当前帧画出球员框
    #         if trackers is not None and primary_id is not None:
    #             player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
    #             x1, y1, x2, y2, _ = player_bbox
    #             frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
    #             frame = cv2.putText(frame, f"Player ID: {primary_id}", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #         new_frames.append(frame)
    #     else:  # 视频结尾跳出循环
    #         break
    # print(f"FPS: {frame_ind / (time.time() - start)}")
    # # 释放打开的视频
    # video.release()

    # # 初始化输出视频
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # output_video = cv2.VideoWriter('/aidata/mmfuck/output_backend_test.mp4', fourcc, fps, (w, h))
    # # 遍历写入视频
    # for frame in new_frames:
    #     output_video.write(frame)
    # # 释放输出的视频
    # output_video.release()

def test_pose_player(video_path, det_engine, pose_engine, model_type='YOLO'):
    # 载入视频，MoviePy 自动处理音频和视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = os.path.join('/aidata/mmfuck/test_video/output',f'propaganda.mp4')
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    skeleton = VISUALIZATION_CFG['coco']['skeleton']
    palette = VISUALIZATION_CFG['coco']['palette']
    link_color = VISUALIZATION_CFG['coco']['link_color']
    point_color = VISUALIZATION_CFG['coco']['point_color']
    player_detector = PlayerDetector(human_thr=0.5, human_max_numbers=3, racket_thr=0.3, human_area_sort=False)
    player_tracker = SortTracker(max_age=20, min_hits=3, resolution=(w, h))
    player_poser = PlayerPoser()

    primary_id = None
    frame_ind = 0
    start = time.time()
    
    all_keypoints = {}
    trails_dict = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_ind += 1
        print(frame_ind)
        human_bboxes, racket_bboxes, ball_bboxes = player_detector.detect(frame)
        trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)

        for tracker in reversed(trackers):
            player_bbox = tracker.squeeze()
            id = int(player_bbox[4])
            keypoints = player_poser.detect(frame, player_bbox)
            if keypoints is not None:
                for (u, v), color in zip(skeleton, link_color):
                    if np.isnan(keypoints[u, 0]) or np.isnan(keypoints[u, 1]) or np.isnan(keypoints[v, 0]) or np.isnan(keypoints[v, 1]):
                        continue
                    cv2.line(frame, (int(keypoints[u, 0]), int(keypoints[u, 1])), (int(keypoints[v, 0]), int(keypoints[v, 1])), palette[color], 2, cv2.LINE_AA)
                # for kpt in keypoints:
                #     x, y = kpt
                #     frame = cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)  # 在图像上绘制关键点
                all_keypoints[id] = keypoints
        # 移除掉不在跟踪器中的关键点
        all_keypoints = {id: kpts for id, kpts in all_keypoints.items() if id in trackers[:, 4]}
        frame = draw_keypoints_with_trails(frame, all_keypoints, trails_dict, point_color)
        if frame_ind % 100 == 0:
            cv2.imwrite('frame.jpg', frame)
        out.write(frame)

def test_action_player(video_path):
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_out = os.path.join('/aidata/mmfuck/test_video/output',f'propaganda.mp4')
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    skeleton = VISUALIZATION_CFG['coco']['skeleton']
    palette = VISUALIZATION_CFG['coco']['palette']
    link_color = VISUALIZATION_CFG['coco']['link_color']
    point_color = VISUALIZATION_CFG['coco']['point_color']
    player_detector = PlayerDetector(human_thr=0.5, human_max_numbers=3, racket_thr=0.3, human_area_sort=False)
    player_tracker = SortTracker(max_age=20, min_hits=3, resolution=(w, h))
    player_poser = PlayerPoser()
    player_action = PlayerAction()
    primary_id = None
    frame_ind = 0
    start = time.time()
    temp_shot_count = None
    all_keypoints = {}
    trails_dict = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_ind += 1
        print(frame_ind)
        human_bboxes, racket_bboxes, ball_bboxes = player_detector.detect(frame)
        trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)

        if trackers is not None and primary_id is not None:
            player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
            id = int(player_bbox[4])
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
                    cv2.rectangle(frame, (900, 300), (900 + text_width + 20, 300 + text_height + 20), (186, 196, 206), -1)
                    cv2.putText(frame, count_text, (900, 300 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 31, 32), 2)
            trackers = trackers[trackers[:, 4] == primary_id]
            # 每隔300帧，检查primary_id对应的action_counter是否有更新：
            if frame_ind % 300 == 1 and primary_id is not None and primary_id in actioncounter_with_id:
                if temp_shot_count == actioncounter_with_id[primary_id]:
                    primary_id = None
                else:
                    temp_shot_count = actioncounter_with_id[primary_id].copy()

        out.write(frame)    
            
            # cv2.imwrite('frame.jpg', frame)
            # new_frames.append(frame)

    print(f"FPS: {frame_ind / (time.time() - start)}")
    # print(action_timestamps_with_id)
    # # 保存动作时间戳
    # with open('action_timestamps_match.json', 'w') as f:
    #     json.dump(action_timestamps_with_id, f)
        
    # with open('action_counter_match.json', 'w') as f:
    #     json.dump(actioncounter_with_id, f)
    # 释放打开的视频
    cap.release()
    out.release()


def auto_edit(video_path):
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_out = os.path.join('/aidata/mmfuck/test_video/output',f'propaganda.mp4')
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    skeleton = VISUALIZATION_CFG['coco']['skeleton']
    palette = VISUALIZATION_CFG['coco']['palette']
    link_color = VISUALIZATION_CFG['coco']['link_color']
    point_color = VISUALIZATION_CFG['coco']['point_color']
    player_detector = PlayerDetector(human_thr=0.5, human_max_numbers=3, racket_thr=0.3, human_area_sort=False)
    player_tracker = SortTracker(max_age=20, min_hits=3, resolution=(w, h))
    player_poser = PlayerPoser()
    player_action = PlayerAction()
    editor = AutoEditor(video_duration_frames)
    primary_id = None
    frame_ind = 0
    start = time.time()
    temp_shot_count = None
    all_keypoints = {}
    trails_dict = {}
    while True:
        t1 = time.time()
        ret, frame = cap.read()
        print('read time:', time.time()-t1)
        if not ret:
            break
        frame_ind += 1
        print(frame_ind)
        human_bboxes, racket_bboxes, ball_bboxes = player_detector.detect(frame)
        trackers, matched_dets, primary_id = player_tracker.update(human_bboxes, racket_bboxes)

        if trackers is not None and primary_id is not None:
            player_bbox = trackers[trackers[:, 4] == primary_id].squeeze()
            id = int(player_bbox[4])
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
                    cv2.rectangle(frame, (900, 300), (900 + text_width + 20, 300 + text_height + 20), (186, 196, 206), -1)
                    cv2.putText(frame, count_text, (900, 300 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 31, 32), 2)
            trackers = trackers[trackers[:, 4] == primary_id]
            # 每隔300帧，检查primary_id对应的action_counter是否有更新：
            if frame_ind % 300 == 1 and primary_id is not None and primary_id in actioncounter_with_id:
                if temp_shot_count == actioncounter_with_id[primary_id]:
                    primary_id = None
                else:
                    temp_shot_count = actioncounter_with_id[primary_id].copy()
    # 获得第一个目标的动作时间戳
    action_timestamps = action_timestamps_with_id[list(action_timestamps_with_id.keys())[0]]
    # 获得高质量片段的区间
    rally_intervals = editor.get_rallys(action_timestamps)

    output_path = f'/aidata/mmfuck/test_video/output/clips'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    current_segment_index = 0
    current_frame_index = 1

    # output = cv2.VideoWriter(os.path.join(output_path, 'clips_0.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # while videof.isOpened() and current_segment_index < len(rally_intervals):
    #     ret, frame = videof.read()
    #     print(current_frame_index)
    #     if not ret:
    #         break

    #     start_frame, end_frame = rally_intervals[current_segment_index]
    #     # 如果当前帧在当前片段的范围内，则写入输出文件
    #     if start_frame <= current_frame_index <= end_frame:
    #         # frame = cv2.resize(frame, )
    #         output.write(frame)

    #     # 如果当前帧达到了当前片段的结束帧，则移动到下一个片段
    #     if current_frame_index == end_frame:
    #         current_segment_index += 1
    #         output.release()
    #         if current_segment_index < len(rally_intervals):
    #             output = cv2.VideoWriter(os.path.join(output_path, f'clips_{current_segment_index}.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), fps, (w, h))

    #     current_frame_index += 1

    # 释放资源
    # videof.release()
    # output.release()
    cv2.destroyAllWindows()


def test_detect_tennis_ball(image_path):
    # 读取图像
    frame = cv2.imread(image_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 初始化网球检测器
    tennis_ball_detector = tennis.TennisBallDetector(cache_len=8)

    # 检测网球位置
    position = tennis_ball_detector.detect_tennis_ball(frame)

    PIL_image = Image.fromarray(frame)
    draw_x = position[1]
    draw_y = position[0]
    bbox = (draw_x - 5, draw_y - 5, draw_x + 5, draw_y + 5)
    draw = ImageDraw.Draw(PIL_image)
    draw.ellipse(bbox, outline='red', fill="blue")
    del draw

    PIL_image.show()


def test_dataclass():
    video_analysis = VideoAnalysisEntity(
        id=1,
        userId="123",
        videoName="test_video_name",
        videoUrl="http://aliyun",
        funcValue="0",
        analyzeTime=datetime.datetime.now(),
        status=200,
        pinNum=10000,
        scoreRet="test_score_ret",
        analyzeItems=[VideoAnalyzeItemEntity(id=1), VideoAnalyzeItemEntity(id=2)]
    )
    video_analysis_dict = asdict(video_analysis)
    print(video_analysis_dict, '\n')

    serialized_data = json.dumps(video_analysis_dict, default=VideoAnalysisEntity.serialize_complex_types).encode('utf-8')
    print(serialized_data, '\n')

    deserialized_data = json.loads(serialized_data.decode('utf-8'))
    print(deserialized_data, '\n')

    new_video_analysis = VideoAnalysisEntity(**deserialized_data)
    print(type(new_video_analysis.analyzeItems[0]))
    new_video_analysis_dict = asdict(new_video_analysis)
    print(new_video_analysis_dict, '\n')


def upload_video_to_oss(video_path):
    # 先解析配置文件
    conf = util.load_yaml_config(r"./config/config.yaml")
    # 初始化oss服务
    oss = OSSHelper(**conf["oss"])
    # 计算视频名称的uuid
    video_uuid = uuid.uuid4()
    # 上传视频
    try:
        processUrl = oss.upload_file(video_uuid, video_path)
        print(processUrl)
    except Exception as e:
        print("upload error: ", e)


def download_video_to_local(object_file_url):
    # 先解析配置文件
    conf = util.load_yaml_config(r"./config/config.yaml")
    # 初始化oss服务
    oss = OSSHelper(**conf["oss"])
    # 下载视频
    try:
        local_temp_video_path = oss.download_file(object_file_url)
        print(local_temp_video_path)
    except Exception as e:
        print("upload error: ", e)


if __name__ == '__main__':
    # 测试检测球场线
    # test_detect_court(r"./static/video/video_input1.mp4")

    # test_detect_player('/aidata/tronevan/Dataset/clips_new/bh-vl-0/IMG_0564(1)_clip_00000_offset-0.2.mp4', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine')

    # test_track_player('/aidata/mmfuck/short3.mp4', '/aidata/mmfuck/mmdeploy/mmdeploy_models/mmdet/rtmdet_tiny_8xb32-300e_coco_fp16/end2end.engine')

    # test_pose_player('/aidata/mmfuck/prop.mp4', '/aidata/mmfuck/yolov8_trt_static/yolov8s_FP16.trt', '/aidata/mmfuck/ViTPose-Pytorch/models/vitpose_small.engine')
    # test_action_player('/aidata/mmfuck/test_video/input/707b806b3f9548dca14f478cc29d8798/test-001.mp4')

    # test_detect_tennis_ball(r"./static/image/frame_20.jpg")

    # auto_edit('/aidata/mmfuck/test_video/input/707b806b3f9548dca14f478cc29d8798/test-001.mp4')
    
    # 测试数据实体类的序列化和反序列化
    # test_dataclass()

    # 上传本地视频到oss
    # upload_video_to_oss(r"/aidata/mmfuck/test_video/input/707b806b3f9548dca14f478cc29d8798/Raw_Video/video40.mp4")

    # 下载oss视频到本地
    # download_video_to_local(r"https://astree.oss-cn-shanghai.aliyuncs.com/1_1713701511071.mp4")
