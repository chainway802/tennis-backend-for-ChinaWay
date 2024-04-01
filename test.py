# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 20:05
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2

import utils

import tennis


def test_detect_court(video_path):
    # 加载视频
    video = cv2.VideoCapture(video_path)
    # 获取视频属性
    fps, total_frame_length, w, h = utils.get_video_properties(video)

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
                lines, pmatrix = tennis.detect_court(frame)
            else:  # 其他帧跟踪场地线
                lines, pmatrix = tennis.detect_court(frame, pmatrix=pmatrix)
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
    output_video = cv2.VideoWriter(r"./static/video/Output_Court.mp4", fourcc, fps, (w, h))
    # 遍历写入视频
    for frame in new_frames:
        output_video.write(frame)
    # 释放输出的视频
    output_video.release()


if __name__ == '__main__':
    test_detect_court(r"./static/video/video_input3.mp4")
