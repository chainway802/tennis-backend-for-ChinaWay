# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/01 20:16
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import cv2
import yaml
import numpy as np





def get_video_properties(video):
    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    # get videos properties
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
        length = int(video.get(cv2.cv.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.cv.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.cv.CAP_PROP_FRAME_HEIGHT))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        v_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, length, v_width, v_height


def load_yaml_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return conf