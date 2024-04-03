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
from .ProcessConfigs import IMAGE_MEAN, IMAGE_STD

class PlayerDetectionModel(TRTModel):

    def __init__(self):
        print("初始化球员检测模型")

    def _pre_process(self, data, channel_convert=False, resized_shape=(320, 320)):
        img = cv2.resize(img, resized_shape).astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        image_channels = img.shape[0]
        if channel_convert:
            # 将channel维度的顺序调换为(2,1,0)
            img = img[::-1, :, :] #::-1表示逆序，即将第一个维度的顺序调换为(2,1,0)
            for c in range(image_channels):
                # 逆序获取均值和标准差
                img[c] = (img[c] - IMAGE_MEAN[-c]) / IMAGE_STD[-c]
        else:
            for c in range(image_channels):
                img[c] = (img[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
        img = np.expand_dims(img, axis=0)
        return img       
        print("球场检测模型中的预处理")

    def _post_process(self, det_results, img_shape, max_numbers_by_area, max_numbers, resized_shape=(320, 320), thr=0.4, label=0, area_sort=False):
        bboxes = det_results['dets'].squeeze()
        labels = np.expand_dims(det_results['labels'].squeeze(), axis=-1)
        bboxes_with_labels = np.concatenate((bboxes, labels), axis=-1)
        # bboxes_with_labels的维度为(N, 6)，其中N为检测到的目标个数，5代表(x1, y1, x2, y2, score, label)
        bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 5] == label]
        bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 4] > thr]
        # print('numbers of detected people:', len(bboxes_with_labels))
        if len(bboxes_with_labels) == 0:
            return None
        # 计算检测框的面积
        if area_sort:
            bboxes_area = (bboxes_with_labels[:, 2] - bboxes_with_labels[:, 0]) * (bboxes_with_labels[:, 3] - bboxes_with_labels[:, 1])
            # 按照面积从大到小排序, 取前max_num_people_by_area个
            bboxes_with_labels = bboxes_with_labels[bboxes_area.argsort()[::-1]][:max_numbers_by_area]
        bboxes_nms = nms(bboxes_with_labels[:,:5], iou_threshold=0.2, max_numbers=max_numbers)
        
        for i in range(len(bboxes_nms)):
            x1, y1, x2, y2, score= bboxes_nms[i]
            # 将坐标映射回原图
            x1, y1, x2, y2 = x1 * img_shape[1]/resized_shape[1], y1 * img_shape[0]/resized_shape[0], x2 * img_shape[1]/resized_shape[1], y2 * img_shape[0]/resized_shape[0]
            # 将坐标限制在图片范围内
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_shape[1], x2), min(img_shape[0], y2)
            bboxes_nms[i] = [x1, y1, x2, y2, score]
        return bboxes_nms

