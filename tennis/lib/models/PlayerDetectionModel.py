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
from ..utils.process import IMAGE_MEAN, IMAGE_STD, nms
class PlayerDetectionModel(TRTModel):

    def __init__(self):
        super().__init__(max_batch_size=1)
        print("初始化球员检测模型")

    def _pre_process(self, frame, channel_convert=False, resized_shape=(320, 320)):
        frame = cv2.resize(frame, resized_shape).astype(np.float32)
        frame = np.transpose(frame, (2, 0, 1))
        image_channels = frame.shape[0]
        if channel_convert:
            # 将channel维度的顺序调换为(2,1,0)
            frame = frame[::-1, :, :] #::-1表示逆序，即将第一个维度的顺序调换为(2,1,0)
            for c in range(image_channels):
                # 逆序获取均值和标准差
                frame[c] = (frame[c] - IMAGE_MEAN[-c]) / IMAGE_STD[-c]
        else:
            for c in range(image_channels):
                frame[c] = (frame[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
        frame_pre = np.expand_dims(frame, axis=0)
        return frame_pre       
        print("球场检测模型中的预处理")

    def inference(self, frame, channel_convert=False, human_max_numbers_by_area=3, racket_max_numbers_by_area=2, 
                  human_max_numbers=2, racket_max_numbers=1, human_thr=0.4, racket_thr=0.3, 
                  human_area_sort=True, racket_area_sort=True):
        
        data_pre = self._pre_process(frame, channel_convert=channel_convert, resized_shape=self.inputshape)
        det_results = super().inference(data_pre)
        human_bboxes_post = self._post_process(det_results, frame.shape, max_numbers_by_area=human_max_numbers_by_area, 
                                               max_numbers=human_max_numbers, resized_shape=self.inputshape, 
                                               thr=human_thr, label=0, area_sort=human_area_sort)
        racket_bboxes_post = self._post_process(det_results, frame.shape, max_numbers_by_area=racket_max_numbers_by_area, 
                                                max_numbers=racket_max_numbers, resized_shape=self.inputshape, 
                                                thr=racket_thr, label=38, area_sort=racket_area_sort)
        return human_bboxes_post, racket_bboxes_post
    
    def _post_process(self, det_results, img_shape, max_numbers_by_area, max_numbers, resized_shape=(320, 320), thr=0.4, label=0, area_sort=False):
        bboxes = det_results['dets'].squeeze()
        labels = np.expand_dims(det_results['labels'].squeeze(), axis=-1)
        bboxes_with_labels = np.concatenate((bboxes, labels), axis=-1)
        # bboxes_with_labels的维度为(N, 6)，其中N为检测到的目标个数，5代表(x1, y1, x2, y2, score, label)
        bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 5] == label]
        bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 4] > thr]
        if len(bboxes_with_labels) == 0:
            return None
        # 计算检测框的面积
        if area_sort:
            bboxes_area = (bboxes_with_labels[:, 2] - bboxes_with_labels[:, 0]) * (bboxes_with_labels[:, 3] - bboxes_with_labels[:, 1])
            # 按照面积从大到小排序, 取前max_num_people_by_area个
            bboxes_with_labels = bboxes_with_labels[bboxes_area.argsort()[::-1]][:max_numbers_by_area]
        bboxes_nms = nms(bboxes_with_labels[:,:5], iou_threshold=0.2, max_numbers=max_numbers)
        bboxes_post = np.zeros_like(bboxes_nms)
        for i in range(len(bboxes_nms)):
            x1, y1, x2, y2, score= bboxes_nms[i]
            # 将坐标映射回原图
            x1, y1, x2, y2 = x1 * img_shape[1]/resized_shape[1], y1 * img_shape[0]/resized_shape[0], x2 * img_shape[1]/resized_shape[1], y2 * img_shape[0]/resized_shape[0]
            # 将坐标限制在图片范围内
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_shape[1], x2), min(img_shape[0], y2)
            bboxes_post[i] = [x1, y1, x2, y2, score]
        return bboxes_post

