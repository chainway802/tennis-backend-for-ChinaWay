# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:27
@Version  :   1.0
@License  :   (C)Copyright 2024
"""

from .TRTModel import TRTModel
import cv2
import numpy as np
from ..utils.process import IMAGE_MEAN, IMAGE_STD, get_heatmap_maximum, refine_keypoints_dark_udp

class PlayerPoseEstimationModel(TRTModel):

    def __init__(self):
        super().__init__(max_batch_size=1)
        print("初始化球员姿态检测模型")

    def _pre_process(self, frame, bbox, channel_convert=False, resized_shape=(320, 320), resolution=(1920, 1080)):
        
        x1, y1, x2, y2 = bbox
        # 裁剪图片并进行预处理，推理
        img_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        cv2.imwrite("img_crop.jpg", img_crop)
        img_crop = cv2.resize(img_crop, resized_shape).astype(np.float32) #resize后的shape是resized_shape的倒序
        img_crop = np.transpose(img_crop, (2, 0, 1))
        image_channels = img_crop.shape[0]
        if channel_convert:
            # 将channel维度的顺序调换为(2,1,0)
            img_crop = img_crop[::-1, :, :] #::-1表示逆序，即将第一个维度的顺序调换为(2,1,0)
            for c in range(image_channels):
                # 逆序获取均值和标准差
                img_crop[c] = (img_crop[c] - IMAGE_MEAN[-c]) / IMAGE_STD[-c]
        else:
            for c in range(image_channels):
                img_crop[c] = (img_crop[c] - IMAGE_MEAN[c]) / IMAGE_STD[c]
        img_pre = np.expand_dims(img_crop, axis=0)
        return img_pre       
    
    def inference(self, frame, bbox, channel_convert=False):
        x1, y1, x2, y2, _ = bbox
        x1, y1, x2, y2 = max(0, x1-5), max(0, y1-30), min(frame.shape[1], x2+5), min(frame.shape[0], y2+5)      
        data_pre = self._pre_process(frame, [x1,y1,x2,y2], channel_convert=channel_convert, resized_shape=self.inputshape[::-1], resolution=frame.shape[:2][::-1])
        pose_results = super().inference(data_pre)
        img_crop_shape = (y2-y1, x2-x1)
        kpts_post = self._post_process(pose_results, self.inputshape, img_crop_shape, (x1, y1))
        return kpts_post
    
    def _post_process(self, pose_results, resized_shape, img_crop_shape, bbox_xy1):
        heatmaps = pose_results['output'].squeeze()
        heatmaps_shape = (heatmaps.shape[1], heatmaps.shape[2])
        keypoints, scores = get_heatmap_maximum(heatmaps)
        # unsqueeze the instance dimension for single-instance results
        keypoints = keypoints[None]
        scores = scores.squeeze()
        keypoints = refine_keypoints_dark_udp(
            keypoints, heatmaps, blur_kernel_size=11).squeeze()

        x, y = np.array(keypoints[:, 0]), np.array(keypoints[:, 1])
        #img_shape为y,x resized_shape为x,y
        # 列表每个元素的值乘以图片的宽高比例，再加上bbox的左上角坐标
        if resized_shape[0]/heatmaps_shape[0] == resized_shape[1]/heatmaps_shape[1]:
            scale = resized_shape[0]/heatmaps_shape[0]
            x = x * scale * img_crop_shape[1]/resized_shape[1] + bbox_xy1[0]
            y = y * scale * img_crop_shape[0]/resized_shape[0] + bbox_xy1[1]
            kpts_post = np.stack((x, y), axis=-1)

            print("球员姿态检测模型后处理")
            return kpts_post
