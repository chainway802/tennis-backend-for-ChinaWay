# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/03 18:27
@Version  :   1.0
@License  :   (C)Copyright 2024
"""


import cv2
import numpy as np
from .TRTModel import TRTModel
from ..utils.process import IMAGE_MEAN, IMAGE_STD, get_heatmap_maximum, refine_keypoints_dark_udp

class PlayerPoseEstimationModel(TRTModel):

    def __init__(self, 
                 engine_file_path=None,
                 onnx_file_path=None,
                 use_onnx=True,
                 precision_flop="FP16",
                 img_size=(256, 192),
                 heatmap_size=(17, 64, 48),
                 dynamic_shapes={},
                 dynamic_max_batch_size=1,
                 channel_convert=True):
        """
        初始化姿态检测模型

        :param engine_file_path: trt模型权重路径
        :param onnx_file_path: onnx模型权重路径
        :param use_onnx: 是否使用onnx模型
        :param precision_flop: 使用的精度类型，可选["FP32", "FP16", "int8"]
        :param img_size: 输入模型的图像大小
        :param heatmap_size: 输出的热图大小
        :param dynamic_shapes: 自定义动态维度
        :param dynamic_max_batch_size: 动态batch size的最大值
        """
        super(PlayerPoseEstimationModel, self).__init__(engine_file_path, onnx_file_path, 
                                                   use_onnx, precision_flop, img_size,
                                                   dynamic_shapes, dynamic_max_batch_size) 
        self.heatmap_size = heatmap_size

    def _pre_process(self, frame, bbox, channel_convert=False, resized_shape=(320, 320), resolution=(1920, 1080)):
        
        x1, y1, x2, y2 = bbox
        # 裁剪图片并进行预处理，推理
        img_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
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
        return np.ascontiguousarray(img_pre, dtype=np.float32)     
    
    def inference(self, frame, bbox, channel_convert=False):
        
        self.inputshape = self.engine.get_binding_shape(0)[-2:]

        x1, y1, x2, y2, _ = bbox
        # 检查bbox合理性
        x1, y1, x2, y2 = np.clip(x1-5, 0, frame.shape[1]), np.clip(y1-30, 0, frame.shape[0]), np.clip(x2+5, 0, frame.shape[1]), np.clip(y2+5, 0, frame.shape[0])
        if x1 >= x2 or y1 >= y2:
            return None
        
        infer_input = self._pre_process(frame, [x1,y1,x2,y2], channel_convert=channel_convert, resized_shape=self.inputshape[::-1], resolution=frame.shape[:2][::-1])
        
        if self.use_onnx:
            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            infer_output = self.ort_session.run([output_name], {input_name: infer_input})
        else:
            infer_output = self.base_inference(infer_input)
            
        img_crop_shape = (y2-y1, x2-x1)
        kpts_post = self._post_process(infer_output, self.inputshape, img_crop_shape, (x1, y1))
        return kpts_post
    
    def _post_process(self, infer_output, resized_shape, img_crop_shape, bbox_xy1):
        infer_output = infer_output[0]
        heatmaps = infer_output.reshape(self.heatmap_size)
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
            return kpts_post
