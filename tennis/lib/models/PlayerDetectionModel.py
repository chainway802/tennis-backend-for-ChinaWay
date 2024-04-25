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

    def __init__(self, 
                 engine_file_path=None,
                 onnx_file_path=None,
                 use_onnx=False,
                 precision_flop="FP16",
                 img_size=(640, 640),
                 dynamic_shapes={},
                 dynamic_max_batch_size=1,
                 model_type='YOLO'):
        
        """
        初始化球员检测模型

        :param engine_file_path: trt模型权重路径
        :param onnx_file_path: onnx模型权重路径
        :param use_onnx: 是否使用onnx模型
        :param precision_flop: 使用的精度类型，可选["FP32", "FP16", "int8"]
        :param img_size: 输入模型的图像大小
        :param dynamic_shapes: 自定义动态维度
        :param dynamic_max_batch_size: 动态batch size的最大值
        :param model_type: 算法类型，可选["YOLO", "RTMDet"]
        """
        # self.cfx = cuda.Device(0).make_context()
        super(PlayerDetectionModel, self).__init__(engine_file_path, onnx_file_path, 
                                                   use_onnx, precision_flop, img_size,
                                                   dynamic_shapes, dynamic_max_batch_size)
        self.model_type = model_type

    def _pre_process(self, frame, channel_convert=False, resized_shape=(320, 320)):
        if self.model_type == 'YOLO':
            oh, ow, _ = frame.shape
            h, w = resized_shape
            # 初始化固定输入大小的张量
            input_image = np.ones((h, w, 3)) * 128
            # 计算宽高比不变的缩放因子
            ratio = min(h / oh, w / ow)
            # 缩放图像
            nh, nw = int(oh * ratio), int(ow * ratio)
            resized_img = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            # 填充图像
            input_image[(h - nh) // 2: (h - nh) // 2 + nh, (w - nw) // 2: (w - nw) // 2 + nw, :] = resized_img
            # 变换维度和数值范围
            input_image = input_image.transpose((2, 0, 1))
            input_image = input_image / 255
            frame_pre = np.expand_dims(input_image, axis=0)
            frame_pre = np.ascontiguousarray(frame_pre, dtype=np.float32)
        else:
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
        return np.ascontiguousarray(frame_pre, dtype=np.float32)      

    def inference(self, frame, channel_convert=False, human_max_numbers_by_area=3, racket_max_numbers_by_area=2, 
                  human_max_numbers=2, racket_max_numbers=1, human_thr=0.4, racket_thr=0.3, 
                  human_area_sort=True, racket_area_sort=True):
         
        self.inputshape = self.engine.get_binding_shape(0)[-2:]
        
        infer_input = self._pre_process(frame, channel_convert=channel_convert, resized_shape=self.inputshape)
        
        if self.use_onnx:
            input_name = self.ort_session.get_inputs()[0].name
            output_name = self.ort_session.get_outputs()[0].name
            infer_output = self.ort_session.run([output_name], {input_name: infer_input})
        else:
            infer_output = self.base_inference(infer_input)
            
        human_bboxes_post = self._post_process(infer_output, frame.shape, max_numbers_by_area=human_max_numbers_by_area, 
                                            max_numbers=human_max_numbers, resized_shape=self.inputshape, 
                                            thr=human_thr, label=0, area_sort=human_area_sort)
        racket_bboxes_post = self._post_process(infer_output, frame.shape, max_numbers_by_area=racket_max_numbers_by_area, 
                                                max_numbers=racket_max_numbers, resized_shape=self.inputshape, 
                                                thr=racket_thr, label=38, area_sort=racket_area_sort)
        ball_bboxes_post = self._post_process(infer_output, frame.shape, max_numbers_by_area=3, 
                                            max_numbers=2, resized_shape=self.inputshape,
                                            thr=0.2, label=32, area_sort=False)
        return human_bboxes_post, racket_bboxes_post, ball_bboxes_post
    
    def _post_process(self, infer_output, img_shape, max_numbers_by_area=3, max_numbers=2, resized_shape=(320, 320), thr=0.4, label=0, area_sort=False):
        if self.model_type == 'YOLO':
            # 变换输出数据格式
            infer_output = infer_output[0]
            det_results = infer_output.reshape((84, 8400))
            # 将(84, 8400)处理成(8400, 85)  85= box:4  conf:1 cls:80
            pred = np.transpose(det_results, (1, 0))  # (8400, 84)
            pred_class = pred[..., 4:]
            labels = np.argmax(pred_class, axis=-1).reshape(-1, 1)
            pred_conf = np.max(pred_class, axis=-1).reshape(-1, 1)
            bboxes_with_labels = np.concatenate([pred[..., :4], pred_conf, labels], axis=-1)
            bboxes_with_labels = bboxes_with_labels[np.argsort(-bboxes_with_labels[:, 4])]
            bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 5] == label]
            bboxes_with_labels = bboxes_with_labels[bboxes_with_labels[:, 4] > thr]
            if len(bboxes_with_labels) == 0:
                return None
            # 进行非极大值抑制nms，(N, 6) 6->[x,y,w,h,conf(最大类别概率),class]
            for i, bbox in enumerate(bboxes_with_labels):
                # xywh -> xyxy
                bbox[0] = bbox[0] - bbox[2] / 2
                bbox[1] = bbox[1] - bbox[3] / 2
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]
                bboxes_with_labels[i] = bbox
                
            bboxes_nms = nms(bboxes_with_labels[:,:5], iou_threshold=0.35, max_numbers=max_numbers)
            # 转换数据最终输出的格式
            bboxes_post = np.zeros_like(bboxes_nms)
            for i in range(len(bboxes_nms)):
                # 获取图像大小
                h, w = resized_shape
                oh, ow, _ = img_shape
                # 转换坐标表示形式
                x1, y1, x2, y2, score = bboxes_nms[i]
                # 计算原图在等比例缩放后的尺寸
                ratio = min(h / oh, w / ow)
                nh, nw = oh * ratio, ow * ratio
                # 计算平移的量
                w_move, h_move = abs(w - nw) // 2, abs(h - nh) // 2
                ret_x1, ret_x2 = (x1 - w_move) / ratio, (x2 - w_move) / ratio
                ret_y1, ret_y2 = (y1 - h_move) / ratio, (y2 - h_move) / ratio
                bboxes_post[i] = [ret_x1, ret_y1, ret_x2, ret_y2, score]
        else:
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
            bboxes_nms = nms(bboxes_with_labels[:,:5], iou_threshold=0.35, max_numbers=max_numbers)
            bboxes_post = np.zeros_like(bboxes_nms)
            for i in range(len(bboxes_nms)):
                x1, y1, x2, y2, score= bboxes_nms[i]
                # 将坐标映射回原图
                x1, y1, x2, y2 = x1 * img_shape[1]/resized_shape[1], y1 * img_shape[0]/resized_shape[0], x2 * img_shape[1]/resized_shape[1], y2 * img_shape[0]/resized_shape[0]
                # 将坐标限制在图片范围内
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_shape[1], x2), min(img_shape[0], y2)
                bboxes_post[i] = [x1, y1, x2, y2, score]
        return bboxes_post

