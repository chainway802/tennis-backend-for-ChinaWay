import numpy as np
import os
import cv2
import ctypes
from scipy.optimize import linear_sum_assignment
IMAGE_MEAN = [103.5300, 116.2800, 123.6750]
IMAGE_STD = [57.375,57.12,58.395]
SIMCC_SPLIT_RATIO = 2
VISUALIZATION_CFG = dict(
    coco=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)],
        palette=[(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
                 (255, 153, 255), (153, 204, 255), (255, 102, 255),
                 (255, 51, 255), (102, 178, 255), (51, 153, 255),
                 (255, 153, 153), (255, 102, 102), (255, 51, 51),
                 (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
                 (0, 0, 255), (255, 0, 0), (255, 255, 255)],
        link_color=[
            0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
        ],
        point_color=[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]),
    coco_wholebody=dict(
        skeleton=[(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                  (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                  (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                  (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                  (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                  (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                  (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                  (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                  (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                  (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                  (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                  (129, 130), (130, 131), (131, 132)],
        palette=[(51, 153, 255), (0, 255, 0), (255, 128, 0), (255, 255, 255),
                 (255, 153, 255), (102, 178, 255), (255, 51, 51)],
        link_color=[
            1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1,
            1, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        point_color=[
            0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2,
            2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1,
            1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
        ],
        sigmas=[
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068,
            0.066, 0.066, 0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043,
            0.040, 0.035, 0.031, 0.025, 0.020, 0.023, 0.029, 0.032, 0.037,
            0.038, 0.043, 0.041, 0.045, 0.013, 0.012, 0.011, 0.011, 0.012,
            0.012, 0.011, 0.011, 0.013, 0.015, 0.009, 0.007, 0.007, 0.007,
            0.012, 0.009, 0.008, 0.016, 0.010, 0.017, 0.011, 0.009, 0.011,
            0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010, 0.034, 0.008,
            0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009, 0.009,
            0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032,
            0.02, 0.019, 0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047,
            0.026, 0.025, 0.024, 0.035, 0.018, 0.024, 0.022, 0.026, 0.017,
            0.021, 0.021, 0.032, 0.02, 0.019, 0.022, 0.031
        ]))



'''加载tensorrt插件库'''
def load_tensorrt_plugin() -> bool:
    """Load TensorRT plugins library.

    Returns:
        bool: True if TensorRT plugin library is successfully loaded.
    """
    lib_path = '/aidata/mmfuck/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so'  # 这个插件文件，不同服务器得重新编译获得
    success = False
    if os.path.exists(lib_path):
        ctypes.CDLL(lib_path)
        print(f'Successfully loaded tensorrt plugins from {lib_path}')
        success = True
    else:
        print(f'Could not load the library of tensorrt plugins. \
            Because the file does not exist: {lib_path}')
    return success


'''目标检测算法数据处理'''
# 图片预处理
def PrePocress(img, channel_convert=False, resized_shape=(320, 320)):
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

# 非极大值抑制
def nms(bboxes_with_labels, iou_threshold=0.2, max_numbers=2):
    if len(bboxes_with_labels) == 0:
        return []

    # bboxes_with_labels = bboxes_with_labels[np.argsort(-bboxes_with_labels[:, 4])]
    selected_bboxes = []

    while bboxes_with_labels.shape[0] > 0 and len(selected_bboxes) < max_numbers: #args.max_num_people:
        # 选择得分最高的bbox
        bbox = bboxes_with_labels[0]
        selected_bboxes.append(bbox)
        if len(bboxes_with_labels) == 1:
            break
        # 计算选中的bbox与其它bbox的IOU
        ious = compute_iou(bbox, bboxes_with_labels[1:])
        # 保留IOU小于阈值的bbox
        bboxes_with_labels = bboxes_with_labels[1:][ious < iou_threshold]

    return np.array(selected_bboxes)

# 计算交并比
def compute_iou(bbox, bboxes):
    x1 = np.maximum(bbox[0], bboxes[:, 0])
    y1 = np.maximum(bbox[1], bboxes[:, 1])
    x2 = np.minimum(bbox[2], bboxes[:, 2])
    y2 = np.minimum(bbox[3], bboxes[:, 3])

    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])

    union_area = bbox_area + bboxes_area - intersection_area

    return intersection_area / (union_area + 1e-6)

# 将bbox坐标映射回原图
def PostProcessBbox(det_results, img_shape, max_numbers_by_area, max_numbers, resized_shape=(320, 320), thr=0.4, label=0, area_sort=False):
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


'''姿态估计算法decode'''
def PostProcessKpts(kpts, heatmaps_shape, img_shape, bbox_xy1, resized_shape=(256, 192)):
    x, y = np.array(kpts[:, 0]), np.array(kpts[:, 1])
    #img_shape为y,x resized_shape为x,y
    # 列表每个元素的值乘以图片的宽高比例，再加上bbox的左上角坐标
    if resized_shape[0]/heatmaps_shape[0] == resized_shape[1]/heatmaps_shape[1]:
        scale = resized_shape[0]/heatmaps_shape[0]
        x = x * scale * img_shape[1]/resized_shape[1] + bbox_xy1[0]
        y = y * scale * img_shape[0]/resized_shape[0] + bbox_xy1[1]
        kpts = np.stack((x, y), axis=-1)
    return kpts

'''UDP'''
def get_heatmap_maximum(heatmaps: np.ndarray):
    """Get maximum response location and value from heatmaps.

    Note:
        batch_size: B
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (B, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (B, K)
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, (
        f'Invalid shape {heatmaps.shape}')

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)
    # unravel_index函数将扁平化的数组索引转换为多维数组索引
    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals

def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """Modulate heatmap distribution with Gaussian.

    Note:
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps

def refine_keypoints_dark_udp(keypoints: np.ndarray, heatmaps: np.ndarray,
                              blur_kernel_size: int) -> np.ndarray:
    """Refine keypoint predictions using distribution aware coordinate decoding
    for UDP. See `UDP`_ for details. The operation is in-place.

    Note:

        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - heatmap size: [W, H]

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size of the heatmap
            modulation

    Returns:
        np.ndarray: Refine keypoint coordinates in shape (N, K, D)

    .. _`UDP`: https://arxiv.org/abs/1911.07524
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian,
                                  derivative).squeeze()

    return keypoints

'''simcc'''
def get_simcc_maximum(simcc_x, simcc_y):
    """Get maximum response location and value from simcc representations.

    rewrite to support `torch.Tensor` input type.

    Args:
        simcc_x : x-axis SimCC in shape (N, K, Wx)
        simcc_y : y-axis SimCC in shape (N, K, Wy)

    Returns:
        tuple:
        - locs : locations of maximum heatmap responses in shape
            (N, K, 2)
        - vals : values of maximum heatmap responses in shape
            (N, K)
    """
    N, K, _ = simcc_x.shape
    simcc_x = simcc_x.reshape(-1, simcc_x.shape[2])
    simcc_y = simcc_y.reshape(-1, simcc_y.shape[2])
    x_locs = np.argmax(simcc_x, axis=1).reshape(-1, 1)
    y_locs = np.argmax(simcc_y, axis=1).reshape(-1, 1)
    locs = np.concatenate((x_locs, y_locs), axis=1).astype(float)
    
    max_val_x = np.max(simcc_x, axis=1, keepdims=True)
    max_val_y = np.max(simcc_y, axis=1, keepdims=True)
    vals = np.min(np.concatenate([max_val_x, max_val_y], axis=1), axis=1)
    
    locs = locs.reshape(N, K, 2) / SIMCC_SPLIT_RATIO
    vals = vals.reshape(N, K)
    return locs, vals

'''SortTracker数据处理'''
def convert_bbox_to_z(bbox): #将bbox由[x1,y1,x2,y2]形式转为 [框中心点x,框中心点y,框面积s,宽高比例r]^T
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2]-bbox[0]
    h = bbox[3]-bbox[1]
    x = bbox[0]+w/2.
    y = bbox[1]+h/2.
    s = w*h    #scale is just area
    r = w/float(h)
    return np.array([x,y,s,r]).reshape((4,1))  #将数组转为4行一列形式，即[x,y,s,r]^T
 
def convert_x_to_bbox(x,score=None): #将[x,y,s,r]形式的bbox，转为[x1,y1,x2,y2]形式
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2]*x[3])  #w=sqrt(w*h * w/h)
    h = x[2]/w              #h=w*h/w
    if(score==None): #如果检测框不带置信度
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))  #返回[x1,y1,x2,y2]
    else:            #如果加测框带置信度
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5)) #返回[x1,y1,x2,y2,score]
 
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.2):  #用于将检测与跟踪进行关联
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if detections is None:
        return np.empty((0,2),dtype=int), np.empty((0,5),dtype=int), np.arange(len(trackers))
    if len(trackers)==0:  #如果跟踪器为空
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32) # 检测器与跟踪器IOU矩阵
    
    for d,det in enumerate(detections):
        iou_matrix[d,:] = compute_iou(det,trackers)  #计算检测器与跟踪器的IOU并赋值给IOU矩阵对应位置
        # for t,trk in enumerate(trackers):
        #     iou_matrix[d,t] = compute_iou(det,trk)   #计算检测器与跟踪器的IOU并赋值给IOU矩阵对应位置
    matched_indices = linear_sum_assignment(-iou_matrix)    # 参考：https://blog.csdn.net/herr_kun/article/details/86509591    加上负号是因为linear_assignment求的是最小代价组合，而我们需要的是IOU最大的组合方式，所以取负号
    matched_indices = np.asarray(matched_indices)
    matched_indices = np.transpose(matched_indices)
    unmatched_detections = []    #未匹配上的检测器
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):  #如果检测器中第d个检测结果不在匹配结果索引中，则d未匹配上
            unmatched_detections.append(d)
    unmatched_trackers = []      #未匹配上的跟踪器
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):  #如果跟踪器中第t个跟踪结果不在匹配结果索引中，则t未匹配上
            unmatched_trackers.append(t)
            # if t == 0:
            #     print('debug')
    
    #filter out matched with low IOU   过滤掉那些IOU较小的匹配对
    matches = []  #存放过滤后的匹配结果
    for m in matched_indices:   #遍历粗匹配结果
        if(iou_matrix[m[0],m[1]]<iou_threshold):   #m[0]是检测器ID， m[1]是跟踪器ID，如它们的IOU小于阈值则将它们视为未匹配成功
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))          #将过滤后的匹配对维度变形成1x2形式
    if(len(matches)==0):           #如果过滤后匹配结果为空，那么返回空的匹配结果
        matches = np.empty((0,2),dtype=int)  
    else:                          #如果过滤后匹配结果非空，则按0轴方向继续添加匹配对
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)  #其中跟踪器数组是5列的（最后一列是ID）


'''
动作分类
'''
'''关键点归一化'''
'''基于关键点自身进行最大最小归一化'''
'''对每一帧单独，因此保留了关键点相对运动信息'''
def keypoint_normalization_within_frame(keypoints):

    normalized_keypoints = keypoints.copy()
    for i in range(keypoints.shape[0]):
        normalized_keypoints[i, :, 0] = (keypoints[i, :, 0] - keypoints[i, :, 0].min()) / (keypoints[i, :, 0].max() - keypoints[i, :, 0].min())
        normalized_keypoints[i, :, 1] = (keypoints[i, :, 1] - keypoints[i, :, 1].min()) / (keypoints[i, :, 1].max() - keypoints[i, :, 1].min())
    return normalized_keypoints

'''分析上升下降事件发生索引的字典'''    
def analyze_and_update_action_probabilities(action_probabilities, threshold=0.35):
    # 初始化记录事件发生索引的字典
    event_indices = {'rise': None, 'fall': None}

    # 遍历列表寻找上升和下降事件
    for i in range(1, len(action_probabilities)):
        if action_probabilities[i] > threshold and action_probabilities[i-1] <= threshold:
            event_indices['rise'] = i
        elif action_probabilities[i] < threshold and action_probabilities[i-1] >= threshold:
            event_indices['fall'] = i
    
    return event_indices

def update_sequence(keypoints, primary_id, window_size, action_model, kptseq_with_id, 
                      actionseq_with_id, actioncounter_with_id, action_prob_sequences_with_id, action_timestamps_with_id):
    # 遍历每个关键点和ID
    # 初始化字典
    if primary_id not in kptseq_with_id:
        kptseq_with_id[primary_id] = []
    if primary_id not in actionseq_with_id:
        actionseq_with_id[primary_id] = [0 for _ in range(window_size-1)]
    if primary_id not in actioncounter_with_id:
        # 将其他id的动作计数累加到primary_id中
        actioncounter_with_id[primary_id] = {0: 0, 1: 0, 2: 0, 3: 0}
        for id, action_counter in actioncounter_with_id.items():
            if id != primary_id:
                for action_type, count in action_counter.items():
                    actioncounter_with_id[primary_id][action_type] += count
    if primary_id not in action_prob_sequences_with_id:
        action_prob_sequences_with_id[primary_id] = {0: [0 for _ in range(window_size-1)], 1: [0 for _ in range(window_size-1)], 
                                                2: [0 for _ in range(window_size-1)], 3: [0 for _ in range(window_size-1)]}
    if primary_id not in action_timestamps_with_id:
        action_timestamps_with_id[primary_id] = {0: [], 1: [], 2: [], 3: []}
        for id, action_timestamps in action_timestamps_with_id.items():
            if id != primary_id:
                for action_type, timestamps in action_timestamps.items():
                    action_timestamps_with_id[primary_id][action_type].extend(timestamps)
    # if primary_id not in action_prob_sequences_with_id_static:
    #     action_prob_sequences_with_id_static[primary_id] = {0: [0 for _ in range(window_size-1)], 1: [0 for _ in range(window_size-1)], 
    #                                                 2: [0 for _ in range(window_size-1)], 3: [0 for _ in range(window_size-1)]}
    kptseq_with_id[primary_id].append(keypoints)
    
    # 处理窗口
    if len(kptseq_with_id[primary_id]) >= window_size:
        kptseq_with_id[primary_id] = kptseq_with_id[primary_id][-window_size:]
        current_window = np.array(kptseq_with_id[primary_id])
        current_window = keypoint_normalization_within_frame(current_window).reshape(1, window_size, -1)
        # clasify_result = action_model(current_window)['output']
        ort_inputs = {action_model.get_inputs()[0].name: current_window}
        clasify_result = action_model.run(['output'], ort_inputs)[0]
        action_probs = clasify_result.squeeze().tolist()  # 转换为列表
        for action_type, prob in enumerate(action_probs):
            action_prob_sequences_with_id[primary_id][action_type].append(prob)
            # action_prob_sequences_with_id_static[primary_id][action_type].append(prob)
        shot_type1 = np.argmax(clasify_result, axis=1)
        actionseq_with_id[primary_id].append(shot_type1[0])
            
def update_action_counter(action_prob_sequences_with_id, actioncounter_with_id, frames_thr, action_timestamps_with_id, frame_id):
    # 分析动作概率序列并更新动作计数器
    for id, prob_sequences in action_prob_sequences_with_id.items():
        for action_type, prob_sequence in prob_sequences.items():
            if len(prob_sequence) >= frames_thr:
                events = analyze_and_update_action_probabilities(prob_sequence)
                if events['rise'] is not None and events['fall'] is not None:
                    if events['fall'] - events['rise'] >= frames_thr: # 上升后下降,且持续时间超过frames_thr帧
                        actioncounter_with_id[id][action_type] += 1
                        action_prob_sequences_with_id[id][action_type] = prob_sequence[events['fall']:]
                        action_timestamps_with_id[id][action_type].append([frame_id-events['fall']+events['rise'], frame_id])
                    elif 0 < events['fall'] - events['rise'] < frames_thr: # 上升后下降，但持续时间不足frames_thr帧
                        action_prob_sequences_with_id[id][action_type] = prob_sequence[events['fall']:]
                    else:  # 下降后上升
                        action_prob_sequences_with_id[id][action_type] = prob_sequence[events['rise']-1:]
                elif events['rise'] is not None:  # 仅发生上升
                    action_prob_sequences_with_id[id][action_type] = prob_sequence[events['rise']-1:]
                elif events['fall'] is not None:  # 仅发生下降
                    action_prob_sequences_with_id[id][action_type] = prob_sequence[events['fall']:]
                else:  # 两者都没有发生
                    action_prob_sequences_with_id[id][action_type] = prob_sequence[-frames_thr:]
                