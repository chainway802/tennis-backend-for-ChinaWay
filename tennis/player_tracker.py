from filterpy.kalman import KalmanFilter  #filterpy包含了一些常用滤波器的库
import numpy as np
from .lib.utils.process import convert_bbox_to_z, convert_x_to_bbox, associate_detections_to_trackers, compute_iou

__all__ = [
    "SortTracker"
]

class KalmanBoxTracker(object):
    """
    This class represents the internel state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox):
        """
        Initialises a tracker using initial bounding box.  使用初始边界框初始化跟踪器
        """
        #define constant velocity model                #定义匀速模型
        self.kf = KalmanFilter(dim_x=7, dim_z=4)       #状态变量是7维， 观测值是4维的，按照需要的维度构建目标
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities 对未观测到的初始速度给出高的不确定性
        self.kf.P *= 10.          # 默认定义的协方差矩阵是np.eye(dim_x)，将P中的数值与10， 1000相乘，赋值不确定性
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
    
        self.kf.x[:4] = convert_bbox_to_z(bbox)  #将bbox转为 [x,y,s,r]^T形式，赋给状态变量X的前4位
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
 
    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]
 
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class SortTracker(object):
    def __init__(self, max_age=30, min_hits=3, resolution=(1920,1080)):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        self.resolution = resolution
        self.primary_id = None
    
    def update(self, dets, racket_dets):  #输入的是检测结果[x1,y1,x2,y2,score]形式
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections.   #每一帧都得调用一次，即便检测结果为空
        Returns the a similar array, where the last column is the object ID.                    #返回相似的数组，最后一列是目标ID
        NOTE: The number of objects returned may differ from the number of detections provided.  #返回的目标数量可能与提供的检测数量不同
        """
        self.frame_count += 1   #帧计数
        #get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5)) # 根据当前所有卡尔曼跟踪器的个数创建二维零矩阵，维度为：卡尔曼跟踪器ID个数x 5 (这5列内容为bbox与ID)
        to_del = []                             #存放待删除
        ret = []                                #存放最后返回的结果
        ret_d = []                              #存放匹配成功的检测结果
        for t,trk in enumerate(trks):      #循环遍历卡尔曼跟踪器列表
            pos = self.trackers[t].predict()[0]           #用卡尔曼跟踪器t 预测 对应物体在当前帧中的bbox
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if(np.any(np.isnan(pos))):                     #如果预测的bbox为空，那么将第t个卡尔曼跟踪器删除
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  #将预测为空的卡尔曼跟踪器所在行删除，最后trks中存放的是上一帧中被跟踪的所有物体在当前帧中预测的非空bbox
        for t in reversed(to_del): #对to_del数组进行倒序遍历
            self.trackers.pop(t)   #从跟踪器中删除 to_del中的上一帧跟踪器ID

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)  #对传入的检测结果 与 上一帧跟踪物体在当前帧中预测的结果做关联，返回匹配的目标矩阵matched, 新增目标的矩阵unmatched_dets, 离开画面的目标矩阵unmatched_trks
    
        #update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):    # 对卡尔曼跟踪器做遍历
            if(t not in unmatched_trks):                   #如果上一帧中的t还在当前帧画面中（即不在当前预测的离开画面的矩阵unmatched_trks中）
                d = matched[np.where(matched[:,1]==t)[0],0]  #说明卡尔曼跟踪器t是关联成功的，在matched矩阵中找到与其关联的检测器d
                ret_d.append(np.concatenate((dets[d,:][0],[trk.id+1])).reshape(1,-1))                              #将关联成功的检测结果d存入ret_d
                trk.update(dets[d,:][0])                     #用关联的检测结果d来更新卡尔曼跟踪器（即用后验来更新先验）
          
        #create and initialise new trackers for unmatched detections  #对于新增的未匹配的检测结果，创建并初始化跟踪器
        for i in unmatched_dets:                  #新增目标
            trk = KalmanBoxTracker(dets[i,:])     #将新增的未匹配的检测结果dets[i,:]传入KalmanBoxTracker
            self.trackers.append(trk)             #将新创建和初始化的跟踪器trk 传入trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):       #对新的卡尔曼跟踪器集进行倒序遍历
            d = trk.get_state()[0]                #获取trk跟踪器的状态 [x1,y1,x2,y2] 
            #将d的x1,y1,x2,y2限制在画面内
            d[0] = max(0, d[0])                   
            d[1] = max(0, d[1])
            d[2] = min(self.resolution[0], d[2])
            d[3] = min(self.resolution[1], d[3])
            
            # 找到trk对应的检测结果dets的索引
            if((trk.time_since_update < self.max_age) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
            i -= 1
            #remove dead tracklet
            if(trk.time_since_update > self.max_age):
                # if i == 0:
                #     print('debug')
                self.trackers.pop(i)
        
        
                
        if(len(ret)>0):
            ret_trackers = np.array(ret).squeeze(1)
            # 如果primary_id不在trackers中，将其置为None，并重新利用球拍位置确定运动员id
            if  self.primary_id not in ret_trackers[:,4]:
                self.primary_id = None
                # 计算trackers与racket_bboxes的iou, 重新确定运动员id
                if (racket_dets is not None) and (len(ret) > 0):
                    racket_iou = compute_iou(racket_dets.squeeze(), ret_trackers)
                    # 取出iou最大的对应的trackers的id
                    self.primary_id = ret_trackers[np.argmax(racket_iou), 4]
            return ret_trackers, ret_d, self.primary_id
        return np.empty((0,5)), np.empty((0,1,5)), None
   