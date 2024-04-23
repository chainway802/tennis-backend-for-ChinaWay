# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 23:17
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import time

import yaml
import json
import random
import argparse

import threading
from dataclasses import asdict
from kafka import KafkaConsumer, KafkaProducer
from concurrent.futures import ThreadPoolExecutor

from entity.VideoAnalysisEntity import VideoAnalysisEntity


def app():
    # 消费者配置
    consumer_config = {
        'bootstrap_servers': "14.116.187.117:9092",
        'group_id': "video-group",
        'auto_offset_reset': "earliest",
        'value_deserializer': lambda m: VideoAnalysisEntity(**json.loads(m.decode('utf-8')))
    }
    # 初始化视频剪辑结果消费者
    video_clip_result_consumer = KafkaConsumer("video-clip-task-notification", **consumer_config)
    # 生产者配置
    producer_config = {
        'bootstrap_servers': "14.116.187.117:9092",
        'value_serializer': lambda m: json.dumps(asdict(m), default=VideoAnalysisEntity.serialize_complex_types).encode('utf-8')
    }
    # 初始化视频剪辑消息生产者
    video_clip_producer = KafkaProducer(**producer_config)
    # 启动多线程
    video_clip_result_consume_threads = []
    video_clip_produce_threads = []
    for i in range(1):
        t = threading.Thread(target=consume_video_clip_result, args=(video_clip_result_consumer,))
        t.start()
        video_clip_result_consume_threads.append(t)
    for i in range(1):
        t = threading.Thread(target=produce_video_clip_message, args=(video_clip_producer,))
        t.start()
        video_clip_produce_threads.append(t)
    # 阻塞
    for t in video_clip_result_consume_threads:
        t.join()
    for t in video_clip_produce_threads:
        t.join()


def produce_video_clip_message(producer):
    ind = 1
    while True:
        video_analysis_entity = VideoAnalysisEntity(id=ind, userId=str(ind), 
            videoUrl='https://astree.oss-cn-shanghai.aliyuncs.com/b5b741a0-a92a-41dc-9a5d-9bbbc8799b44_1713888959185.mp4', funcValue='1')
        ind += 1
        producer.send("video-clip", key=str(video_analysis_entity.id).encode('utf-8'), value=video_analysis_entity)
        producer.flush()
        time.sleep(2)
        break


def consume_video_clip_result(consumer):
    ind = 1
    for message in consumer:
        print(message, '\n')


if __name__ == '__main__':
    # main()

    # 生成视频剪辑消息体
    app()
