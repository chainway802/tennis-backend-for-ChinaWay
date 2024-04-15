# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/15 15:48
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
from kafka import KafkaConsumer, KafkaProducer
from concurrent.futures import ThreadPoolExecutor

import utils

video_clip_consumer = None
AI_coach_consumer = None
result_producer = None
video_clip_consume_threads = []
AI_coach_consume_threads = []

thread_pool = None


def process_message(message):
    if message.topic == "video-clip":
        pass
    elif message.topic == "video-AI-analyze":
        pass


def parse_argument(conf):
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-nw1", type=int, help="读取video_clip订阅的消息队列的num_workers")
    parser.add_argument("-nw2", type=int, help="读取AI_coach订阅的消息队列的num_workers")
    parser.add_argument("-mw", type=int, help="线程池的max_workers")
    args = parser.parse_args()

    # 依次更新配置
    if args.nw1 is not None:
        conf["service"]["kafka"]["video_clip"]["num_workers"] = args.nw1
    if args.nw2 is not None:
        conf["service"]["kafka"]["AI_coach"]["num_workers"] = args.nw2
    if args.mw is not None:
        conf["service"]["thread_pool"]["max_workers"] = args.mw

    return conf


def consume_data(consumer):
    for message in consumer:
        thread_pool.submit(process_message, message)


def produce_message(producer):
    message = {
        "key": 1,
        "value": "test"
    }
    topics = ["video-clip", "video-AI-analyze"]
    # 发送字典到指定的主题
    ind = 1
    while True:
        topic = random.choice(topics)
        send_data = {
            "key": ind,
            "value": message["value"] + "_" + str(ind)
        }
        ind += 1
        producer.send(topic, send_data)
        producer.flush()
        time.sleep(0.5)


def main():
    # 读取配置文件
    conf = utils.load_yaml_config(r"./config.yaml")

    # 解析命令行参数，更新配置
    conf = parse_argument(conf)

    # 初始化消息队列
    global video_clip_consumer
    global AI_coach_consumer
    global result_producer
    # 消费者配置
    consumer_config = {
        'bootstrap_servers': conf["service"]["kafka"]["bootstrap_servers"],
        'group_id': conf["service"]["kafka"]["group_id"],
        'auto_offset_reset': conf["service"]["kafka"]["auto_offset_reset"],
        'value_deserializer': lambda m: json.loads(m.decode('utf-8'))
    }
    # 初始化消费者
    video_clip_consumer = KafkaConsumer(conf["service"]["kafka"]["video_clip"]["consumer"], **consumer_config)
    AI_coach_consumer = KafkaConsumer(conf["service"]["kafka"]["AI_coach"]["consumer"], **consumer_config)
    # 生产者配置
    producer_config = {
        'bootstrap_servers': conf["service"]["kafka"]["bootstrap_servers"],
        'value_serializer': lambda m: json.dumps(m).encode('utf-8')
    }
    # 初始化AI分析结果生产者
    result_producer = KafkaProducer(**producer_config)

    # 初始化线程池
    global thread_pool
    thread_pool = ThreadPoolExecutor(max_workers=conf["service"]["thread_pool"]["max_workers"])

    # 启动各消息队列的消费线程
    global video_clip_consume_threads
    global AI_coach_consume_threads
    for i in range(conf["service"]["kafka"]["video_clip"]["num_workers"]):
        t = threading.Thread(target=consume_data, args=(video_clip_consumer,))
        t.start()
        video_clip_consume_threads.append(t)
    for i in range(conf["service"]["kafka"]["AI_coach"]["num_workers"]):
        t = threading.Thread(target=consume_data, args=(AI_coach_consumer,))
        t.start()
        AI_coach_consume_threads.append(t)

    # # 启动往消息队列写数据的线程进行测试
    # producer = KafkaProducer(**producer_config)
    # produce_threads = []
    # for i in range(2):
    #     t = threading.Thread(target=produce_message, args=(producer,))
    #     t.start()
    #     produce_threads.append(t)
    # for t in produce_threads:
    #     t.join()

    # 主进程阻塞
    for t in video_clip_consume_threads:
        t.join()
    for t in AI_coach_consume_threads:
        t.join()


if __name__ == '__main__':
    main()
