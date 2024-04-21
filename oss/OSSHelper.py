# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 23:34
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import os
import time
import oss2


__all__ = [
    "OSSHelper"
]


class OSSHelper(object):
    """
    调用阿里云OSS服务
    """

    def __init__(self, access_key_id, access_key_secret, endpoint, bucket_name, region, OSS_DOMAIN="oss-cn-shanghai.aliyuncs.com"):
        # 初始化参数
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.region = region
        self.OSS_DOMAIN = OSS_DOMAIN

        # 使用代码嵌入的RAM用户的访问密钥配置访问凭证
        self.auth = oss2.AuthV4(self.access_key_id, self.access_key_secret)
        # 初始化bucket
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name, region=self.region)

    def upload_file(self, event_id, file_local_temp_path):
        """
        上传文件

        :param event_id: 当前事件的id
        :param file_local_temp_path: 文件本地的缓存路径
        :return: oss文件地址
        """
        # 计算oss存储的文件名称
        event_id_str = str(event_id)  # 将事件is转换成字符串
        ms_timestamp_str = str(int(time.time() * 1000))  # 获取毫秒级时间戳
        ext = os.path.splitext(os.path.basename(file_local_temp_path))[1]  # 获取文件后缀
        object_file_name = event_id_str + "_" + ms_timestamp_str + ext

        # 上传文件
        self.bucket.put_object_from_file(object_file_name, file_local_temp_path)

        return "https://" + self.bucket_name + "." + self.OSS_DOMAIN + "/" + object_file_name

    def download_file(self):
        pass

    def delete_file(self):
        pass
