# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 18:08
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
import yaml



def load_yaml_config(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    return conf

