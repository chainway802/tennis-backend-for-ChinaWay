# -*- encoding: utf-8 -*-
"""
@author   :   yykzjh    
@Contact  :   1378453948@qq.com
@DateTime :   2024/04/20 20:32
@Version  :   1.0
@License  :   (C)Copyright 2024
"""
from service import AlgorithmService

import os
os.chdir("tennis-backend-new/tennis-backend")



if __name__ == '__main__':

    service = AlgorithmService(r"./config/config.yaml")
    service.start()
