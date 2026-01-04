'''
Author: wang yining
Date: 2025-10-24 21:59:29
LastEditTime: 2025-10-29 17:45:11
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/constant.py
Description: 
e-mail: wangyining0408@outlook.com
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/24 20:03
# @Author : ZhangXi
import numpy as np

#导入外参
from my_project.tiaozhanbei.constants import *

# ======================
# YOLO 模型路径
# ======================
YOLO_MODEL_SHOES_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/best_shoe.pt"

# ======================
# 模型路径
# ======================
SHOE_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/shoes.stl"


# ======================
# 抓取姿态存储路径
# ======================
#/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/shoes_grasps.pickle
#/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/filter_shoe_grasps.pickle
GRASP_PATH_SHOES = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/filter_shoes_grasps.pickle"


# ======================
# 轨迹导出目录
# ======================
TRAJ_DIR = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/exported"

TARGET_POSITIONS = [
    [0.25, 0.2, 0.05],
]

