'''
Author: wang yining
Date: 2025-10-24 21:59:29
LastEditTime: 2025-10-29 18:19:42
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/constant.py
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
YOLO_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/best_bottle.pt"

# ======================
# 模型路径
# ======================
COKE_CAN_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/Coke can.stl"
WATER_GANTEN_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/water Ganten.stl"
TEA_DONGFANG_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/tea dongfang.stl"

# ======================
# 抓取姿态存储路径
# ======================


COKE_CAN_GRASP_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/filter_Coke_can_grasps.pickle"
WATER_GANTEN_GRASP_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/filter_water_Ganten_grasps100.pickle"
TEA_DONGFANG_GRASP_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/filter_tea_dongfang_grasps100.pickle"

# ======================
# 轨迹导出目录
# ======================
TRAJ_DIR = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/exported"

TARGET_POSITIONS = [
    [0.45, 0.12, 0.15],
    [0.25,-0.3, 0.00],
]

# MIDDLE_CAM_C2W = np.array([
#     [0.0090370, -0.6821888, 0.7311201, -0.00295266],
#     [-0.9999384, -0.0108772, 0.0022105, -0.28066693],
#     [0.0064445, -0.7310951, -0.6822451, 0.51193761],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# LEFT_HAND_EYE = np.array([
#     [-0.06882986630842862, 0.935322218300393, 0.3470371540873631, -0.06109518881982929],
#     [-0.9974575797038769, -0.07095746357893451, -0.006589435084847506, -0.004634302125341375],
#     [0.018461628052671437, -0.34660839753351097, 0.9378282339631908, 0.03970341679422087],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# RIGHT_HAND_EYE = np.array([
#     [-0.06882986630842862, 0.935322218300393, 0.3470371540873631, -0.06109518881982929],
#     [-0.9974575797038769, -0.07095746357893451, -0.006589435084847506, -0.004634302125341375],
#     [0.018461628052671437, -0.34660839753351097, 0.9378282339631908, 0.03970341679422087],
#     [0.0, 0.0, 0.0, 1.0]
# ])
