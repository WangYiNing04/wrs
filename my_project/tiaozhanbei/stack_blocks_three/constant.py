'''
Author: wang yining
Date: 2025-10-24 21:59:29
LastEditTime: 2025-10-28 14:40:59
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/constant.py
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
YOLO_MODEL_BLOCKS_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/best_block.pt"

# ======================
# 方块模型路径
# ======================
BLOCK_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl"

# ======================
# 抓取姿态存储路径
# ======================

#/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/block_grasps.pickle
#/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/manual_grasps.pickle
GRASP_PATH_BLOCKS = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/block_grasps.pickle"

# ======================
# 轨迹导出目录
# ======================
TRAJ_DIR = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/exported"

TARGET_POSITIONS = [
    [0.25, -0.30, 0.00],
    [0.25, -0.30, 0.05],
    [0.25, -0.30, 0.10]
]

# MIDDLE_CAM_C2W = np.array([
#     [0.0090370, -0.6821888, 0.7311201, -0.00295266],
#     [-0.9999384, -0.0108772, 0.0022105, -0.28066693],
#     [0.0064445, -0.7310951, -0.6822451, 0.51193761],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# MIDDLE_CAM_C2W = np.array(
#     [[-0.007095074546184508, -0.7024172019490986, 0.7117301012163516, 0.0010473400000000016], 
#      [-0.9994108113434423, 0.028882953839319204, 0.01854212767861934, -0.30716693000000005], 
#      [-0.03358117994085078, -0.7111791988900975, -0.7022082731221432, 0.5094376100000003], 
#      [0.0, 0.0, 0.0, 1.0]]
# )

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
