'''
Author: wang yining
Date: 2025-10-25 17:41:17
LastEditTime: 2025-10-28 15:25:54
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/stack_bowls_three/constant.py
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

TARGET_POSITIONS = [
    [0.25, -0.30, 0.00],
    [0.25, -0.30, 0.01],
    [0.25, -0.30, 0.02]
]


# # 中间摄像头外参
# MIDDLE_CAM_C2W = np.array([
#     [0.0090370, -0.6821888, 0.7311201, -0.00295266],
#     [-0.9999384, -0.0108772, 0.0022105, -0.28066693],
#     [0.0064445, -0.7310951, -0.6822451, 0.51193761],
#     [0.0, 0.0, 0.0, 1.0]
# ])

# # 左右手眼标定矩阵
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


# =======================================
# 🍽 抓碗任务配置（新增）
# =======================================

# YOLO 模型路径（抓碗）
YOLO_MODEL_BOWLS_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_bowls_three/best_bowl.pt"

# 碗模型路径
BOWL_MODEL_PATH = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/bowl.stl"

# 抓碗抓取姿态路径
GRASP_PATH_BOWLS = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_bowls_three/bowl_grasps.pickle"

# 抓碗轨迹导出目录
TRAJ_DIR_BOWLS = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_bowls_three/exported"


