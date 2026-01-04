'''
Author: wang yining
Date: 2025-10-27 14:07:43
LastEditTime: 2025-10-27 20:38:07
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/reason_filter_grasp.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg
import wrs.robot_sim.robots.piper.piper_single_arm as psa
import numpy as np
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from direct.task.TaskManagerGlobal import taskMgr
planner = ppp.PickPlacePlanner(robot)
id = [34, 35, 38, 39, 41, 44, 45, 46]
grasp_collection = gg.GraspCollection.load_from_disk(file_name=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/task_sim/piper_gripper_grasps.pickle')


filter_grasps = gg.GraspCollection()

for i in id:
    filter_grasps.append(grasp_collection[i])

filter_grasps.save_to_disk(file_name='filter_grasps.pickle')