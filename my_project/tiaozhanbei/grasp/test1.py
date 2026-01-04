'''
Author: wang yining
Date: 2025-10-21 12:41:27
LastEditTime: 2025-10-28 13:24:20
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/filter_grasp.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg, gpa
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import *
import numpy as np 

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
grasp_collection = GraspCollection()
#manual_grasps.pickle manual_grasps filter_grasps piper_gripper_grasps.pickle
grasp_collection = grasp_collection.load_from_disk(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/shoe_grasps.pickle')
gripper = pg.PiperGripper()

# 创建新的抓取集合来保存过滤后的抓取
filtered_grasp_collection = GraspCollection()


#print(grasp.ee_values)
#gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
#gripper.gen_meshmodel(alpha=1).attach_to(base)


# for grasp in grasp_collection:
#     # 获取末端执行器Z轴在世界坐标下的方向
#     z_axis_world = grasp.ac_rotmat[:, 2]

#     # 如果Z轴朝下（即z分量<0），跳过该抓取
#     if z_axis_world[2] < 0.0:  # 或者用 < 0.3 更宽松一些
#         continue

#     # 其它过滤条件
#     if grasp.ee_values <= 0.06 and grasp.ac_pos[2] >= 0.05:
#         filtered_grasp_collection.append(grasp)


# 过滤条件：
# 1️⃣ ee_values <= 0.06
# 2️⃣ ac_pos[2] >= 0.05 (高度大于等于 5cm)
for grasp in grasp_collection:
    # 获取末端执行器Z轴在世界坐标下的方向
    z_axis_world = grasp.ac_rotmat[:, 2]

    y_axis_world = grasp.ac_rotmat[:, 1]
    # 如果Z轴朝下（即z分量<0），跳过该抓取
    # if z_axis_world[2] > 0.0 or y_axis_world[1] > 0.0 or grasp.ac_pos[2] < 0.075:  # 或者用 < 0.3 更宽松一些
    #     continue
    # if grasp.ac_pos[2] > 0.12 or grasp.ac_pos[2] < 0.02:
    #     continue
    if z_axis_world[2] > 0.0:
        continue
    # if y_axis_world[1] > 0.0:
    #     continue
    if grasp.ee_values > 0.07:
        continue
    if grasp.ee_values < 0.05:
        continue

    # if grasp.ee_values < 0.08 :
    #     continue
    # if grasp.ee_values <= 0.06 and grasp.ac_pos[2] >= 0.05:
    #     filtered_grasp_collection.append(grasp)

    filtered_grasp_collection.append(grasp)


# 保存过滤后的抓取集合
filtered_grasp_collection.save_to_disk(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/place_shoe/filter_shoe_grasps.pickle')

obj_cmodel = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/shoes.stl")
obj_cmodel.pos=np.array([0,0,0])
obj_cmodel.show_local_frame()
obj_cmodel.attach_to(base)
base.run()