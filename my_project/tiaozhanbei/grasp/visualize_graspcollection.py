'''
Author: wang yining
Date: 2025-10-21 12:41:27
<<<<<<< HEAD
LastEditTime: 2025-10-31 12:05:41
=======
LastEditTime: 2025-10-21 16:02:32
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/visualize_graspcollection.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg, gpa
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import *
<<<<<<< HEAD
import numpy as np 
from collections import defaultdict

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
grasp_collection = GraspCollection()
#manual_grasps.pickle manual_grasps filter_grasps piper_gripper_grasps.pickle
grasp_collection = grasp_collection.load_from_disk(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/manual_grasps/tea_dongfang_manual_grasps.pickle')
gripper = pg.PiperGripper()
print(len(grasp_collection))
print(f"总抓取姿态数量: {len(grasp_collection)}")

# 统计不同夹爪宽度的数量
width_counts = defaultdict(int)
for grasp in grasp_collection:
    print(grasp.ac_pos)
    width = grasp.ee_values  # 假设ee_values代表夹爪宽度
    width_counts[width] += 1

#夹爪宽度筛选

# 打印统计结果
print("\n不同夹爪宽度统计:")
for width, count in sorted(width_counts.items()):
    print(f"宽度 {width:.4f}: {count} 个")

#高度筛选

# 统计抓取点高度小于 5cm 的数量
height = 0.02
low_height_count = sum(1 for grasp in grasp_collection if grasp.ac_pos[2] < height)
print(f"\n高度小于 {height*100}cm 的抓取姿态数量: {low_height_count}")


#可视化有限数量夹爪

i = 0
for grasp in grasp_collection:
    if i > 30:
        break
    #print(grasp.ee_values)

    gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    gripper.gen_meshmodel(alpha=1).attach_to(base)
    i += 1

# #统计向下的抓取数量 可视化向下的抓取
# downward_grasps = [
#     grasp for grasp in grasp_collection
#     if grasp.ac_rotmat[:, 2][2] < 0.0  # Z轴朝下
# ]
# print(f"末端Z轴朝下的抓取数量: {len(downward_grasps)}")

# #统计向下的抓取数量 可视化向下的抓取
# downward_grasps = [
#     grasp for grasp in grasp_collection
#     if grasp.ac_rotmat[:, 0][0] < 0.0  # Y轴朝下
# ]
# print(f"末端Y轴朝下的抓取数量: {len(downward_grasps)}")

# i = 0
# for grasp in downward_grasps:
#     if i > 50:
#         break
#     gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
#     gripper.gen_meshmodel(rgb=[1, 0, 0]).attach_to(base)  # 红色可视化
#     i += 1


# 找出高度小于 5cm 的抓取
# low_grasps = [grasp for grasp in grasp_collection if grasp.ac_pos[2] < 0.05]
# print(f"\n高度小于 2cm 的抓取姿态数量: {len(low_grasps)}")

# 可视化所有高度 < 5cm 的抓取
# for grasp in low_grasps:
#     gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
#     gripper.gen_meshmodel(rgb=[1, 0, 0]).attach_to(base)  # 红色半透明


#block.stl
#bowl.stl
#Coke_can.stl 
#tea dongfang.stl
#water Ganten.stl
obj_cmodel = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/tea dongfang.stl")
obj_cmodel.pos=np.array([0,0,0])
obj_cmodel.show_local_frame()
obj_cmodel.attach_to(base)
=======

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
grasp_collection = GraspCollection()
grasp_collection = grasp_collection.load_from_disk(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/manual_grasps.pickle')
gripper = pg.PiperGripper()
for grasp in grasp_collection:
    gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
    gripper.gen_meshmodel(alpha=1).attach_to(base)

>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
base.run()