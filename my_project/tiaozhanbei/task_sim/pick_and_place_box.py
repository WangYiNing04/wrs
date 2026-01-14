#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/19 13:02
# @Author : ZhangXi
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg
import wrs.robot_sim.robots.piper.piper_single_arm as psa
import numpy as np
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from direct.task.TaskManagerGlobal import taskMgr

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
ground.pos = rm.np.array([0, 0, -.5])
ground.attach_to(base)
ground.show_cdprim()

## object holder (起始位置)
<<<<<<< HEAD
holder_1 = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl")
holder_1.rgba = rm.np.array([.5, .5, .5, 1])
h1_gl_pos = np.array([0.27036765, -0.21372601, 0])
h1_gl_rotmat = rm.rotmat_from_euler(0,0,0)
=======
holder_1 = mcm.CollisionModel(r"F:\wrs_tiaozhanbei\0000_examples\objects\tiaozhanbei\cup.stl")
holder_1.rgba = rm.np.array([.5, .5, .5, 1])
h1_gl_pos = np.array([0.3397, -0.2887, 0])
#h1_gl_rotmat = rm.rotmat_from_euler(2.8813, 0.2080, 2.4237)
h1_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
holder_1.pos = h1_gl_pos
holder_1.rotmat = h1_gl_rotmat
mgm.gen_frame().attach_to(holder_1)
# visualize a copy
h1_copy = holder_1.copy()
h1_copy.attach_to(base)
h1_copy.show_cdprim()

<<<<<<< HEAD
holder_2 = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl")
h2_gl_pos = np.array([0.25 ,-0.3, 0])
h2_gl_rotmat = rm.rotmat_from_euler(0,0,0)
#holder_2 = mcm.CollisionModel(r"F:\wrs_tiaozhanbei\0000_examples\objects\tiaozhanbei\cup.stl")

=======
holder_2 = mcm.CollisionModel(r"F:\wrs_tiaozhanbei\0000_examples\objects\tiaozhanbei\cup.stl")
h2_gl_pos = np.array([0.378, -0.099417, 0])
#h2_gl_rotmat = rm.rotmat_from_euler(3.0369, -0.0483, 2.7970)
h2_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
holder_2.pos = h2_gl_pos
holder_2.rotmat = h2_gl_rotmat
# visualize a copy
h2_copy = holder_2.copy()
h2_copy.rgb = rm.const.tab20_list[0]
h2_copy.alpha = .3
h2_copy.attach_to(base)
h2_copy.show_cdprim()

## Piper 机器人
robot = psa.PiperSglArm()
robot.gen_meshmodel().attach_to(base)
# robot.cc.show_cdprim()
# base.run()

# 实例化 RRTConnect 和 PickPlacePlanner
rrtc = rrtc.RRTConnect(robot)
ppp = ppp.PickPlacePlanner(robot)

grasp_collection = gg.GraspCollection.load_from_disk(
<<<<<<< HEAD
    file_name=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/task_sim/manual_grasps.pickle')
=======
    file_name=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\grasp\piper_gripper_grasps.pickle')
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
start_conf = robot.get_jnt_values()
goal_pose_list = [(h2_gl_pos, h2_gl_rotmat)]
box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5]))
box1.attach_to(base)
box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
box2.attach_to(base)

obstacle_list = [box1, box2]

# 生成 Pick and Place 运动
mot_data = ppp.gen_pick_and_place(
    obj_cmodel=holder_1,
    end_jnt_values=start_conf,
    grasp_collection=grasp_collection,
    goal_pose_list=goal_pose_list,
<<<<<<< HEAD
    place_approach_distance_list=[.05],
    place_depart_distance_list=[.05],
=======
    place_approach_distance_list=[.05] * len(goal_pose_list),
    place_depart_distance_list=[.05] * len(goal_pose_list),
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    pick_approach_distance=.05,
    pick_depart_distance=.05,
    pick_depart_direction=rm.const.z_ax,
    obstacle_list=obstacle_list,
    use_rrt=True)

<<<<<<< HEAD
print(mot_data[0])
print(mot_data.mesh_list)
print(type(mot_data))
=======

>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)
<<<<<<< HEAD
 
=======


>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        anime_data.counter = 0

    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    mesh_model.show_cdprim()

    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again

<<<<<<< HEAD
=======

>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
