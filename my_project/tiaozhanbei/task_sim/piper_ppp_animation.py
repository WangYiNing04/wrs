'''
Author: wang yining
Date: 2025-10-30 19:30:15
LastEditTime: 2025-10-31 09:11:36
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/task_sim/piper_ppp_animation.py
Description: 
e-mail: wangyining0408@outlook.com
'''
import math

from wrs import wd, rm, mgm, mcm, cbt, gg, ppp, rrtc
import wrs.robot_sim.robots.piper.piper_dual_arm as pda
import numpy as np
from direct.task.TaskManagerGlobal import taskMgr

base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
mgm.gen_frame().attach_to(base)
# ground
ground = mcm.gen_box(xyz_lengths=rm.vec(5, 5, 1), rgb=rm.vec(.7, .7, .7), alpha=1)
ground.pos = rm.np.array([0, 0, -.5])
ground.attach_to(base)
ground.show_cdprim()
## object holder
holder_1 = mcm.CollisionModel("/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl")
holder_1.rgba = rm.np.array([.5, .5, .5, 1])
h1_gl_pos = rm.np.array([.35, -.5, .0])
h1_gl_rotmat = rm.rotmat_from_euler(0, 0, -math.pi/4)
holder_1.pos = h1_gl_pos
holder_1.rotmat = h1_gl_rotmat
mgm.gen_frame().attach_to(holder_1)
# visualize a copy
h1_copy = holder_1.copy()
h1_copy.attach_to(base)
# h1_copy.show_cdprim()
## object holder goal
holder_2 = mcm.CollisionModel("/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl")
h2_gl_pos = rm.np.array([.4, -0.3, .0])
h2_gl_rotmat = rm.rotmat_from_euler(0, 0, 0)
holder_2.pos = h2_gl_pos
holder_2.rotmat = h2_gl_rotmat
# visualize a copy
h2_copy = holder_2.copy()
h2_copy.rgb = rm.const.tab20_list[0]
h2_copy.alpha = .3
h2_copy.attach_to(base)
# h2_copy.show_cdprim()
box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
box2.attach_to(base)
obs_list = [box2]
robot = pda.DualPiperNoBody()
rbt_s = robot.rgt_arm
robot.gen_meshmodel().attach_to(base)
# robot.cc.show_cdprim()
# base.run()

rrtc = rrtc.RRTConnect(rbt_s)
ppp = ppp.PickPlacePlanner(rbt_s)

grasp_collection = gg.GraspCollection.load_from_disk(file_name='/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/block_grasps.pickle')
start_conf = rbt_s.get_jnt_values()

goal_pose_list = [(h2_gl_pos, h2_gl_rotmat)]
mot_data = ppp.gen_pick_and_place(obj_cmodel=holder_1,
                                  end_jnt_values=start_conf,
                                  grasp_collection=grasp_collection,
                                  goal_pose_list=goal_pose_list,
                                  place_approach_distance_list=[.05] * len(goal_pose_list),
                                  place_depart_distance_list=[.05] * len(goal_pose_list),
                                  pick_approach_distance=.05,
                                  pick_depart_distance=.05,
                                  pick_depart_direction=rm.const.z_ax,
                                  use_rrt=True)


class Data(object):
    def __init__(self, mot_data):
        self.counter = 0
        self.mot_data = mot_data


anime_data = Data(mot_data)


def update(anime_data, task):
    if anime_data.counter > 0:
        anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
    if anime_data.counter >= len(anime_data.mot_data):
        for mesh_model in anime_data.mot_data.mesh_list:
            mesh_model.detach()
        anime_data.counter = 0
    mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
    mesh_model.attach_to(base)
    mesh_model.show_cdprim()
    if base.inputmgr.keymap['space']:
        anime_data.counter += 1
    return task.again


taskMgr.doMethodLater(0.01, update, "update",
                      extraArgs=[anime_data],
                      appendTask=True)

base.run()
