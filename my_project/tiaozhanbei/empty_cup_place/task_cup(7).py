#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/28
# @Author : ZhangXi

import os
import time
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
import wrs.basis.robot_math as rm
from wrs.vision.depth_camera.util_functions import registration_ptpt
import wrs.modeling.geometric_model as gm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.empty_cup_place.constant import YOLO_MODEL_CUPS_PATH, CUP_MODEL_PATH, GRASP_PATH_CUPS, MIDDLE_CAM_C2W
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat

class MultiCameraCupTask:
    def __init__(self):
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.yolo = init_yolo(YOLO_MODEL_CUPS_PATH)

        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W}
        }

    def align_cup_pose(self, base, cup_model_path, real_pcd, visualize=False):
        """ICP 点云配准"""
        cup = mgm.GeometricModel(cup_model_path)
        cup_pcd = cup.sample_surface(radius=0.001, n_samples=8000)
        cup_pcd = cup_pcd[cup_pcd[:, 2] > 0.01]

        if visualize:
            mgm.gen_pointcloud(cup_pcd, rgba=np.array([0, 0, 1, 0.5])).attach_to(base)
            mgm.gen_pointcloud(real_pcd, rgba=np.array([0, 1, 0, 0.5])).attach_to(base)

        print("🥤 开始ICP点云配准...")
        icp_result = registration_ptpt(cup_pcd, real_pcd, downsampling_voxelsize=0.007)
        transformation = icp_result[2]
        cup.homomat = transformation
        cup.attach_to(base)

        aligned_pcd = rm.transform_points_by_homomat(transformation, cup_pcd.copy())
        if visualize:
            mgm.gen_pointcloud(aligned_pcd, rgba=np.array([1, 0, 0, 0.6])).attach_to(base)

        print("✅ ICP配准完成")
        return transformation

    def detect_cups(self, show=False):
        """用点云ICP替代YOLO点云"""
        # 读取点云数据（假设已保存为pkl）
        with open('cup_scene_world_color.pkl', 'rb') as f:
            pcd_data = pickle.load(f)
        cup_pcd_real = pcd_data['points']
        # 过滤
        cup_pcd_real = cup_pcd_real[(cup_pcd_real[:,2]>0.008) & (cup_pcd_real[:,2]<0.09) &
                                    (cup_pcd_real[:,0]<0.6) & (cup_pcd_real[:,1]<-0.05) &
                                    (cup_pcd_real[:,1]>-0.6)]
        # 模型点云
        cup = gm.GeometricModel(CUP_MODEL_PATH)
        cup_pcd_model = cup.sample_surface(radius=0.001, n_samples=10000)
        cup_pcd_model = cup_pcd_model[cup_pcd_model[:,2]>0.005]

        # ICP配准
        icp_result = registration_ptpt(cup_pcd_model, cup_pcd_real, downsampling_voxelsize=0.007)
        transformation = icp_result[2]
        cup.homomat = transformation

        # 可视化
        base = wd.World(cam_pos=[.6, .6, .4], lookat_pos=[0,0,0])
        mgm.gen_frame().attach_to(base)
        cup.attach_to(base)
        aligned_pcd = rm.transform_points_by_homomat(transformation, cup_pcd_model.copy())
        mgm.gen_pointcloud(aligned_pcd, rgba=np.array([1,0,0,0.6])).attach_to(base)
        mgm.gen_pointcloud(cup_pcd_real, rgba=np.array([0,1,0,0.5])).attach_to(base)
        base.run()

        # 返回质心和点云，用于后续抓取
        centroid = aligned_pcd.mean(axis=0)
        return [(0, centroid.tolist(), aligned_pcd)]

    def process_gripper_data(self, data, threshold=0.1):
        arr = np.array(data)
        binary_arr = (arr > threshold).astype(int)
        change_indices = np.where(np.diff(binary_arr) != 0)[0] + 1
        return binary_arr.tolist(), change_indices.tolist()

    def split_trajectory_by_gripper(self, jv, change_indices, threshold=0.05):
        if len(change_indices) != 2:
            raise ValueError(f"期望2个变化点,但找到{len(change_indices)}个")
        cp1, cp2 = change_indices
        stage1 = jv[:cp1]
        stage2 = jv[cp1:cp2]
        stage3 = jv[cp2:]
        return stage1, stage2, stage3

    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH_CUPS):
            return
        obj = mcm.CollisionModel(CUP_MODEL_PATH)
        obj.attach_to(base)
        grasps = gpa.plan_gripper_grasps(self.gripper, obj,
                                         angle_between_contact_normals=rm.radians(175),
                                         rotation_interval=rm.radians(15),
                                         max_samples=20,
                                         min_dist_between_sampled_contact_points=0.03,
                                         contact_offset=0.01)
        grasps.save_to_disk(GRASP_PATH_CUPS)
        print(f"保存抓取姿态，共 {len(grasps)} 个")

    def execute_pick_place(self, pick_pos, place_pos, arm, robot, obstacles):
        block = mcm.CollisionModel(CUP_MODEL_PATH)
        block.pos = np.array(pick_pos, dtype=float)
        goal_rot = rm.rotmat_from_euler(0,0,0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_CUPS)

        mot_data = planner.gen_pick_and_place(obj_cmodel=block,
                                             end_jnt_values=robot.get_jnt_values(),
                                             grasp_collection=grasps,
                                             goal_pose_list=[(place_pos, goal_rot)],
                                             pick_approach_direction=-rm.const.z_ax,
                                             place_approach_distance_list=[.05],
                                             place_depart_distance_list=[.05],
                                             pick_approach_distance=.05,
                                             pick_depart_distance=.05,
                                             obstacle_list=obstacles,
                                             use_rrt=True)
        if mot_data is None:
            print("⚠️ 轨迹规划失败")
            return None

        jv, ev = mot_data.jv_list, mot_data.ev_list
        binary_arr, change_indices = self.process_gripper_data(ev)
        approach_path, pick_path, depart_path = self.split_trajectory_by_gripper(jv, change_indices)

        arm.open_gripper(width=0.08)
        arm.move_j(jv[0], speed=10, block=True)
        arm.move_jntspace_path(approach_path, speed=10)
        arm.close_gripper()
        arm.move_jntspace_path(pick_path, speed=10)
        arm.open_gripper(width=0.08)
        arm.move_jntspace_path(depart_path, speed=10)
        return mot_data

    def choose_arm(self, pos):
        _, y = pos
        if y[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    def run(self, show_camera=False):
        base = wd.World(cam_pos=[.6, .6, .4], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)

        obstacles = []
        self.create_grasps(base)
        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)

        objects = self.detect_cups(show=show_camera)
        if not objects:
            print("⚠️ 没检测到物品！")
            return False

        # 抓取 ID=1 的物品
        pick_obj = next((pos for cls_id, pos, pcd in objects if cls_id == 1), None)
        pick_pcd_real = next((pcd for cls_id, _, pcd in objects if cls_id == 1), None)

        # 放置位置 ID=0 的物品
        place_obj = next((pos for cls_id, pos, _ in objects if cls_id == 0), None)

        if pick_obj is None or pick_pcd_real is None or place_obj is None:
            print("⚠️ 检测结果不完整！")
            if pick_obj is None:
                print(" -> 错误：未检测到抓取物品 (ID 1)。")
            if pick_pcd_real is None:
                print(" -> 错误：未获取到抓取物品点云 (ID 1)，无法进行 ICP。")
            if place_obj is None:
                print(" -> 错误：未检测到放置物品质心 (ID 0)。")
            return False

        # 对抓取物品执行 ICP 配准
        self.align_cup_pose(base, CUP_MODEL_PATH, pick_pcd_real, visualize=True)
        base.run()

        # 选择抓取机械臂
        arm, robot = self.choose_arm((0, pick_obj))
        print(f"\n 抓取物品 {pick_obj} → 放置 {place_obj}")
        mot_data = self.execute_pick_place(pick_obj, place_obj, arm, robot, obstacles)

        if mot_data is None:
            print("抓取失败")
            return False
        print("✅ 放置完成！")
        return True


def main():
    task = MultiCameraCupTask()
    try:
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)
        success = task.run(show_camera=False)
        print("任务成功 ✅" if success else "任务失败 ❌")
    except KeyboardInterrupt:
        print("⚠️ 捕获 Ctrl+C，机械臂回零")
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)


if __name__ == '__main__':
    main()
