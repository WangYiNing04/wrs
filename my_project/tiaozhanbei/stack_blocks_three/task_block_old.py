#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/18 9:34
# @Author : ZhangXi

import os
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from my_project.tiaozhanbei.stack_blocks_three.constant import YOLO_MODEL_PATH, BLOCK_MODEL_PATH, GRASP_PATH, TRAJ_DIR, \
    TARGET_POSITIONS, MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE

from ultralytics import YOLO
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, yolo_detect_world_positions

class MultiCameraBlockTask:
    def __init__(self):
        # ========== 硬件与模型 ==========
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.yolo = init_yolo(YOLO_MODEL_PATH)

        # 摄像头定义
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            "left": {"cam": init_camera(camera_id='left'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            "right": {"cam": init_camera(camera_id='right'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }

    # -------------------------
    # YOLO多摄像头检测
    # -------------------------
    def detect_blocks(self, show=False, eps=0.03):
        """
        多摄像头检测方块，并根据空间位置去重
        :param show: 是否显示每个相机画面
        :param eps: 聚类半径，单位 m
        :return: 去重后的 (cls_id, [x, y, z]) 列表
        """
        all_results = []
        for name, cam_info in self.cameras.items():
            cam = cam_info["cam"]
            pcd, pcd_color, depth_img, color_img = cam.get_pcd_texture_depth()

            if cam_info["type"] == "fixed":
                pcd_world = transform_points_by_homomat(cam_info["c2w"], pcd)
            else:
                pcd_world = self.rbt_s.transform_point_cloud_handeye(
                    cam_info["handeye"], pcd,
                    component_name='lft_arm' if name == 'left' else 'rgt_arm'
                )

            results = yolo_detect_world_positions(self.yolo, color_img, pcd_world, show=show)
            if results:
                all_results.extend(results)  # 保留 (cls_id, pos) 的结构

            if show:
                cv2.imshow(f"{name}_camera", color_img)
                cv2.waitKey(1)

        if show:
            cv2.destroyAllWindows()

        # 如果没有检测到任何物体
        if not all_results:
            return []

        # ---------------------------
        # 按 cls_id 分组并去重
        # ---------------------------
        deduped_results = []
        all_cls_ids = [r[0] for r in all_results]
        all_positions = np.array([r[1] for r in all_results])
        unique_cls = set(all_cls_ids)

        for cls in unique_cls:
            mask = [i for i, c in enumerate(all_cls_ids) if c == cls]
            cls_positions = all_positions[mask]

            # DBSCAN 聚类
            clustering = DBSCAN(eps=eps, min_samples=1).fit(cls_positions)
            labels = clustering.labels_

            for lbl in np.unique(labels):
                cluster_points = cls_positions[labels == lbl]
                centroid = cluster_points.mean(axis=0)
                deduped_results.append((cls, centroid.tolist()))

        return deduped_results

    # -------------------------
    # 生成抓取姿态
    # -------------------------
    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH):
            return
        print("🧩 生成抓取姿态中...")
        obj = mcm.CollisionModel(BLOCK_MODEL_PATH)
        obj.attach_to(base)
        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=20,
            min_dist_between_sampled_contact_points=0.03,
            contact_offset=0.01
        )
        grasps.save_to_disk(GRASP_PATH)
        print(f"✅ 保存抓取姿态，共 {len(grasps)} 个")

    # -------------------------
    # 执行轨迹动画
    # -------------------------
    def _execute_trajectory(self, arm, mot_data, base, show_sim=False):
        for jv, ev in zip(mot_data.jv_list, mot_data.ev_list):
            arm.move_j(jv, speed=10)
            arm.gripper_control(angle=0.07 if ev >= 0.09 else 0.0, effort=0)
            time.sleep(0.2)
        if not show_sim:
            return
        class AnimeData:
            def __init__(self, mot_data): self.mot_data = mot_data; self.counter = 0
        anime_data = AnimeData(mot_data)
        def update(anime_data, task):
            if anime_data.counter > 0:
                anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
            if anime_data.counter >= len(anime_data.mot_data):
                anime_data.counter = len(anime_data.mot_data) - 1
                return task.again
            mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
            mesh_model.attach_to(base)
            mesh_model.show_cdprim()
            if base.inputmgr.keymap.get('space', False):
                anime_data.counter += 1
            return task.again
        base.taskMgr.doMethodLater(0.01, update, "update", extraArgs=[anime_data], appendTask=True)
        print("🔹 按空格播放轨迹帧，关闭仿真窗口继续任务...")
        base.run()

    # -------------------------
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, start_pos, goal_pos, arm, robot, base, obstacles, show_sim=False):
        print(f"🤖 从 {start_pos} 抓取 → 放到 {goal_pos}")
        block = mcm.CollisionModel(BLOCK_MODEL_PATH)
        block.pos = np.array(start_pos,dtype=float)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH)

        mot_data = planner.gen_pick_and_place(
            obj_cmodel=block,
            end_jnt_values=robot.get_jnt_values(),
            grasp_collection=grasps,
            goal_pose_list=[(goal_pos, goal_rot)],
            pick_approach_direction=-rm.const.z_ax,
            place_approach_distance_list=[.05],
            place_depart_distance_list=[.05],
            pick_approach_distance=.05,
            pick_depart_distance=.05,
            obstacle_list=obstacles,
            use_rrt=True
        )
        if mot_data is None:
            print("⚠️ 轨迹规划失败！")
            return False

        self._execute_trajectory(arm, mot_data, base, show_sim=show_sim)
        return True

    # -------------------------
    # 机械臂选择
    # -------------------------
    def choose_arm(self, block_pos):
        _, pos = block_pos
        if pos[1] > -0.3:
            return self.left_arm, self.rbt_s.lft_arm
        else:
            return self.right_arm, self.rbt_s.rgt_arm
    # -------------------------
    # 主任务入口
    # -------------------------
    def run(self, show_camera=False, show_sim=False):
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)
        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
        ]
        [o.attach_to(base) for o in obstacles]

        self.create_grasps(base)
        self.left_arm.move_j([0]*6, speed=20)
        self.right_arm.move_j([0]*6, speed=20)

        blocks = self.detect_blocks(show=show_camera)
        if len(blocks) < 3:
            print(f"⚠️ 检测到 {len(blocks)} 个方块，不足三个！")
            return False

        for i, (block, target) in enumerate(zip(blocks, TARGET_POSITIONS)):
            cls_id, pos = block
            arm, robot = self.choose_arm(block)
            print(f"\n=== 第 {i+1} 个方块 ===")
            if not self.execute_pick_place(pos, target, arm, robot, base, obstacles, show_sim=show_sim):
                print(f"❌ 第 {i+1} 个方块堆叠失败")
                base.run()
                return False
            print(f"✅ 第 {i+1} 个方块堆叠成功")

        print("\n🎯 所有方块堆叠完成！")
        return True

# ==================================
# main
# ==================================
def main():
    task = MultiCameraBlockTask()
    try:
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)
        success = task.run(show_camera=False, show_sim=False)
        print("任务成功 ✅" if success else "任务失败 ❌")
    except KeyboardInterrupt:
        print("\n⚠️ 捕获到 Ctrl+C，机械臂回到全零位...")
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("⚠️ 出现异常，机械臂回到全零位...")
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)

if __name__ == '__main__':
    main()