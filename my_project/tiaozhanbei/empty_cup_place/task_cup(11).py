#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/25
# @Author : ZhangXi

import os
import time
import cv2
import numpy as np
import time
from sklearn.cluster import DBSCAN
import wrs.basis.robot_math as rm
# --- ICP/点云相关引用 ---
from wrs.vision.depth_camera.util_functions import registration_ptpt
import wrs.modeling.geometric_model as gm
# ------------------------
from my_project.tiaozhanbei.empty_cup_place.constant import YOLO_MODEL_CUPS_PATH, CUP_MODEL_PATH, GRASP_PATH_CUPS, \
    MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, \
    yolo_detect_world_positions


class MultiCameraCupTask:
    def __init__(self, resources=None):
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.yolo = init_yolo(YOLO_MODEL_CUPS_PATH)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()

        # 摄像头定义 (仅使用middle)
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
        }
        print(self.cameras)
        print("杯子任务初始化完毕")

    # -------------------------
    # (process_gripper_data 和 split_trajectory_by_gripper 保持不变)
    # -------------------------
    def process_gripper_data(self, data, threshold=0.05):
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

    # ------------------------------------------------------------------
    # 修改 detect_cups (添加点云返回)
    # ------------------------------------------------------------------
    def detect_cups(self, show=False, eps=0.03):
        """
        使用多摄像头和 YOLO 检测杯子/杯垫并去重。
        返回: [(cls_id, [x, y, z], pcd), ...] 世界坐标系下的质心和点云列表
        """
        all_results = []
        for name, cam_info in self.cameras.items():
            cam = cam_info["cam"]
            try:
                depth_frame, color_frame = cam._current_frames()
                color_img = np.asanyarray(color_frame.get_data())
                detections = self.yolo(color_img, verbose=False)[0]
                if detections is None or len(detections.boxes) == 0:
                    continue

                for (x1, y1, x2, y2), cls_id, conf in zip(
                        detections.boxes.xyxy.cpu().numpy(),
                        detections.boxes.cls.cpu().numpy(),
                        detections.boxes.conf.cpu().numpy()):
                    if conf < 0.3:
                        continue

                    # 保持原有的 bbox 扩展逻辑
                    margin_y = 50
                    x1_new = max(0, x1 - margin_y)
                    x2_new = min(color_img.shape[1], x2 + margin_y)
                    y1_new = max(0, y1 - margin_y)
                    y2_new = min(color_img.shape[0], y2 + margin_y)

                    points_3d = cam.points_in_color_bbox((x1_new, y1_new, x2_new, y2_new))
                    if len(points_3d) == 0:
                        continue

                    if cam_info["type"] == "fixed":
                        points_world = transform_points_by_homomat(cam_info["c2w"], points_3d)
                    else:
                        points_world = self.rbt_s.transform_point_cloud_handeye(
                            cam_info["handeye"], points_3d,
                            component_name='lft_arm' if name == 'left' else 'rgt_arm'
                        )

                    centroid = points_world.mean(axis=0)
                    all_results.append((int(cls_id), centroid.tolist(), points_world))

                if show:
                    cv2.imshow(f"{name}_camera", color_img)
                    cv2.waitKey(1)
            except Exception as e:
                print(f"⚠️ {name} 摄像头检测失败: {e}")
                continue

        if show:
            cv2.destroyAllWindows()
        if not all_results:
            return []

        # 按 cls_id 分组并 DBSCAN 去重 (聚合点云)
        deduped_results = []
        all_cls_ids = [r[0] for r in all_results]
        all_positions = np.array([r[1] for r in all_results])
        unique_cls = set(all_cls_ids)

        for cls in unique_cls:
            mask = [i for i, c in enumerate(all_cls_ids) if c == cls]
            if not mask:
                continue

            cls_positions = all_positions[mask]
            clustering = DBSCAN(eps=eps, min_samples=1).fit(cls_positions)
            labels = clustering.labels_

            for lbl in np.unique(labels):
                cluster_mask = (labels == lbl)
                cluster_centroids = cls_positions[cluster_mask]
                centroid = cluster_centroids.mean(axis=0)

                original_indices_mask = np.where(cluster_mask)[0]
                original_indices = [mask[i] for i in original_indices_mask]
                all_cluster_pcds = [all_results[i][2] for i in original_indices]
                aggregated_pcd = np.concatenate(all_cluster_pcds, axis=0)

                deduped_results.append((cls, centroid.tolist(), aggregated_pcd))

        return deduped_results

    # ------------------------------------------------------------------
    # align_cup_pose (ICP 函数，格式遵循 align_shoe_pose / align_block_pose)
    # ------------------------------------------------------------------
    def align_cup_pose(self, base, cup_model_path, real_pcd, visualize=False):
        """点云配准以修正杯子姿态，同时可选可视化"""
        # 1. 加载模型并采样点云
        cup = gm.GeometricModel(cup_model_path)
        model_pcd = cup.sample_surface(radius=0.001, n_samples=5000)
        model_pcd = model_pcd[model_pcd[:, 2] > 0.005]  # 过滤模型底部的点

        # 2. 可视化原始点云
        if visualize:
            gm.gen_pointcloud(model_pcd, rgba=np.array([0, 0, 1, 0.5])).attach_to(base)  # 蓝色: 模型
            gm.gen_pointcloud(real_pcd, rgba=np.array([0, 1, 0, 0.5])).attach_to(base)  # 绿色: 实际点云

        print("☕ 开始ICP点云配准以匹配杯子方向...")
        # 3. 执行 ICP
        icp_result = registration_ptpt(model_pcd, real_pcd, downsampling_voxelsize=0.005)
        transformation = icp_result[2]

        # 4. 更新模型姿态并附加到世界
        cup.homomat = transformation
        cup.attach_to(base)

        # 5. 可视化对齐后的点云
        aligned_pcd = rm.transform_points_by_homomat(transformation, model_pcd.copy())
        if visualize:
            gm.gen_pointcloud(aligned_pcd, rgba=np.array([1, 0, 0, 0.6])).attach_to(base)  # 红色: 配准结果

        print("✅ ICP配准完成")
        return transformation

    # -------------------------
    # 生成抓取姿态 (修改: 增加 base 参数)
    # -------------------------
    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH_CUPS):
            return
        print("☕ 生成抓取姿态中...")
        obj = mcm.CollisionModel(CUP_MODEL_PATH)
        obj.attach_to(base)  # 附加到 base

        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=10,
            min_dist_between_sampled_contact_points=0.01,
            contact_offset=0.01
        )

        grasps.save_to_disk(GRASP_PATH_CUPS)
        print(f"✅ 保存抓取姿态，共 {len(grasps)} 个")

    # -------------------------
    # 执行 pick & place (保持不变)
    # -------------------------
    def execute_pick_place(self, pick_pos, place_pos, arm: PiperArmController, robot, obstacles):
        block = mcm.CollisionModel(CUP_MODEL_PATH)
        block.pos = np.array(pick_pos, dtype=float)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_CUPS)
        print(len(grasps))
        mot_data = planner.gen_pick_and_place(
            obj_cmodel=block,
            end_jnt_values=robot.get_jnt_values(),
            grasp_collection=grasps,
            goal_pose_list=[(place_pos, goal_rot)],
            pick_approach_direction=-rm.const.z_ax,
            place_approach_distance_list=[.05],
            place_depart_distance_list=[.07],
            pick_approach_distance=.03,
            pick_depart_distance=.05,
            obstacle_list=obstacles,
            use_rrt=False
        )
        if mot_data is None:
            print("⚠️ 轨迹规划失败！")
            return None

        jv = mot_data.jv_list
        ev = mot_data.ev_list

        binary_arr, change_indices = self.process_gripper_data(ev)
        print(change_indices)
        approach_path, pick_path, depart_path = self.split_trajectory_by_gripper(jv, change_indices)
        arm.open_gripper(width=0.03)
        arm.move_j(jv[0], speed=20, block=True)

        time.sleep(0.1)
        arm.move_jntspace_path(approach_path, speed=20)
        time.sleep(0.1)
        arm.close_gripper()
        time.sleep(0.1)
        arm.move_jntspace_path(pick_path, speed=20)
        time.sleep(0.1)
        arm.open_gripper(width=0.03)
        time.sleep(0.1)
        arm.move_jntspace_path(depart_path, speed=20)

        return mot_data

    # -------------------------
    # 机械臂选择 (保持不变)
    # -------------------------
    def choose_arm(self, pos):
        # pos 是 (cls_id, [x, y, z])
        _, y = pos
        if y[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    # -------------------------
    # 主任务入口 (集成ICP流程 和 统一 base)
    # -------------------------
    def run(self, show_camera=False):
        # 统一初始化 base
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)

        # 障碍物 (保持不变)
        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.23, 0.07])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.375, 0.07]))
        ]
        [o.attach_to(base) for o in obstacles]

        self.create_grasps(base)

        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)
        time_start = time.time()

        objects = self.detect_cups(show=show_camera)
        if not objects:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False

        # 找到杯子(1)和杯垫(0)
        # pick_obj_data 是 (cls_id, pos, pcd)
        pick_obj_data = next(((cls_id, pos, pcd) for cls_id, pos, pcd in objects if cls_id == 1), None)
        place_obj_pos = next((pos for cls_id, pos, _ in objects if cls_id == 0), None)

        if pick_obj_data is None or place_obj_pos is None:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False

        cls_id, pick_obj_pos, pick_obj_pcd = pick_obj_data

        # --- 修改: 模仿 task_shoes.py 对真实点云进行过滤 ---
        # 过滤点云, 移除桌面和边界外的点
        cup_pcd_real = pick_obj_pcd[
            (pick_obj_pcd[:, 2] > 0.008) & (pick_obj_pcd[:, 2] < 0.08) &  # Z轴高度过滤
            (pick_obj_pcd[:, 0] < 0.6) &  # X轴过滤 (小于 0.6m)
            (pick_obj_pcd[:, 1] < -0.05) & (pick_obj_pcd[:, 1] > -0.6)  # Y轴过滤 (在 -0.05m 到 -0.6m 之间)
            ]

        # --- ICP 配准流程 ---
        if len(cup_pcd_real) > 100:
            # 调用 align_cup_pose，并传入 base 和 show_camera 参数
            self.align_cup_pose(base, CUP_MODEL_PATH, cup_pcd_real, visualize=True)
        else:
            print("⚠️ 杯子点云过滤后数量不足, 跳过ICP。")
        # -----------------------------------------------

        # --- 保持原有抓取逻辑: 使用质心, 且 z=0 ---
        # pick_obj_mod 是用于轨迹规划的抓取目标位置
        pick_obj_mod = pick_obj_pos.copy()
        pick_obj_mod[2] = 0.015
        # ----------------- -----------------------
        
        # 执行抓放
        arm, robot = self.choose_arm((cls_id, pick_obj_pos))  # 使用原始质心判断用手
        print(f"\n☕ 抓取杯子 {pick_obj_pos} → 放置 {place_obj_pos}")
        mot_data = self.execute_pick_place(pick_obj_mod, place_obj_pos, arm, robot, obstacles)

        if mot_data is None:
            print("❌ 抓取失败")
            return False
        end_time = time.time()

        print(f"'推理用时{end_time - time_start}'")

        base.run()
        print("✅ 抓放杯子完成！")
        return True


# ==================================
# main
# ==================================
def main():
    task = MultiCameraCupTask()
    try:
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)
        # 传入 show_camera=True 可以看到 ICP 和抓取仿真
        success = task.run(show_camera=False)
        print("任务成功 ✅" if success else "任务失败 ❌")
    except KeyboardInterrupt:
        print("\n⚠️ 捕获到 Ctrl+C，机械臂回到全零位...")
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("⚠️ 出现异常，机械臂回到全零位...")
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)


if __name__ == '__main__':
    main()