#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/27
# @Author : ZhangXi

import os
import pickle
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
import wrs.basis.robot_math as rm
from wrs.vision.depth_camera.util_functions import registration_ptpt
import wrs.modeling.geometric_model as gm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, yolo_detect_world_positions
from my_project.tiaozhanbei.place_shoe.constant import YOLO_MODEL_SHOES_PATH, SHOE_MODEL_PATH, GRASP_PATH_SHOES, \
    MIDDLE_CAM_C2W


def align_z_to_up(R):
    """
    Adjusts a rotation matrix so that its z-axis aligns with (0, 0, 1),
    while preserving x/y directions as much as possible.
    """
    R = np.array(R, dtype=np.float64)
    assert R.shape == (3, 3)

    # Force the z-axis to (0, 0, 1)
    z_new = np.array([0, 0, 1], dtype=np.float64)

    # Project original x-axis onto the plane orthogonal to new z
    x_old = R[:, 0]
    x_new = x_old - np.dot(x_old, z_new) * z_new
    x_new /= np.linalg.norm(x_new) + 1e-9

    # Recompute y-axis using cross product to ensure orthogonality
    y_new = np.cross(z_new, x_new)
    y_new /= np.linalg.norm(y_new) + 1e-9

    R_new = np.column_stack((x_new, y_new, z_new))
    return R_new

class MultiCameraShoeTask:
    def __init__(self):
        # ===== 硬件与模型 =====
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.yolo = init_yolo(YOLO_MODEL_SHOES_PATH)

        # 摄像头定义
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            # "left": {"cam": init_camera(camera_id='243322074546'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            # "right": {"cam": init_camera(camera_id='243322071033'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }

    def detect_shoes(self, show=False, eps=0.03):
        """使用多摄像头和YOLO检测鞋子(带点云)和垫子(仅中心点)"""
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

                    # 提取该检测框中的点云
                    points_3d = cam.points_in_color_bbox((x1, y1, x2, y2))
                    if len(points_3d) == 0:
                        continue

                    # 转换到世界坐标
                    if cam_info["type"] == "fixed":
                        points_world = transform_points_by_homomat(cam_info["c2w"], points_3d)
                    else:
                        points_world = self.rbt_s.transform_point_cloud_handeye(
                            cam_info["handeye"], points_3d,
                            component_name='lft_arm' if name == 'left' else 'rgt_arm'
                        )

                    centroid = points_world.mean(axis=0)
                    # 保存 (类别, 质心, 点云)
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

        # DBSCAN 去重（合并多相机结果）
        deduped_results = []
        all_cls_ids = [r[0] for r in all_results]
        all_positions = np.array([r[1] for r in all_results])
        unique_cls = set(all_cls_ids)

        for cls in unique_cls:
            mask = [i for i, c in enumerate(all_cls_ids) if c == cls]
            cls_positions = all_positions[mask]
            clustering = DBSCAN(eps=eps, min_samples=1).fit(cls_positions)
            labels = clustering.labels_
            for lbl in np.unique(labels):
                cluster_points = cls_positions[labels == lbl]
                centroid = cluster_points.mean(axis=0)
                if cls == 0:
                    # 鞋子 → 返回点云用于ICPTrue
                    deduped_results.append((cls, centroid.tolist(), all_results[mask[0]][2]))
                else:
                    # 垫子 → 只返回质心，无需ICP
                    deduped_results.append((cls, centroid.tolist(), None))

        return deduped_results

    
    def process_gripper_data(self, data, threshold=0.1029):
        """
        处理夹爪数据，将大的数改为1，小的数改为0，并返回变化点索引
        
        参数:
            data: 输入数组，包含夹爪宽度数据
            threshold: 判断大小的阈值，默认0.05
            
        返回:
            tuple: (处理后的二进制数组, 变化点索引列表)
        """
        # 将数据转换为numpy数组
        arr = np.array(data)
        
        # 创建二进制数组：大于阈值设为1，否则设为0
        binary_arr = (arr > threshold).astype(int)
        
        # 找到数值变化的索引
        change_indices = np.where(np.diff(binary_arr) != 0)[0] + 1
        
        return binary_arr.tolist(), change_indices.tolist()
    
    def split_trajectory_by_gripper(self, jv, change_indices, threshold=0.05):
        """
        根据夹爪数据的变化点将关节速度轨迹分割为三段
        
        参数:
            jv: 关节速度数组，形状为(n,6)的二维数组
            gripper_data: 夹爪宽度数据，一维数组
            threshold: 判断夹爪开合的阈值
            
        返回:
            dict: 包含三段轨迹的字典 {
                'stage1': 第一阶段轨迹,
                'stage2': 第二阶段轨迹,
                'stage3': 第三阶段轨迹,
                'change_points': 变化点索引
            }
        """
        # 确保有两个变化点
        if len(change_indices) != 2:
            raise ValueError(f"期望2个变化点,但找到{len(change_indices)}个")
        
        # 获取两个变化点
        cp1, cp2 = change_indices
        
        # 分割轨迹
        stage1 = jv[:cp1]      # 第一阶段：从开始到第一个变化点
        stage2 = jv[cp1:cp2]   # 第二阶段：第一个变化点到第二个变化点
        stage3 = jv[cp2:]      # 第三阶段：第二个变化点到结束
        
        return stage1,stage2,stage3
    
    def align_shoe_pose(self, shoe_model_path, real_pcd, visualize=False):
        """点云配准以修正鞋子姿态，同时可选可视化"""
        shoe = mgm.GeometricModel(shoe_model_path)
        
        shoe_pcd = shoe.sample_surface(radius=0.001, n_samples=8000)
        shoe_pcd = shoe_pcd[shoe_pcd[:, 2] > 0.019]

        

        print("🦶 开始ICP点云配准以匹配鞋子方向...")
        icp_result = registration_ptpt(shoe_pcd, real_pcd, downsampling_voxelsize=0.007)
        transformation = icp_result[2]
        shoe.homomat = transformation
        

        # 显示配准后的模型点云
        aligned_pcd = rm.transform_points_by_homomat(transformation, shoe_pcd.copy())
        

        print("✅ ICP配准完成")
        return transformation

    def create_grasps(self):
        """生成抓取姿态"""
        if os.path.exists(GRASP_PATH_SHOES):
            return
        obj = mcm.CollisionModel(SHOE_MODEL_PATH)
        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(160),
            rotation_interval=rm.radians(30),
            max_samples=100,
            min_dist_between_sampled_contact_points=0.01,
            contact_offset=0.01
        )
        grasps.save_to_disk(GRASP_PATH_SHOES)
        print(f"保存鞋子抓取姿态，共 {len(grasps)} 个")

    def execute_pick_place(self, pick_pos, place_pos, place_rot, arm : PiperArmController, robot, obstacles):
        block = mcm.CollisionModel(SHOE_MODEL_PATH)
        block.homomat = np.asarray(pick_pos, dtype=float)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_SHOES)

        mot_data = planner.gen_pick_and_place(
            obj_cmodel=block,
            end_jnt_values=robot.get_jnt_values(),
            grasp_collection=grasps,
            goal_pose_list=[(place_pos, place_rot)],
            pick_approach_direction=-rm.const.z_ax,
            place_approach_distance_list=[.05],
            place_depart_distance_list=[.05],
            pick_approach_distance=.05,
            pick_depart_distance=.05,
            obstacle_list=obstacles,
            use_rrt=False
        )
        if mot_data is None:
            print("⚠️ 轨迹规划失败！")
            return None

        jv = mot_data.jv_list
        ev = mot_data.ev_list
        print(ev)
        binary_arr, change_indices = self.process_gripper_data(ev)
        print(change_indices)
        approach_path,pick_path,depart_path = self.split_trajectory_by_gripper(jv,change_indices)
        arm.open_gripper(width=0.08)
        arm.move_j(jv[0],speed=10,block=True)

        time.sleep(0.1)
        arm.move_jntspace_path(approach_path,speed=10)
        time.sleep(0.1)
        arm.close_gripper()
        time.sleep(0.1)
        arm.move_jntspace_path(pick_path,speed=10)
        time.sleep(0.1)
        arm.open_gripper(width = 0.08)
        time.sleep(0.1) 
        arm.move_jntspace_path(depart_path,speed=10)

        return mot_data

    def choose_arm(self, pos):
        _, y = pos
        if y[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    def run(self, show_camera=False):
        obstacles = []
        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)

        objects = self.detect_shoes(show=show_camera)
        if not objects:
            print("⚠️ 没检测到鞋子或垫子！")
            return False

        # 🦶 鞋子（要ICP）
        pick_obj = next((pos for cls_id, pos, _ in objects if cls_id == 0), None)
        shoe_pcd_real = next((pcd for cls_id, _, pcd in objects if cls_id == 0), None)

        shoe_pcd_real = shoe_pcd_real[
            (shoe_pcd_real[:, 2] > 0.01) & (shoe_pcd_real[:, 2] < 0.09) & (shoe_pcd_real[:, 0] < 0.6)
            & (shoe_pcd_real[:, 1] < -0.05) & (shoe_pcd_real[:, 1] > -0.6)]

        # 🟩 垫子（不ICP）
        place_obj = next((pos for cls_id, pos, _ in objects if cls_id == 1), None)

        if pick_obj is None or place_obj is None or shoe_pcd_real is None:
            print("⚠️ 检测结果不完整！")
            if pick_obj is None:
                print(" -> 错误：未检测到鞋子质心 (ID 0)。")
            if shoe_pcd_real is None:
                print(" -> 错误：未获取到鞋子点云 (ID 0)，无法进行 ICP。")
            if place_obj is None:
                print(" -> 错误：未检测到垫子质心 (ID 1)。")
            return False

        # 对鞋子执行ICP配准
        pick_obj_pose = self.align_shoe_pose(SHOE_MODEL_PATH, shoe_pcd_real,visualize=True) 
        pick_obj_pose = pick_obj_pose.copy()
        pick_obj_pose[:3,:3] = align_z_to_up(pick_obj_pose[:3,:3])
        # 抓取并放置
        arm, robot = self.choose_arm((0, pick_obj))

        place_obj[2] = 0.02
        
        print(f"\n 抓取鞋子 {pick_obj} → 放置 {place_obj}")
        print(pick_obj_pose)
        place_obj_rot = rm.rotmat_from_euler(0, 0, 0)
        print(place_obj_rot)
        mot_data = self.execute_pick_place(pick_obj_pose, place_obj, place_obj_rot, arm, robot, obstacles)
        
        
        if mot_data is None:
            print("抓取失败")
            return False
        print("✅ 放鞋子完成！")
        return True


def main():
    task = MultiCameraShoeTask()
    try:
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)
        success = task.run(show_camera=False)
        print("任务成功 ✅" if success else "任务失败 ❌")
    except KeyboardInterrupt:
        print("\n⚠️ 捕获到 Ctrl+C,机械臂回零...")
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)


if __name__ == '__main__':
    main()