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
from my_project.tiaozhanbei.empty_cup_place.constant import YOLO_MODEL_CUPS_PATH, CUP_MODEL_PATH, GRASP_PATH_CUPS,  \
    MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
import wrs.modeling.geometric_model as gm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, yolo_detect_world_positions
from my_project.tiaozhanbei.empty_cup_place.detect_mini import *

class MultiCameraCupTask:
    def __init__(self):
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.yolo = init_yolo(YOLO_MODEL_CUPS_PATH)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.visualize = False
        # # 摄像头定义
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            #"left": {"cam": init_camera(camera_id='left'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            # "right": {"cam": init_camera(camera_id='right'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }
        print(self.cameras)
        print("杯子任务初始化完毕")


    def process_gripper_data(self, data, threshold=0.05):
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
    
    # -------------------------00
    # YOLO多摄像头检测
    # -------------------------
    def detect_cups(self, show=False, eps=0.03):
        """
        使用多摄像头和 YOLO 检测杯子/杯垫并去重。
        返回: [(cls_id, [x, y, z]), ...] 世界坐标系下的质心列表
        """
        all_results = []

        for name, cam_info in self.cameras.items():
            cam = cam_info["cam"]

            try:
                # 获取彩色帧用于 YOLO
                depth_frame, color_frame = cam._current_frames()
                color_img = np.asanyarray(color_frame.get_data())

                # YOLO 检测
                detections = self.yolo(color_img, verbose=False)[0]
                if detections is None or len(detections.boxes) == 0:
                    continue

                for (x1, y1, x2, y2), cls_id, conf in zip(
                        detections.boxes.xyxy.cpu().numpy(),
                        detections.boxes.cls.cpu().numpy(),
                        detections.boxes.conf.cpu().numpy()):
                    if conf < 0.3:
                        continue

                    # 框内点云（深度相机坐标系）
                    points_3d = cam.points_in_color_bbox((x1, y1, x2, y2))
                    if len(points_3d) == 0:
                        continue

                    # 转世界坐标
                    if cam_info["type"] == "fixed":
                        points_world = transform_points_by_homomat(cam_info["c2w"], points_3d)
                    else:
                        points_world = self.rbt_s.transform_point_cloud_handeye(
                            cam_info["handeye"], points_3d,
                            component_name='lft_arm' if name == 'left' else 'rgt_arm'
                        )

                    # 质心
                    centroid = points_world.mean(axis=0)
                    all_results.append((int(cls_id), centroid.tolist()))

                # 可视化彩色图像
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

        # ---------------------------
        # 按 cls_id 分组并 DBSCAN 去重
        # ---------------------------
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
                deduped_results.append((cls, centroid.tolist())) 

        return deduped_results
    
    def crop_pointcloud_world(self, pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
        """
        在世界坐标系下裁剪点云到指定范围
        
        Args:
            pcd_world: 世界坐标系下的点云数据 (N, 3)
            x_range: X轴范围 (min, max)，默认(0, 0.6)
            y_range: Y轴范围 (min, max)，默认(0, -0.6) 
            z_range: Z轴范围 (min, max)，默认(0.07, 0.08)
            
        Returns:
            tuple: (裁剪后的点云数据, 原始点云数量, 裁剪后点云数量)
        """
        if pcd_world is None or len(pcd_world) == 0:
            print("输入点云为空")
            return None, 0, 0
            
        # 记录原始点云数量
        original_count = len(pcd_world)
        
        # 提取坐标
        x = pcd_world[:, 0]
        y = pcd_world[:, 1] 
        z = pcd_world[:, 2]
        
        # 创建裁剪掩码
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        y_mask = (y >= y_range[1]) & (y <= y_range[0])  # 注意Y轴范围是(0, -0.6)，所以是y >= -0.6 and y <= 0
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        
        # 组合所有掩码
        combined_mask = x_mask & y_mask & z_mask
        
        # 应用掩码裁剪点云
        cropped_pcd = pcd_world[combined_mask]
        cropped_count = len(cropped_pcd)
        
        # 打印统计信息
        print(f"点云裁剪统计:")
        print(f"  原始点云数量: {original_count}")
        print(f"  裁剪后点云数量: {cropped_count}")
        print(f"  裁剪范围: X[{x_range[0]}, {x_range[1]}], Y[{y_range[1]}, {y_range[0]}], Z[{z_range[0]}, {z_range[1]}]")
        print(f"  保留比例: {cropped_count/original_count*100:.2f}%")
        
        if cropped_count > 0:
            # 打印裁剪后点云的坐标范围
            print(f"  裁剪后点云范围:")
            print(f"    X: [{cropped_pcd[:, 0].min():.4f}, {cropped_pcd[:, 0].max():.4f}]")
            print(f"    Y: [{cropped_pcd[:, 1].min():.4f}, {cropped_pcd[:, 1].max():.4f}]")
            print(f"    Z: [{cropped_pcd[:, 2].min():.4f}, {cropped_pcd[:, 2].max():.4f}]")
        else:
            print("  警告: 裁剪后没有剩余点云")
            
        return cropped_pcd, original_count, cropped_count
    
    def align_pcd(self, pcd):
        """
        将点云从相机坐标系转换到世界坐标系
        
        Args:
            pcd: 相机坐标系下的点云数据 (N, 3)
            
        Returns:
            np.ndarray: 世界坐标系下的点云数据 (N, 3)
        """
        c2w_mat =  MIDDLE_CAM_C2W # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
    def process_pointcloud(self, pcd_camera):

        """
        完整的点云处理流程：相机坐标系 -> 世界坐标系 -> 裁剪
        
        Args:
            pcd_camera: 相机坐标系下的点云数据 (N, 3)
            
        Returns:
            tuple: (裁剪后的点云数据, 原始点云数量, 裁剪后点云数量)
        """
        # 步骤1: 转换到世界坐标系
        pcd_world = self.align_pcd(pcd_camera)
        
        # 步骤2: 裁剪到指定范围
        cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
            pcd_world, 
            x_range=(0, 0.6), 
            y_range=(0, -0.6), 
            z_range=(0.07, 0.08)
        )
        
        return cropped_pcd, original_count, cropped_count
    
    def print_cropped_pointcloud_with_center(self, cropped_pcd):
        """
        打印裁剪后的点云，按高度排序，并计算中心点

        Args:
            cropped_pcd: 裁剪后的点云数据 (N, 3)
            camera_role: 相机角色名称

        return:
            x,y,z 杯口中心点
        """
        if cropped_pcd is None or len(cropped_pcd) == 0:
            print("没有裁剪后的点云数据")
            return
            
        print("\n===  裁剪后点云详细信息 ===")
        print(f"点云数量: {len(cropped_pcd)}")

        # 按高度（Z坐标）降序排序，优先显示高度高的点
        sorted_indices = np.argsort(cropped_pcd[:, 2])[::-1]  # 降序排序
        sorted_pcd = cropped_pcd[sorted_indices]

        # 打印前20个最高点（避免输出过多）
        print(f"\n前20个最高点（按高度降序）:")
        print("序号    X坐标(m)    Y坐标(m)    Z坐标(m)    高度(cm)")
        print("-" * 60)
        for i in range(min(20, len(sorted_pcd))):
            point = sorted_pcd[i]
            print(f"{i+1:3d}    {point[0]:8.4f}    {point[1]:8.4f}    {point[2]:8.4f}    {point[2]*100:6.2f}")

        if len(sorted_pcd) > 20:
            print(f"... (还有 {len(sorted_pcd) - 20} 个点未显示)")

        # 计算中心点
        center_point = np.mean(cropped_pcd, axis=0)
        print(f"\n点云中心点:")
        print(f"  X: {center_point[0]:.4f} m")
        print(f"  Y: {center_point[1]:.4f} m") 
        print(f"  Z: {center_point[2]:.4f} m ({center_point[2]*100:.2f} cm)")

        # 计算点云范围
        min_coords = np.min(cropped_pcd, axis=0)
        max_coords = np.max(cropped_pcd, axis=0)
        print(f"\n点云范围:")
        print(f"  X: [{min_coords[0]:.4f}, {max_coords[0]:.4f}] m (跨度: {max_coords[0]-min_coords[0]:.4f} m)")
        print(f"  Y: [{min_coords[1]:.4f}, {max_coords[1]:.4f}] m (跨度: {max_coords[1]-min_coords[1]:.4f} m)")
        print(f"  Z: [{min_coords[2]:.4f}, {max_coords[2]:.4f}] m (跨度: {max_coords[2]-min_coords[2]:.4f} m)")

        # 计算高度统计
        heights = cropped_pcd[:, 2]
        print(f"\n高度统计:")
        print(f"  平均高度: {np.mean(heights):.4f} m ({np.mean(heights)*100:.2f} cm)")
        print(f"  最高点: {np.max(heights):.4f} m ({np.max(heights)*100:.2f} cm)")
        print(f"  最低点: {np.min(heights):.4f} m ({np.min(heights)*100:.2f} cm)")
        print(f"  高度标准差: {np.std(heights):.4f} m ({np.std(heights)*100:.2f} cm)")

        print("=" * 50)

        return center_point[0],center_point[1],center_point[2]
    
    def detect_cup_use_cloud_points(self):
        for name, cam_info in self.cameras.items():
            cam = cam_info["cam"]

            while True:
                # 获取相机数据
                pcd, pcd_color, depth_img, color_img = cam.get_pcd_texture_depth()
                
                if pcd is not None:
                    # 处理点云：相机坐标系 -> 世界坐标系 -> 裁剪
                    cropped_pcd, original_count, cropped_count = self.process_pointcloud(pcd)
                    
                    # if cropped_pcd is not None and len(cropped_pcd) > 0:
                    #     #print(f"[{role}相机] 处理完成: {len(cropped_pcd)} 个点")
                    # else:
                    #     #print(f"[{role}相机] 没有符合条件的点云")

                    # 打印裁剪后的点云（按高度排序）并计算中心点
                    if cropped_pcd is not None and len(cropped_pcd) > 0:
                        cup_x,cup_y,cup_z = self.print_cropped_pointcloud_with_center(cropped_pcd)
                        
                        return cup_x,cup_y,cup_z
                                        
                    else:
                        print("获取点云失败")

    def process_gripper_data(self, data, threshold=0.05):
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

    # -------------------------
    # 生成抓取姿态
    # -------------------------
    def create_grasps(self):
        if os.path.exists(GRASP_PATH_CUPS):
            return
        print("☕ 生成抓取姿态中...")
        obj = mcm.CollisionModel(CUP_MODEL_PATH)
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
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, pick_pos, place_pos, arm:PiperArmController, robot, obstacles):
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
            pick_approach_direction= -rm.const.z_ax,
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

        # 先执行机械臂动作
        # for jv, ev in zip(mot_data.jv_list, mot_data.ev_list):
        #     arm.move_jntspace_path(jv, )
        #     arm.gripper_control(angle=0.04 if ev >= 0.09 else 0.0, effort=0)
        #     time.sleep(0.2)

        jv = mot_data.jv_list
        ev = mot_data.ev_list

        binary_arr, change_indices = self.process_gripper_data(ev)
        print(change_indices)
        approach_path,pick_path,depart_path = self.split_trajectory_by_gripper(jv,change_indices)
        arm.open_gripper(width=0.03)
        arm.move_j(jv[0],speed=20,block=True)

        time.sleep(0.1)
        arm.move_jntspace_path(approach_path,speed=20)
        time.sleep(0.1)
        arm.close_gripper()
        time.sleep(0.1)
        arm.move_jntspace_path(pick_path,speed=20)
        time.sleep(0.1)
        arm.open_gripper(width = 0.03)
        time.sleep(0.1) 
        arm.move_jntspace_path(depart_path,speed=20)

        # arm.move_j(mot_data.jv_list[0],speed=10)
        # arm.move_jntspace_path(mot_data.jv_list)

        return mot_data

    # -------------------------
    # 机械臂选择
    # -------------------------
    def choose_arm(self, pos):
        _, y = pos
        if y[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    
    # -------------------------
    # 主任务入口
    # -------------------------
    def run(self, show_camera=False):
        if self.visualize:
            base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
            mgm.gen_frame().attach_to(base)

        # 障碍物
        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.23, 0.07])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.375, 0.07]))
        ]
        if self.visualize:
            [o.attach_to(base) for o in obstacles]

        if self.visualize:
            self.create_grasps()
        else:
            self.create_grasps()

        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)
        time_start = time.time()
        objects = self.detect_cups(show=show_camera)
        if not objects:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False
        
        #返回杯子坐标
        cup_x,cup_y,cup_z = self.detect_cup_use_cloud_points()
        # 找到杯子和目标位置
        #pick_obj = next((pos for cls_id, pos in objects if cls_id == 1), None)
        pick_obj = [cup_x,cup_y,cup_z]
        print(f"pick_obj:{pick_obj}")
        place_obj = next((pos for cls_id, pos in objects if cls_id == 0), None)
        if pick_obj is None or place_obj is None:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False
        # 将 pick_obj 的 z 坐标改为 0
        pick_obj_mod = pick_obj.copy()
        pick_obj_mod[2] = 0.0

        # 执行抓放，存储轨迹用于统一仿真
        arm, robot = self.choose_arm((1, pick_obj))
        print(f"\n☕ 抓取杯子 {pick_obj} → 放置 {place_obj}")
        mot_data = self.execute_pick_place(pick_obj_mod, place_obj, arm, robot, obstacles)
        if mot_data is None:
            print("❌ 抓取失败")
            return False
        end_time = time.time()

        print(f"'推理用时{end_time-time_start}'")
        # -------------------------
        # 统一仿真回放
        # -------------------------
        if self.visualize:
            print("\n🎬 开始统一仿真回放...")
            for mesh in mot_data.mesh_list:
                mesh.attach_to(base)
                mesh.show_cdprim()
            base.run()

        print("✅ 抓放杯子完成！")
        return True


# ==================================
# main
# ==================================
def main():
    task = MultiCameraCupTask()
    try:
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)
        success = task.run(show_camera=False)
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
