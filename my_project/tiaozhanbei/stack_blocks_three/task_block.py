#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/18 9:34
# @Author : ZhangXi

import os
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import wrs.basis.robot_math as rm
from my_project.tiaozhanbei.stack_blocks_three.constant import YOLO_MODEL_BLOCKS_PATH, BLOCK_MODEL_PATH, GRASP_PATH_BLOCKS, \
    TARGET_POSITIONS, MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
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
        self.yolo = init_yolo(YOLO_MODEL_BLOCKS_PATH)

        # 摄像头定义
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            #"left": {"cam": init_camera(camera_id='left'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            #"right": {"cam": init_camera(camera_id='right'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }


    # -------------------------
    # 点云裁切 + 颜色聚类检测
    # -------------------------
    def detect_blocks(self, show=False, eps=0.03):
        """
        使用 YOLO + 深度点云检测方块，返回世界坐标系下的质心，按颜色返回
        """
        all_results = []

        for name, cam_info in self.cameras.items():
            cam = cam_info["cam"]
            try:
                # 获取彩色帧用于 YOLO
                depth_frame, color_frame = cam._current_frames()
                color_img = np.asanyarray(color_frame.get_data())
                depth_img = np.asanyarray(depth_frame.get_data())

                # YOLO 检测
                detections = self.yolo(color_img, verbose=False)[0]
                if detections is None or len(detections.boxes) == 0:
                    continue

                for (x1, y1, x2, y2), cls_id, conf in zip(
                        detections.boxes.xyxy.cpu().numpy(),
                        detections.boxes.cls.cpu().numpy(),
                        detections.boxes.conf.cpu().numpy()):
                    if conf < 0.1:
                        continue

                    # 框内点云（深度相机坐标系）
                    points_3d = cam.points_in_color_bbox((x1, y1, x2, y2))

                    # 如果点云为空，用框中心像素深度代替
                    if len(points_3d) == 0:
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        z = depth_img[cy, cx]
                        if z == 0:
                            continue  # depth 无效就跳过
                        # 将像素坐标转换到相机坐标系
                        fx, fy, cx_cam, cy_cam = cam.get_intrinsics()  # 你需要保证相机有这个方法
                        x_cam = (cx - cx_cam) * z / fx
                        y_cam = (cy - cy_cam) * z / fy
                        points_3d = np.array([[x_cam, y_cam, z]])

                    # 转世界坐标
                    if cam_info["type"] == "fixed":
                        points_world = transform_points_by_homomat(cam_info["c2w"], points_3d)
                    else:
                        points_world = self.rbt_s.transform_point_cloud_handeye(
                            cam_info["handeye"], points_3d,
                            component_name='lft_arm' if name == 'left' else 'rgt_arm'
                        )

                            # ---------------------------
                    # 筛选有效点（移除无效区域）
                    # ---------------------------
                    valid_mask = (
                        (points_world[:, 0] <= 0.6) &   # X ≤ 0.6
                        (points_world[:, 2] >= 0) &      # Z ≥ 0
                        (points_world[:, 1] <= 0.1) &    # Y ≤ 0.1
                        (points_world[:, 1] >= -0.7)     # Y ≥ -0.7
                    )
                    points_world = points_world[valid_mask]
                    
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

        # 按颜色 id 排序（红0 → 绿1 → 蓝2）
        COLOR_SEQUENCE = [0, 1, 2] # <--- 修改：将 1,2,3 改为 0,1,2
        color_to_block = {}
        for color_id in COLOR_SEQUENCE:
            for cls, pos in all_results:
                if cls == color_id:
                    color_to_block[color_id] = (cls, pos)
                    break

        # 检查是否检测到三种颜色
        if len(color_to_block) < 3:
            print(f"⚠️ 检测到的颜色不足三个，仅检测到: {list(color_to_block.keys())}")
            return []

        return [color_to_block[cid] for cid in COLOR_SEQUENCE]


    def process_gripper_data(self, data, threshold=0.1):
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
        if os.path.exists(GRASP_PATH_BLOCKS):
            return
        print("🧩 生成抓取姿态中...")
        obj = mcm.CollisionModel(BLOCK_MODEL_PATH)
 
        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=20,
            min_dist_between_sampled_contact_points=0.03,
            contact_offset=0.01
        )
        grasps.save_to_disk(GRASP_PATH_BLOCKS)
        print(f"✅ 保存抓取姿态，共 {len(grasps)} 个")

    def _execute_trajectory(self, arm, mot_data, steps_per_segment=5):
        """
        在原 mot_data 基础上插值执行，使机械臂运动更平滑
        :param steps_per_segment: 每两帧之间插值步数
 ``23       """
        jv_list = mot_data.jv_list
        ev_list = mot_data.ev_list

        for k in range(len(jv_list) - 1):
            start_j = np.array(jv_list[k])
            end_j = np.array(jv_list[k + 1])
            start_gripper = 0.07 if ev_list[k] >= 0.09 else 0.0
            end_gripper = 0.07 if ev_list[k + 1] >= 0.09 else 0.0

            for i in range(1, steps_per_segment + 1):
                alpha = i / steps_per_segment
                jv = start_j * (1 - alpha) + end_j * alpha
                gripper_angle = start_gripper * (1 - alpha) + end_gripper * alpha
                arm.move_m(jv, kp = 10, kd = 0.8,vel_ref = 5)
                arm.gripper_control(angle=gripper_angle)
                time.sleep(0.02)

        # 执行最后一帧
        arm.move_j(jv_list[-1], speed=10)
        arm.gripper_control(angle=0.1 if ev_list[-1] >= 0.09 else 0.0)

 

    # -------------------------
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, start_pos, goal_pos, arm: PiperArmController, robot, obstacles, use_rrt:bool):
        print(f"🤖 从 {start_pos} 抓取 → 放到 {goal_pos}")
        cls_id, pos = start_pos
        block = mcm.CollisionModel(BLOCK_MODEL_PATH)
        block.pos = np.array(pos, dtype=float)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_BLOCKS)

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
            use_rrt=use_rrt
        )
        if mot_data is None:
            print("⚠️ 轨迹规划失败！")
            return False

        jv = mot_data.jv_list
        ev = mot_data.ev_list
        print(ev)
        binary_arr, change_indices = self.process_gripper_data(ev)
        print(change_indices)
        approach_path,pick_path,depart_path = self.split_trajectory_by_gripper(jv,change_indices)
        arm.open_gripper(width=0.08)
        arm.move_j(jv[0],speed=20,block=True)

        time.sleep(0.1)
        arm.move_jntspace_path(approach_path,speed=20)
        time.sleep(0.1)
        arm.close_gripper()
        time.sleep(0.1)
        arm.move_jntspace_path(pick_path,speed=20)
        time.sleep(0.1)
        arm.open_gripper(width = 0.08)
        time.sleep(1) 
        arm.move_jntspace_path(depart_path,speed=20)

        return mot_data
    # -------------------------
    # 机械臂选择
    # -------------------------
    def choose_arm(self, block_pos):
        # block_pos 是 (cls_id, [x, y, z])
        _, pos = block_pos
        if pos[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    # -------------------------
    # 主任务入口
    # -------------------------
    def run(self, show_camera=False):
  
        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.23, 0.07])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.375, 0.07]))
        ]


        self.create_grasps()
        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)

        # -------------------------
        # 第一步：检测所有方块
        # -------------------------
        blocks = self.detect_blocks(show=show_camera)
        if len(blocks) == 0:
            print("⚠️ 未检测到任何方块！")
            return False
        
        # -------------------------
        # 第二步：按颜色分类
        # -------------------------
        color_to_block = {0: None, 1: None, 2: None} # <--- 修改：将 1,2,3 改为 0,1,2
        for cls_id, pos in blocks:
            color_to_block[cls_id] = (cls_id, pos)

        detected_colors = [k for k, v in color_to_block.items() if v is not None]
        if len(detected_colors) < 3:
            print(f"⚠️ 检测到的颜色不足三个，仅检测到 {detected_colors}")
            return False

        # -------------------------
        # 第三步：按颜色顺序执行抓取与放置
        # 红(0) → 绿(1) → 蓝(2)
        # -------------------------
        color_sequence = [0, 1, 2] # <--- 修改：将 1,2,3 改为 0,1,2
        all_mot_data = []
        color_name_map = {0: "红色", 1: "绿色", 2: "蓝色"} # <--- 修改：将 1,2,3 改为 0,1,2

        use_rrt = False
        for i, color_id in enumerate(color_sequence):

            #最后一个方块添加障碍,并且启用rrt
            if i == 2:
                #添加障碍
                obstacles.append(mcm.gen_box(xyz_lengths=[0.05, 0.05, 0.10], pos=np.array([0.25, -0.3, 0])))
                use_rrt = False
            block = color_to_block[color_id]
            target = TARGET_POSITIONS[i]
            color_name = color_name_map[color_id]

            print(f"\n=== 开始抓取第 {i + 1} 个方块：{color_name} ===")

            # 判断使用哪只手
            arm, robot = self.choose_arm(block)
            arm_name = "左臂" if arm is self.left_arm else "右臂"
            print(f"👉 使用 {arm_name} 抓取 {color_name} 方块")

            # --- 修改 z 坐标为 0，用于抓取 ---
            block_for_pick = (block[0], block[1].copy())  # 先复制原始坐标
            block_for_pick[1][2] = 0.0  # 强制 z = 0

            mot_data = self.execute_pick_place(block_for_pick, target, arm, robot, obstacles, use_rrt)

            if mot_data is None:
                print(f"❌ {color_name} 方块堆叠失败")
                continue

            print(f"✅ {color_name} 方块堆叠成功（由 {arm_name} 完成）")
        

 
        return True


# ==================================
# main
# ==================================
def main():
    task = MultiCameraBlockTask()
    try:
        task.left_arm.move_j([0]*6, speed=20)
        task.right_arm.move_j([0]*6, speed=20)
        start_time = time.time()
        success = task.run(show_camera=False)
        end_time = time.time()
        print(f"推理时间:{start_time -  end_time}")
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