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
# --- 新增引用 (模仿 task_shoes.py) ---
from wrs.vision.depth_camera.util_functions import registration_ptpt
import wrs.modeling.geometric_model as gm
# ------------------------------------
from my_project.tiaozhanbei.stack_blocks_three.constant import YOLO_MODEL_BLOCKS_PATH, BLOCK_MODEL_PATH, \
    GRASP_PATH_BLOCKS, \
    TARGET_POSITIONS, MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, \
    yolo_detect_world_positions


# ------------------------------------------------------------------
# 新增函数：强制 Z 轴垂直向上 (从 task_cup.py 迁移)
# ------------------------------------------------------------------
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
    norm_x = np.linalg.norm(x_new)
    if norm_x < 1e-9:
        # If the original x-axis was aligned with z, use a default x
        x_new = np.array([1, 0, 0])
    else:
        x_new /= norm_x

    # Recompute y-axis using cross product to ensure orthogonality
    y_new = np.cross(z_new, x_new)
    norm_y = np.linalg.norm(y_new)
    if norm_y < 1e-9:
        # Should not happen if x_new and z_new are orthogonal
        y_new = np.array([0, 1, 0])
    else:
        y_new /= norm_y

    R_new = np.column_stack((x_new, y_new, z_new))
    return R_new


class MultiCameraBlockTask:
    def __init__(self):
        # ========== 硬件与模型 ==========
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.yolo = init_yolo(YOLO_MODEL_BLOCKS_PATH)

        # 摄像头定义 (模仿 task_shoes.py)
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            # "left": {"cam": init_camera(camera_id='left'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            # "right": {"cam": init_camera(camera_id='right'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }

    # ------------------------------------------------------------------
    # 替换 detect_blocks
    # (使用 task_shoes.py 的检测与DBSCAN去重逻辑)
    # ------------------------------------------------------------------
    def detect_blocks(self, show=False, eps=0.03):
        """使用多摄像头和YOLO检测方块(带点云), 并使用DBSCAN去重"""
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
                    # 使用方块原有的置信度 (task_block.py 中未明确, 假设 0.1)
                    if conf < 0.1:
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
            if not mask:
                continue

            cls_positions = all_positions[mask]
            clustering = DBSCAN(eps=eps, min_samples=1).fit(cls_positions)
            labels = clustering.labels_

            for lbl in np.unique(labels):
                cluster_mask = (labels == lbl)
                cluster_centroids = cls_positions[cluster_mask]
                centroid = cluster_centroids.mean(axis=0)

                # 聚合此聚类中的所有点云
                original_indices_mask = np.where(cluster_mask)[0]
                original_indices = [mask[i] for i in original_indices_mask]
                all_cluster_pcds = [all_results[i][2] for i in original_indices]
                aggregated_pcd = np.concatenate(all_cluster_pcds, axis=0)

                # 方块需要点云用于后续ICP
                deduped_results.append((cls, centroid.tolist(), aggregated_pcd))

        print(f"DBSCAN去重后，检测到 {len(deduped_results)} 个方块。")
        return deduped_results

    # ------------------------------------------------------------------
    # 更新 align_block_pose (集成 Z 轴修正)
    # ------------------------------------------------------------------
    def align_block_pose(self, base, block_model_path, real_pcd, visualize=False):
        """点云配准以修正方块姿态，同时可选可视化，并在配准后强制 Z 轴垂直向上"""
        # 使用 gm.GeometricModel (同 task_shoes.py)
        block = gm.GeometricModel(block_model_path)
        # 使用 task_block.py 原有的采样参数
        model_pcd = block.sample_surface(radius=0.001, n_samples=8000)
        # 使用 task_shoes.py 的z轴过滤
        model_pcd = model_pcd[model_pcd[:, 2] > 0.005]

        if visualize:
            # 在世界中显示原始模型点云和实际点云
            gm.gen_pointcloud(model_pcd, rgba=np.array([0, 0, 1, 0.5])).attach_to(base)  # 蓝色: 模型
            gm.gen_pointcloud(real_pcd, rgba=np.array([0, 1, 0, 0.5])).attach_to(base)  # 绿色: 实际点云

        print("🧩 开始ICP点云配准以匹配方块方向...")
        # 使用 task_block.py 原有的ICP降采样参数
        icp_result = registration_ptpt(model_pcd, real_pcd, downsampling_voxelsize=0.01)
        transformation = icp_result[2]

        # --- 新增步骤：强制 Z 轴向上 ---
        T = transformation
        R = T[:3, :3]
        p = T[:3, 3]

        R_aligned = align_z_to_up(R)
        T_aligned = rm.homomat_from_posrot(p, R_aligned)

        block.homomat = T_aligned
        # --------------------------------

        block.attach_to(base)

        # 显示配准后的模型点云
        aligned_pcd = rm.transform_points_by_homomat(T_aligned, model_pcd.copy())
        if visualize:
            gm.gen_pointcloud(aligned_pcd, rgba=np.array([1, 0, 0, 0.6])).attach_to(base)  # 红色: 配准结果

        print("✅ ICP配准完成，Z轴已强制向上")
        return T_aligned

    # -------------------------
    # (以下函数保持 task_block.py 原有逻辑)
    # -------------------------

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
        stage1 = jv[:cp1]  # 第一阶段：从开始到第一个变化点
        stage2 = jv[cp1:cp2]  # 第二阶段：第一个变化点到第二个变化点
        stage3 = jv[cp2:]  # 第三阶段：第二个变化点到结束

        return stage1, stage2, stage3

    # -------------------------
    # 生成抓取姿态
    # (修改：增加 base 参数用于可视化, 模仿 task_shoes.py)
    # -------------------------
    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH_BLOCKS):
            return
        print("🧩 生成抓取姿态中...")
        obj = mcm.CollisionModel(BLOCK_MODEL_PATH)
        obj.attach_to(base)  # <--- 增加 attach_to(base)

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

    # (保留 task_block.py 原有的 _execute_trajectory, 即使它未被调用)
    def _execute_trajectory(self, arm, mot_data, steps_per_segment=5):
        """
        在原 mot_data 基础上插值执行，使机械臂运动更平滑
        :param steps_per_segment: 每两帧之间插值步数
        """
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
                arm.move_m(jv, kp=10, kd=0.8, vel_ref=5)
                arm.gripper_control(angle=gripper_angle)
                time.sleep(0.02)

        # 执行最后一帧
        arm.move_j(jv_list[-1], speed=10)
        arm.gripper_control(angle=0.1 if ev_list[-1] >= 0.09 else 0.0)

    # -------------------------
    # 执行 pick & place (保持 task_block.py 原有逻辑)
    # -------------------------
    def execute_pick_place(self, start_pos, goal_pos, arm: PiperArmController, robot, obstacles, use_rrt: bool, pick_depart_distance = .05):
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
            pick_depart_distance=pick_depart_distance,
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
        approach_path, pick_path, depart_path = self.split_trajectory_by_gripper(jv, change_indices)
        arm.open_gripper(width=0.08)
        arm.move_j(jv[0], speed=20, block=True)

        time.sleep(0.1)
        arm.move_jntspace_path(approach_path, speed=20)
        time.sleep(0.1)
        arm.close_gripper()
        time.sleep(0.1)
        arm.move_jntspace_path(pick_path, speed=20)
        time.sleep(0.1)
        arm.open_gripper(width=0.08)
        time.sleep(1)
        arm.move_jntspace_path(depart_path, speed=20)

        return mot_data

    # -------------------------
    # 机械臂选择 (保持 task_block.py 原有逻辑)
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
    # (修改：集成ICP流程)
    # -------------------------
    def run(self, show_camera=False):

        # --- 新增: 初始化 base (模仿 task_shoes.py) ---
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)
        # ----------------------------------------------

        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.23, 0.07])),
            mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.375, 0.07]))
        ]
        # --- 新增: 附加障碍物到 base ---
        [o.attach_to(base) for o in obstacles]
        # -------------------------------

        # --- 修改: 传入 base ---
        self.create_grasps(base)
        self.left_arm.move_j([0] * 6, speed=20)
        self.right_arm.move_j([0] * 6, speed=20)

        # -------------------------
        # 第一步：检测所有方块 (使用新的 detect_blocks)
        # -------------------------
        blocks = self.detect_blocks(show=show_camera)
        if len(blocks) == 0:
            print("⚠️ 未检测到任何方块！")
            return False

        # -------------------------
        # 第二步：按颜色分类 (修改：适配新的 blocks 格式)
        # -------------------------
        color_to_block = {0: None, 1: None, 2: None}
        # --- 修改: blocks 现在是 (cls, pos, pcd) 元组列表 ---
        for cls_id, pos, pcd in blocks:
            if cls_id in color_to_block:
                color_to_block[cls_id] = (cls_id, pos, pcd)  # 存储 pcd

        detected_colors = [k for k, v in color_to_block.items() if v is not None]
        if len(detected_colors) < 3:
            print(f"⚠️ 检测到的颜色不足三个，仅检测到 {detected_colors}")
            return False

        # -------------------------
        # 第三步：按颜色顺序执行抓取与放置 (保持原有顺序)
        # 红(0) → 绿(1) → 蓝(2)
        # -------------------------
        color_sequence = [0, 1, 2]
        all_mot_data = []
        color_name_map = {0: "红色", 1: "绿色", 2: "蓝色"}

        use_rrt = False
        for i, color_id in enumerate(color_sequence):

            # 保持原有的RRT和障碍物逻辑
            if i == 2:
                # 添加障碍
                obs_extra = mcm.gen_box(xyz_lengths=[0.05, 0.05, 0.10], pos=np.array([0.25, -0.3, 0]))
                obstacles.append(obs_extra)
                obs_extra.attach_to(base)  # 附加到 base
                use_rrt = False

            block_data = color_to_block[color_id]
            if block_data is None:
                print(f"⚠️ 字典中未找到 {color_name_map[color_id]} 方块的数据, 跳过。")
                continue

            # --- 修改: 解包 (cls, pos, pcd) ---
            cls_id, pos, pcd = block_data
            block = (cls_id, pos)  # (用于 choose_arm)
            target = TARGET_POSITIONS[i]
            color_name = color_name_map[color_id]

            print(f"\n=== 开始抓取第 {i + 1} 个方块：{color_name} ===")

            # --- 新增: ICP 配准流程 (模仿 task_shoes.py) ---
            # 使用 task_block.py 原有的点云过滤
            filtered_pcd = pcd[(pcd[:, 2] > 0.005) & (pcd[:, 2] < 0.06)]
            if len(filtered_pcd) < 100:
                print(f"⚠️ 方块 {cls_id} 过滤后点云太少, 跳过ICP。")
            else:
                # 调用更新后的 align_block_pose，它会强制 Z 轴向上
                self.align_block_pose(base, BLOCK_MODEL_PATH, filtered_pcd, visualize=show_camera)

            # (可选) 像 task_shoes.py 一样, 在抓取前运行仿真以查看ICP结果
            if show_camera:
                print("🎬 仿真显示 ICP 结果...")
                base.run()
            # ----------------------------------------------

            # 判断使用哪只手 (保持原有逻辑)
            arm, robot = self.choose_arm(block)
            arm_name = "左臂" if arm is self.left_arm else "右臂"
            print(f"👉 使用 {arm_name} 抓取 {color_name} 方块")

            # --- 修改 z 坐标为 0，用于抓取 (保持原有逻辑) ---
            block_for_pick = (block[0], block[1].copy())  # 先复制原始坐标
            block_for_pick[1][2] = 0.0  # 强制 z = 0 (使用检测到的质心, 保持原有逻辑)

            mot_data = self.execute_pick_place(block_for_pick, target, arm, robot, obstacles, use_rrt)
            if i == 2:
                mot_data = self.execute_pick_place(block_for_pick, target, arm, robot, obstacles, use_rrt, pick_depart_distance= 0.15)
            if mot_data is None:
                print(f"❌ {color_name} 方块堆叠失败")
                continue

            # 存储运动数据用于最终的统一回放
            all_mot_data.append((mot_data, base))
            print(f"✅ {color_name} 方块堆叠成功（由 {arm_name} 完成）")

        # -------- 步骤4：统一仿真回放 --------
        if show_camera:
            print("\n🎬 开始统一仿真回放...")
            for mot_data, base in all_mot_data:
                if not hasattr(mot_data, "mesh_list"):
                    continue
                for mesh in mot_data.mesh_list:
                    mesh.attach_to(base)
                    mesh.show_cdprim()
            base.run()

        return True


# ==================================
# main (保持 task_block.py 原有逻辑)
# ==================================
def main():
    task = MultiCameraBlockTask()
    try:
        task.left_arm.move_j([0] * 6, speed=20)
        task.right_arm.move_j([0] * 6, speed=20)
        start_time = time.time()
        success = task.run(show_camera=False)  # 可设为 True 查看ICP
        end_time = time.time()
        print(f"推理时间:{start_time - end_time}")
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