#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/18 9:34
# @Author : ZhangXi

import os
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from my_project.tiaozhanbei.put_bottles_dustbin.constant import YOLO_MODEL_PATH, COKE_CAN_MODEL_PATH, \
WATER_GANTEN_MODEL_PATH, TEA_DONGFANG_MODEL_PATH, COKE_CAN_GRASP_PATH, WATER_GANTEN_GRASP_PATH, TEA_DONGFANG_GRASP_PATH , TRAJ_DIR, \
    TARGET_POSITIONS, MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE

from ultralytics import YOLO
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, yolo_detect_world_positions

class MultiCameraBottleTask:
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
            #"left": {"cam": init_camera(camera_id='left'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            #"right": {"cam": init_camera(camera_id='right'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }

    # -------------------------
    # YOLO多摄像头检测
    # -------------------------/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/tea_dongfang_grasps.pickle
    def detect_bottles(self, show=False, eps=0.03):
        """
        多摄像头检测瓶子，并根据空间位置去重
        :param show: 是否显示每个相机画面
        :param eps: 聚类半径，单位 m
        :return: 去重后的 (cls_id, [x, y, z]) 列表
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


        # ---------------------------
        # 将 z 值设为 0
        # ---------------------------
        deduped_results = [(cls_id, [x, y, 0]) for cls_id, (x, y, z) in deduped_results]

        # ---------------------------
        # 按 y > -0.3 优先，x 越小越优先排序
        # ---------------------------
        deduped_results.sort(key=lambda item: (
            # 优先排序 y > -0.3 的项（False(0) < True(1)，所以用 not 反转）
            not (item[1][1] > -0.3),  # y > -0.3 的排前面
            item[1][0]  # 然后按 x 升序
        ))
        
        print(deduped_results)
        return deduped_results


    # -------------------------
    # YOLO多摄像头检测
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
        stage1 = jv[:cp1]      # 第一阶段：从开始到第一个变化点
        stage2 = jv[cp1:cp2]   # 第二阶段：第一个变化点到第二个变化点
        stage3 = jv[cp2:]      # 第三阶段：第二个变化点到结束
        
        return stage1,stage2,stage3
    # -------------------------
    # 生成抓取姿态
    # -------------------------
    def create_grasps(self, grasp_path, model_path):
        if os.path.exists(grasp_path):
            return
        print("🧩 生成抓取姿态中...")
        obj = mcm.CollisionModel(model_path)

        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=20,
            min_dist_between_sampled_contact_points=0.03,
            contact_offset=0.01
        )
        grasps.save_to_disk(grasp_path)
        print(f"✅ 保存抓取姿态，共 {len(grasps)} 个")

    # -------------------------
    # 执行轨迹动画
    # -------------------------
    def _execute_trajectory(self, arm, mot_data, show_sim=False):

        jv = mot_data.jv_list
        ev = mot_data.ev_list

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
     
     
    # -------------------------
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, start_pos, goal_pos, goal_rot, arm, robot, obstacles, grasp_path, model_path, show_sim=False):

        print(f"🤖 从 {start_pos} 抓取 → 放到 {goal_pos}")
        bottle = mcm.CollisionModel(model_path)
        bottle.pos = np.array(start_pos,dtype=float)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(grasp_path)

        mot_data = planner.gen_pick_and_place(
            obj_cmodel=bottle,
            end_jnt_values=robot.get_jnt_values(),
            grasp_collection=grasps,
            goal_pose_list=[(goal_pos, goal_rot)],
            pick_approach_direction=-rm.const.z_ax,
            place_approach_distance_list=[.05],
            place_depart_distance_list=[.05],
            pick_approach_distance=.05,
            pick_depart_distance=.1,
            obstacle_list=obstacles,
            use_rrt=False
        )
        
        if mot_data is None:
            print("⚠️ 轨迹规划失败！")
            return False

        self._execute_trajectory(arm, mot_data, show_sim=show_sim)
        return True

    # -------------------------
    # 机械臂选择
    # -------------------------
    def choose_arm(self, bottle_pos):
        _, pos = bottle_pos
        if pos[1] > -0.3:
            return self.left_arm, self.rbt_s.lft_arm
        else:
            return self.right_arm, self.rbt_s.rgt_arm
    # -------------------------
    # 主任务入口
    # -------------------------
    def run(self, show_camera=False, show_sim=False):

        obstacles = [
            mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5])),
            mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
            #mcm.gen_box(xyz_lengths=[0.272, 0.01, 0.143], pos=np.array([0.45, -0.091, 0.715]))
            #mcm.CollisionModel(COKE_CAN_MODEL_PATH)
            #mcm.CollisionModel(WATER_GANTEN_MODEL_PATH)
            #mcm.CollisionModel(TEA_DONGFANG_MODEL_PATH)
        ] 


        self.create_grasps( COKE_CAN_GRASP_PATH, COKE_CAN_MODEL_PATH)
        self.create_grasps( WATER_GANTEN_GRASP_PATH, WATER_GANTEN_MODEL_PATH)
        self.create_grasps( TEA_DONGFANG_GRASP_PATH, TEA_DONGFANG_MODEL_PATH)

        self.left_arm.move_j([0]*6, speed=20)
        self.right_arm.move_j([0]*6, speed=20)

        bottles = self.detect_bottles(show=show_camera)
        print(bottles)
        
        if len(bottles) < 3:
            print(f"⚠️ 检测到 {len(bottles)} 个瓶子，不足三个！")
            return False
        
        # for bottle in bottles:
        #     _, pos = bottle
        #     obs = mcm.CollisionModel(COKE_CAN_MODEL_PATH)
        #     obs.pos = pos
        #     obstacles.append(obs)

        for i,bottle in enumerate(bottles):
            cls_id, pos = bottle
            #arm, robot = self.choose_arm(bottle)
            print(cls_id)
            if cls_id == 0:
                grasp_path = COKE_CAN_GRASP_PATH
                model_path = COKE_CAN_MODEL_PATH
            elif cls_id == 1:
                grasp_path = WATER_GANTEN_GRASP_PATH
                model_path = WATER_GANTEN_MODEL_PATH 
            else:
                grasp_path = TEA_DONGFANG_GRASP_PATH
                model_path = TEA_DONGFANG_MODEL_PATH

            print(f"\n=== 第 {i+1} 个瓶子 ===")
            
            if pos[1] > -0.3:
                arm,robot = self.left_arm, self.rbt_s.lft_arm
                target = TARGET_POSITIONS[0]
                goal_rot = rm.rotmat_from_euler(0, -2/np.pi, 0)
                if not self.execute_pick_place(pos, target, goal_rot, arm, robot, obstacles, grasp_path, model_path, show_sim=show_sim):
                    print(f"❌ 第 {i+1} 个瓶子堆叠失败")
                   
                    return False
            else:
                #arm,robot = self.right_arm, self.rbt_s.rgt_arm
                first_target = TARGET_POSITIONS[1] #先放到右边
                goal_rot = rm.rotmat_from_euler(0, 0, 0)
                if not self.execute_pick_place(pos, first_target,goal_rot, self.right_arm, self.rbt_s.rgt_arm, obstacles, grasp_path, model_path, show_sim=show_sim):
                    print(f"❌ 第 {i+1} 个瓶子堆叠失败")
                    
                    return False
                #arm,robot = self.left_arm, self.rbt_s.lft_arm
                second_target = TARGET_POSITIONS[0]
                pos = first_target
                goal_rot = rm.rotmat_from_euler(0, -2/np.pi, 0)
                if not self.execute_pick_place(pos, second_target,goal_rot, self.left_arm, self.rbt_s.lft_arm, obstacles, grasp_path, model_path, show_sim=show_sim):
                    print(f"❌ 第 {i+1} 个瓶子堆叠失败")
                   
                    return False
     
            print(f"✅ 第 {i+1} 个瓶子堆叠成功")

        print("\n🎯 所有瓶子堆叠完成！")
        return True

# ==================================
# main
# ==================================
def main():
    task = MultiCameraBottleTask()
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