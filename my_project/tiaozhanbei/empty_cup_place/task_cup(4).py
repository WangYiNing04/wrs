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
from my_project.tiaozhanbei.empty_cup_place.constant import YOLO_MODEL_CUPS_PATH, CUP_MODEL_PATH, GRASP_PATH_CUPS, \
    MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
import wrs.modeling.geometric_model as gm
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, \
    yolo_detect_world_positions

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/25
# @Author : ZhangXi



class MultiCameraCupTask:
    def __init__(self):
        # ========== 硬件与模型 ==========
        self.visualize = True
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
        self.yolo = init_yolo(YOLO_MODEL_CUPS_PATH)

        # 摄像头定义
        self.cameras = {
            "middle": {"cam": init_camera(camera_id = 'middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            # "left": {"cam": init_camera(camera_id='243322074546'), "type": "handeye", "handeye": LEFT_HAND_EYE},
            # "right": {"cam": init_camera(camera_id='243322071033'), "type": "handeye", "handeye": RIGHT_HAND_EYE}
        }

    # -------------------------
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

                    points_3d = cam.points_in_color_bbox((x1, y1, x2, y2))
                    if len(points_3d) == 0:
                        continue

                    if cam_info["type"] == "fixed":
                        points_world = transform_points_by_homomat(cam_info["c2w"], points_3d)
                    else:
                        points_world = self.rbt_s.transform_point_cloud_handeye(
                            cam_info["handeye"], points_3d,
                            component_name='lft_arm' if name == 'left' else 'rgt_arm'
                        )

                    # 质心
                    centroid = points_world.mean(axis=0)
                    all_results.append((int(cls_id), centroid.tolist(),points_world))

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
                deduped_results.append((cls, centroid.tolist(),all_results[mask[0]][2]))

        return deduped_results

    # -------------------------
    # 生成抓取姿态
    # -------------------------
    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH_CUPS):
            return
        print("☕ 生成抓取姿态中...")
        obj = mcm.CollisionModel(CUP_MODEL_PATH)
        obj.attach_to(base)
        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=20,
            min_dist_between_sampled_contact_points=0.03,
            contact_offset=0.01
        )
        grasps.save_to_disk(GRASP_PATH_CUPS)
        print(f"保存抓取姿态，共 {len(grasps)} 个")

    # -------------------------
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, pick_pos, place_pos, arm, robot, obstacles):
        block = mcm.CollisionModel(CUP_MODEL_PATH)
        block.pos = np.array(pick_pos, dtype=float)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_CUPS)

        mot_data = planner.gen_pick_and_place(
            obj_cmodel=block,
            end_jnt_values=robot.get_jnt_values(),
            grasp_collection=grasps,
            goal_pose_list=[(place_pos, goal_rot)],
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
            return None

        # 先执行机械臂动作
        for jv, ev in zip(mot_data.jv_list, mot_data.ev_list):
            arm.move_m(jv, kp = 15, kd=0.8,vel_ref=10)
            arm.gripper_control(angle=0.07 if ev >= 0.09 else 0.0, effort=0)
            time.sleep(0.2)

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
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)

        # 障碍物
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

        objects = self.detect_cups(show=show_camera)
        if not objects:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False

        if self.visualize:
            mgm.gen_frame().attach_to(base)

            for cls_id, pos, pcd in objects:
                gm.gen_frame(pos=pos, ax_length=.05).attach_to(base)
                gm.gen_sphere(pos=pos, radius=0.01, rgb=[1, 0, 0]).attach_to(base)

                cup = mcm.CollisionModel(CUP_MODEL_PATH)
                cup.pos = np.array(pos, dtype=float)
                cup.attach_to(base)

                # ✅ 在世界坐标中绘制点云
                if pcd is not None:
                    mgm.gen_pointcloud(pcd, rgba=np.array([0, 0, 1, 0.5])).attach_to(base)

                print(f"🎯 检测 {cls_id} → 世界坐标 {np.round(pos, 3)}，点数 {len(pcd) if pcd is not None else 0}")

            base.run()

        # 找到杯子和目标位置
        pick_obj = next((pos for cls_id, pos in objects if cls_id == 1), None)
        place_obj = next((pos for cls_id, pos in objects if cls_id == 0), None)
        if pick_obj is None or place_obj is None:
            print("⚠️ 没有检测到杯子或杯垫！")
            return False

        # 执行抓放，存储轨迹用于统一仿真
        arm, robot = self.choose_arm((1, pick_obj))
        print(f"\n☕ 抓取杯子 {pick_obj} → 放置 {place_obj}")
        mot_data = self.execute_pick_place(pick_obj, place_obj, arm, robot, obstacles)
        if mot_data is None:
            print("❌ 抓取失败")
            return False

        # -------------------------
        # 统一仿真回放
        # -------------------------
        print("\n🎬 开始统一仿真回放...")
        for mesh in mot_data.mesh_list:
            mesh.attach_to(base)
            mesh.show_cdprim()
        # base.run()

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