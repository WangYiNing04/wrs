#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/25 11:11
# @Author : ZhangXi
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/25
# @Author : ZhangXi

import os
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import wrs.basis.robot_math as rm
from my_project.tiaozhanbei.stack_bowls_three.constant import YOLO_MODEL_BOWLS_PATH, BOWL_MODEL_PATH, GRASP_PATH_BOWLS, \
    TARGET_POSITIONS, MIDDLE_CAM_C2W, LEFT_HAND_EYE, RIGHT_HAND_EYE
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
from wrs.robot_sim.robots.piper.piper_dual_arm import DualPiperNoBody
from wrs.robot_con.piper.piper import PiperArmController
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from my_project.tiaozhanbei.yolo_detect.yolo_utils import init_yolo, init_camera, transform_points_by_homomat, yolo_detect_world_positions


class MultiCameraBowlTask:
    def __init__(self, resources=None):
        if resources:
            print("🥣 使用外部 ResourceManager 资源")
            self.left_arm = resources.left_arm
            self.right_arm = resources.right_arm
            self.cameras = resources.cameras
            self.yolo = resources.yolo_bowl
        else:
            print("🥣 独立初始化 Bowl 任务资源")
            self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
            self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
            self.yolo = init_yolo(YOLO_MODEL_BOWLS_PATH)
            self.cameras = {
                "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
            }
        # ========== 硬件与模型 ==========

        self.rbt_s = DualPiperNoBody()
        self.gripper = pg.PiperGripper()
 

    def detect_bowls(self, show=False, eps=0.03):
        """
        使用 YOLO + 点云检测碗，返回世界坐标系下的质心，按 Y 值排序靠左先抓
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
        # 去重聚类
        # ---------------------------
        points_array = np.array([pos for _, pos in all_results])
        clustering = DBSCAN(eps=eps, min_samples=1).fit(points_array)
        deduped_results = []
        for lbl in np.unique(clustering.labels_):
            cluster_pts = points_array[clustering.labels_ == lbl]
            centroid = cluster_pts.mean(axis=0)
            deduped_results.append((1, centroid.tolist()))  # cls_id统一为1

        # 按 Y 值排序（靠左先抓）
        deduped_results.sort(key=lambda x: -x[1][1])

        return deduped_results

    # -------------------------
    # 生成抓取姿态
    # -------------------------
    def create_grasps(self, base):
        if os.path.exists(GRASP_PATH_BOWLS):
            return
        print("🥣 生成抓取姿态中...")
        obj = mcm.CollisionModel(BOWL_MODEL_PATH)
        obj.attach_to(base)
        grasps = gpa.plan_gripper_grasps(
            self.gripper, obj,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=20,
            min_dist_between_sampled_contact_points=0.03,
            contact_offset=0.01
        )
        grasps.save_to_disk(GRASP_PATH_BOWLS)
        print(f"✅ 保存抓取姿态，共 {len(grasps)} 个")

    def _execute_trajectory(self, arm, mot_data, base, show_sim=False):
        for jv, ev in zip(mot_data.jv_list, mot_data.ev_list):
            arm.move_j(jv, speed=10)
            arm.gripper_control(angle=0.05 if ev >= 0.09 else 0.0, effort=0)
            time.sleep(0.2)

        if not show_sim:
            return

        # -------------------------
        # 设置仿真动画
        # -------------------------
        class AnimeData:
            def __init__(self, mot_data):
                self.mot_data = mot_data
                self.counter = 0

        anime_data = AnimeData(mot_data)

        def update(task, anime_data=anime_data):
            if anime_data.counter > 0:
                anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
            if anime_data.counter >= len(anime_data.mot_data):
                return task.done  # 当前物体动画播放完成
            mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
            mesh_model.attach_to(base)
            mesh_model.show_cdprim()
            # 按空格播放下一帧
            if base.inputmgr.keymap.get('space', False):
                anime_data.counter += 1
            return task.cont

        base.taskMgr.add(update, "update")

        # -------------------------
        # 非阻塞循环等待窗口关闭
        # -------------------------
        try:
            while not base.app_closed:  # base.app_closed 表示窗口是否关闭
                base.taskMgr.step()
                time.sleep(0.01)
        except Exception:
            pass

    # -------------------------
    # 执行 pick & place
    # -------------------------
    def execute_pick_place(self, start_pos, goal_pos, arm, robot, obstacles):
        cls_id, pos = start_pos
        block = mcm.CollisionModel(BOWL_MODEL_PATH)
        block.pos = np.array(pos, dtype=float)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        planner = ppp.PickPlacePlanner(robot)
        grasps = gg.GraspCollection.load_from_disk(GRASP_PATH_BOWLS)

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
            return None

        # 先执行机械臂动作
        self._execute_trajectory(arm, mot_data)
        return mot_data

    # -------------------------
    # 机械臂选择
    # -------------------------
    def choose_arm(self, block):
        # block = (cls_id, [x, y, z])
        pos = block[1]
        if pos[1] > -0.3:
            return self.left_arm, self.rbt_s.use_lft()
        else:
            return self.right_arm, self.rbt_s.use_rgt()

    # -------------------------
    # 主任务入口
    # -------------------------
    def run(self, show_camera=False):
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)
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

        # 检测碗
        blocks = self.detect_bowls(show=show_camera)
        if len(blocks) < 3:
            print(f"⚠️ 检测到 {len(blocks)} 个碗，不足三个！")
            return False

        print("🔹 检测到的碗坐标（世界坐标系）：")
        for i, block in enumerate(blocks):
            cls_id, pos = block
            print(f"碗 {i + 1}: {pos}")

        # 按顺序抓取并放置到 TARGET_POSITIONS
        all_mot_data = []
        for block, goal_pos in zip(blocks, TARGET_POSITIONS):
            arm, robot = self.choose_arm(block)
            mot_data = self.execute_pick_place(block, np.array(goal_pos), arm, robot, obstacles)
            if mot_data is not None:
                all_mot_data.append((mot_data, base))
            else:
                print(f"⚠️ 碗 {block} 的抓取放置失败！")

        # 统一仿真回放
        if all_mot_data:
            print("\n🎬 开始统一仿真回放...")
            for mot_data, base in all_mot_data:
                for mesh in mot_data.mesh_list:
                    mesh.attach_to(base)
                    mesh.show_cdprim()
            base.run()  # 自动播放

        return True


# ==================================
# main
# ==================================
def main():
    task = MultiCameraBowlTask()
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
