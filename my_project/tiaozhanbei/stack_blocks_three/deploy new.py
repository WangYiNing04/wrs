'''
Author: wang yining
Date: 2025-10-21 16:35:06
LastEditTime: 2025-10-24 14:42:12
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/deploy new.py
Description: 三个方块堆叠任务
e-mail: wangyining0408@outlook.com
'''

'''
任务流程:
1. 先抓红色方块放到中间
2. 再抓绿色方块放到红色方块上面
3. 最后抓蓝色方块放到绿色方块上面
'''

import os
import time
import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
import wrs.basis.robot_math as rm
from wrs import wd, rm, mgm, mcm, ppp, rrtc, gg, gpa
import wrs.robot_sim.robots.piper.piper_single_arm as psa
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.robot_con.piper.piper import PiperArmController
from wrs.drivers.devices.realsense.realsense_d400s import *
import yaml
from my_project.tiaozhanbei.yolo_detect.yolo_utils import (
    init_yolo,
    init_camera,
    transform_points_by_homomat,
    yolo_detect_world_positions
)

SHOW_IMAGE = True   # 是否显示检测结果
NEIGHBORHOOD_SIZE = 5  # 邻域估计窗口大小

# 简化的点云处理器
class PointCloudProcessor:
    """简化的点云处理器"""
    
    def __init__(self, config_path=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/config/camera_correspondence.yaml'):
        # middle camera hand-eye matrix (相机到世界的变换矩阵)
        self._init_calib_mat = np.array([
            [0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
            [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
            [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 相机相关属性
        self.config_path = config_path
        self.rs_pipelines = {}
        self.camera_active = False
        
        # 初始化相机
        self.initialize_cameras()
    
    def align_pcd(self, pcd):
        """将点云从相机坐标系转换到世界坐标系"""
        c2w_mat = self._init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
    def initialize_cameras(self):
        """初始化相机"""
        try:
            # 读取YAML配置文件
            with open(self.config_path, 'r') as file:
                camera_config = yaml.safe_load(file)

            # 从配置中提取相机ID
            camera_roles = {
                'middle': camera_config['middle_camera']['ID'],
            }

            # 查找实际连接的设备
            available_serials, ctx = find_devices()
            print("检测到设备:", available_serials)

            # 初始化相机（用字典存储，键为角色名称）
            for role, cam_id in camera_roles.items():
                if cam_id in available_serials:
                    print(f"正在初始化 {role} 相机 (ID: {cam_id})")
                    pipeline = RealSenseD400(device=cam_id)
                    pipeline.reset()
                    time.sleep(2)
                    pipeline = RealSenseD400(device=cam_id)  # 重新初始化
                    self.rs_pipelines[role] = pipeline
                    print(f"{role} 相机初始化成功")
                else:
                    print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")
                    
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise
    
    def get_camera_data(self, role='middle'):
        """从指定相机获取点云和图像数据"""
        if role not in self.rs_pipelines:
            print(f"错误: 未找到 {role} 相机")
            return None, None, None, None
            
        try:
            pcd, pcd_color, depth_img, color_img = self.rs_pipelines[role].get_pcd_texture_depth()
            return pcd, pcd_color, depth_img, color_img
        except Exception as e:
            print(f"从 {role} 相机获取数据失败: {e}")
            return None, None, None, None

class BlockDetector:
    """方块检测器，用于检测三个方块"""
    
    def __init__(self, yolo_model_path='/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/best_block.pt'):
        """
        初始化方块检测器
        
        Args:
            yolo_model_path: YOLO模型路径
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.processor = PointCloudProcessor()
        print(f"已加载YOLO模型: {yolo_model_path}")
    
    def detect_blocks(self):
        """
        检测三个方块的位置
        
        Returns:
            list: 包含三个方块世界坐标位置的列表
        """
        print("开始检测方块位置...")
        
        while True:
            try:
                # 获取相机数据
                pcd, pcd_color, depth_img, color_img = self.processor.get_camera_data('middle')

                pcd_world = self.processor.align_pcd(pcd)
                pcd_matrix = pcd.reshape(color_img.shape[0], color_img.shape[1], 3)
                if pcd is None:
                    print("无法获取相机数据")
                    return None
                
                # 使用YOLO检测
                results = yolo_detect_world_positions(self.yolo_model, color_img, pcd_world, show=False)
                
                if results is not None:
                    print("\n=== 检测到的方块坐标 ===")
                    for cls_id, pos in results:
                        print(f"ID:{cls_id} -> X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
                            # --- 退出键 ---
                    break
            except Exception as e:
                print(f"读取帧失败: {e}")
                break
         
        #关闭相机
        
        # 处理检测结果
        blocks_positions = results
        sorted_blocks = sorted(blocks_positions, key=lambda x: x[0])

        # 提取不带 ID 的位置数据
        blocks_positions = [block[1].tolist() for block in sorted_blocks]

        #blocks_positions = [(0,np.array(0.3,0,0)),(1,np.array(0.3,-0.3,0)),(2,np.array(0.3,-0.6,0))]
        blocks_positions = [[0.3,0,0],[0.3,-0.3,0],[0.3,-0.6,0]]
        return blocks_positions

class BlockStackingTask:
    """方块堆叠任务执行器"""
    
    def __init__(self):
        """初始化任务执行器"""
        # 初始化机械臂控制器
        self.left_arm_con = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm_con = PiperArmController(can_name='can1', has_gripper=True)
        
        # 初始化检测器
        self.detector = BlockDetector()
        
        # 方块模型路径
        self.block_path = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl"
        
        # 目标位置（中间位置）
        self.target_positions = [
            [0.25, -0.30, 0.00],      # 红色方块放在中间
            [0.25, -0.30, 0.05],      # 绿色方块放在红色方块上面
            [0.25, -0.30, 0.10]       # 蓝色方块放在绿色方块上面
        ]
        
        # 抓取姿态文件路径 block_grasps manual_grasps piper_gripper_grasps.pickle filter_grasps
        self.grasp_save_path = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/manual_grasps.pickle"
        
        # 初始化夹爪和机器人
        self.gripper = pg.PiperGripper()
        
    def create_grasp_collection(self, obj_path, save_path, base):
        """为方块创建抓取姿态集合"""
        print("正在生成方块抓取姿态集合...")
        
        # # 创建3D环境
        # base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
        # mgm.gen_frame().attach_to(base)
        
        # 加载方块模型
        obj_cmodel = mcm.CollisionModel(obj_path)
        obj_cmodel.attach_to(base)
        
        # 生成抓取姿态
        grasp_collection = gpa.plan_gripper_grasps(
            self.gripper,
            obj_cmodel,
            angle_between_contact_normals=rm.radians(175),
            rotation_interval=rm.radians(15),
            max_samples=100,
            min_dist_between_sampled_contact_points=.03,  # 使用优化后的参数
            contact_offset=.01,
            toggle_dbg=False
        )
        
        print(f"生成了 {len(grasp_collection)} 个抓取姿态")
        #grasp_collection = grasp_collection.limit(20)
        print(f"实际获取抓取数量: {len(grasp_collection)}")
        
        # 保存抓取姿态
        grasp_collection.save_to_disk(file_name=save_path)
        print(f"抓取姿态已保存到: {save_path}")
        
        return grasp_collection
    
    def execute_pick_place(self, block_pos, target_pos, arm_controller, robot, base, obstacle_list):
        """执行单个方块的抓取和放置"""
        print(f"执行抓取和放置: 从 {block_pos} 到 {target_pos}")
        
        # 创建3D环境
        # base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
        # mgm.gen_frame().attach_to(base)
        
        # 设置旋转矩阵
        start_rot = rm.rotmat_from_euler(0, 0, 0)
        goal_rot = rm.rotmat_from_euler(0, 0, 0)
        
        # 创建起始位置的方块
        block_start = mcm.CollisionModel(self.block_path)
        block_start.rgba = rm.np.array([.5, .5, .5, 1])
        block_start.pos = np.array(block_pos)
        block_start.rotmat = start_rot
        mgm.gen_frame().attach_to(block_start)
        
        # 可视化起始位置
        block_start_copy = block_start.copy()
        block_start_copy.attach_to(base)
        block_start_copy.show_cdprim()
        
        # 创建目标位置的方块（半透明）
        block_goal = mcm.CollisionModel(self.block_path)
        block_goal.pos = np.array(target_pos)
        block_goal.rotmat = goal_rot
        block_goal_copy = block_goal.copy()
        block_goal_copy.rgb = rm.const.tab20_list[0]
        block_goal_copy.alpha = .3
        block_goal_copy.attach_to(base)
        block_goal_copy.show_cdprim()
        
        # 创建机器人
        robot.gen_meshmodel().attach_to(base)
        
        # 实例化规划器
        rrtc_planner = rrtc.RRTConnect(robot)
        ppp_planner = ppp.PickPlacePlanner(robot)
        
        # 加载抓取姿态集合
        grasp_collection = gg.GraspCollection.load_from_disk(file_name=self.grasp_save_path)
        start_conf = robot.get_jnt_values()
        goal_pose_list = [(target_pos, goal_rot)]
        
        for grasp in grasp_collection:
                    self.gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
                    self.gripper.gen_meshmodel(alpha=1).attach_to(base)

        # 生成Pick and Place运动
        mot_data = ppp_planner.gen_pick_and_place(
            obj_cmodel=block_start,
            end_jnt_values=start_conf,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            pick_approach_direction=-rm.const.z_ax,
            place_approach_distance_list=[.05] * len(goal_pose_list),
            place_depart_distance_list=[.05] * len(goal_pose_list),
            pick_approach_distance=.05,
            pick_depart_distance=.05,
            pick_depart_direction=rm.const.z_ax,
            obstacle_list=obstacle_list,
            use_rrt=True
        )
        
        if mot_data is None:
            print("错误：无法生成Pick-and-Place运动轨迹！")
            return False
        
        print("Pick-and-Place运动轨迹生成成功！")
        
        # 导出关节角轨迹
        trajectory, traj_path = self.export_joint_trajectory(mot_data)
        
        # 执行运动
        self.execute_motion(arm_controller, traj_path, mot_data)
        
        return True
    

    def export_joint_trajectory(self, mot_data, save_dir=None, filename=None):
        """导出关节角轨迹到JSON文件"""
        # 合并关节角和夹爪宽度
        trajectory = [
            jnt_values.tolist() + [ev]  # 将关节角和夹爪值合并
            for jnt_values, ev in zip(mot_data.jv_list, mot_data.ev_list)
        ]
        
        # 设置默认保存路径
        if save_dir is None:
            save_dir = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/exported"
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 设置默认文件名
        if filename is None:
            filename = f"joint_trajectory_blocks_{int(time.time())}.json"
        
        # 保存为JSON文件
        saved_path = os.path.join(save_dir, filename)
        with open(saved_path, 'w', encoding='utf-8') as f:
            json.dump({"joint_trajectory": trajectory}, f, ensure_ascii=False, indent=2)
        
        print(f"关节角+夹爪轨迹已保存: {saved_path} (共 {len(trajectory)} 个点)")
        return trajectory, saved_path
    
    def execute_motion(self, arm_controller, json_path, mot_data):
        """从JSON文件读取关节角轨迹并执行"""

        arm_controller.gripper_control(angle=0.07, effort=0)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"未找到轨迹文件: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'joint_trajectory' not in data:
            raise ValueError("轨迹文件格式不正确，缺少 'joint_trajectory' 字段")

        trajectory = data['joint_trajectory']
        print(f"读取到 {len(trajectory)} 个关节轨迹点，将依次执行 move_j（6轴）...")
        jv = mot_data.jv_list
        ev = mot_data.ev_list
        print(jv)
        print(ev)
        for i, jv in enumerate(trajectory):
            print(f"执行第 {i+1}/{len(trajectory)} 个点: {jv}")
            arm_controller.move_j(jv[:6], speed=10)
            time.sleep(0.2)

            if ev[i] >= 0.09:
                gripper_width = 0.07
            else:
                gripper_width = 0.0
            print(f"夹爪宽度: {gripper_width}")
            arm_controller.gripper_control(angle=gripper_width, effort=0)
    
    def choose_arm(self, block_pos):
        """根据方块位置选择使用哪只机械臂"""
        if block_pos[1] > -0.3:  # Y坐标大于-0.3使用左臂
            return self.left_arm_con, psa.PiperSglArm()
        else:  # 否则使用右臂
            return self.right_arm_con, psa.PiperSglArm(pos=[0, -0.6, 0])
    
    def run_stacking_task(self):
        """执行完整的方块堆叠任务"""
        print("开始执行三个方块堆叠任务...")

        # 创建3D环境
        base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
        mgm.gen_frame().attach_to(base)

        box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5]))
        box1.attach_to(base)
        box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
        box2.attach_to(base)
        
        obstacle_list = [box1,box2]
        # 1. 检测方块位置
        blocks_positions = self.detector.detect_blocks()

        if not blocks_positions or len(blocks_positions) < 3:
            print(f"检测到{len(blocks_positions) if blocks_positions else 0}个方块，未检测到全部方块")
        
        # 2. 创建抓取姿态集合（如果不存在）
        if not os.path.exists(self.grasp_save_path):
            self.create_grasp_collection(self.block_path, self.grasp_save_path, base)
        try:

            self.left_arm_con.move_j([0, 0, 0, 0, 0, 0], speed=20)

            self.right_arm_con.move_j([0, 0, 0, 0, 0, 0], speed=20)

            try:
                # 3. 按顺序执行堆叠任务
                for i in range(len(blocks_positions)):
                    print(f"\n=== 执行第{i+1}步：抓取第{i+1}个方块 ===")
                    
                    # 获取方块位置
                    block_pos = blocks_positions[i]
                    block_pos[-1] = 0 
                    target_pos = self.target_positions[i]
                    
                    # 选择机械臂
                    arm_controller, robot = self.choose_arm(block_pos)
                    
                    # 执行抓取和放置
                    success = self.execute_pick_place(block_pos, target_pos, arm_controller, robot, base, obstacle_list)
                    
                    if success:
                        print(f"第{i+1}个方块放置成功！")
                    else:
                        print(f"第{i+1}个方块放置失败！")
                        base.run()
                        return False
            except Exception as e:
                print(f"执行过程中出现错误: {e}")
                arm_controller.move_j([0, 0, 0, 0, 0, 0], speed=20)
                base.run()
                
        except KeyboardInterrupt:
            print("用户键盘终止")
            arm_controller.move_j([0, 0, 0, 0, 0, 0], speed=20)
            base.run()

        print("\n=== 所有方块堆叠任务完成！ ===")
        return True

def main():
    """主函数"""

    try:
        # 创建任务执行器
        task_executor = BlockStackingTask()
        
        # 执行堆叠任务
        success = task_executor.run_stacking_task()
        
        if success:
            print("任务执行成功！")
        else:
            print("任务执行失败！")
            
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()



if __name__ == '__main__':
    main()