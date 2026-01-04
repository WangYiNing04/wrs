'''
Author: wang yining
Date: 2025-10-21 16:35:06
LastEditTime: 2025-10-22 21:54:43
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/deploy.py
Description: 三个方块堆叠任务
e-mail: wangyining0408@outlook.com
'''

'''
任务流程：
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
        if pcd is None:
                print("[align_pcd] 输入 pcd 为 None")
                return np.empty((0, 3), dtype=float)

        p = np.array(pcd, dtype=float)

        # 如果是图像形状 (H,W,3) -> reshape
        if p.ndim == 3 and p.shape[2] == 3:
            p = p.reshape(-1, 3)

        if p.size == 0:
            print("[align_pcd] 输入点云为空")
            return np.empty((0, 3), dtype=float)

        # 去掉无效点: depth==0 或 全零 或 包含 nan
        invalid = np.isnan(p).any(axis=1) | (np.all(np.isclose(p, 0.0), axis=1))
        valid_pts = p[~invalid]
        if valid_pts.size == 0:
            print("[align_pcd] 没有有效点 (所有点无效)")
            return np.empty((0, 3), dtype=float)

        c2w = self._init_calib_mat
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        # 显式旋转和平移
        pcd_world = (R @ valid_pts.T).T + t

        return pcd_world
    
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
    """方块检测器，用于检测红、绿、蓝三个方块"""
    
    def __init__(self, yolo_model_path='yolov8n.pt'):
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
            输入:
                pcd: Nx3 numpy array 点云坐标
                pcd_color: Nx3 numpy array 颜色 (RGB, 0~1)
            输出:
                blocks = {
                    'red': {'center': np.array([x, y, z]), 'R': 3x3旋转矩阵},
                    'green': {...},
                    'blue': {...}
                }
        """
        pcd_raw, pcd_color_raw, depth_img, color_img = self.processor.get_camera_data('middle')

        print("color_img[0,0] sample:", color_img[0,0])

        if pcd_raw is None:
            print("[detect_blocks] 未获取到点云")
            return {}

        # 确保为 numpy 数组并 reshape 为 (N,3) / (N,3) colors
        pcd_arr = np.array(pcd_raw, dtype=float)
        color_arr = np.array(pcd_color_raw, dtype=float)

        # 如果为图像形状，reshape
        if pcd_arr.ndim == 3 and pcd_arr.shape[2] == 3:
            pcd_arr = pcd_arr.reshape(-1, 3)
        if color_arr.ndim == 3 and color_arr.shape[2] == 3:
            color_arr = color_arr.reshape(-1, 3)

        print("原始 pcd shapes:", pcd_arr.shape, color_arr.shape)
        print("原始 pcd min/max z:", np.nanmin(pcd_arr[:,2]), np.nanmax(pcd_arr[:,2]))

        # 构造有效 mask：排除 NaN、全零以及 z==0（无深度）
        valid_mask = (~np.isnan(pcd_arr).any(axis=1)) & (~np.all(np.isclose(pcd_arr, 0.0), axis=1)) & (pcd_arr[:,2] > 0.0)
        print(f"有效点数: {np.count_nonzero(valid_mask)}/{len(valid_mask)}")

        if np.count_nonzero(valid_mask) == 0:
            print("[detect_blocks] 没有有效点")
            return {}

        # 只保留有效点和对应颜色（保持对齐）
        pcd_valid = pcd_arr[valid_mask]
        color_valid = color_arr[valid_mask]

        # 将这些有效点变换到世界系（使用上面改进的 align_pcd，但这里我们直接调用 align_pcd）
        pcd_world = self.processor.align_pcd(pcd_valid)  # align_pcd 已会去除无效点，但我们这里已经提前过滤了
        # 注意：align_pcd 返回的是对 valid pts 的变换结果（长度一致）
        print("变换后 pcd_world 示例:", pcd_world.shape, pcd_world[:5])

        # 颜色处理：color_valid 对应 pcd_world 的顺序
        #rgb_uint8 = (np.clip(color_valid, 0.0, 1.0) * 255).astype(np.uint8)

        # 确保颜色值为 uint8
        if color_valid.dtype != np.uint8:
            rgb_uint8 = (np.clip(color_valid, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            rgb_uint8 = color_valid

        # 注意：我们 reshape 为 (N,1,3) 给 cv2，再 reshape 回 (N,3)
        hsv = cv2.cvtColor(rgb_uint8.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        # debug 打印 HSV 范围
        print("HSV stats: H[{:.1f},{:.1f}] S[{:.1f},{:.1f}] V[{:.1f},{:.1f}]".format(
            float(np.min(hsv[:,0])), float(np.max(hsv[:,0])),
            float(np.min(hsv[:,1])), float(np.max(hsv[:,1])),
            float(np.min(hsv[:,2])), float(np.max(hsv[:,2]))
        ))

        # 高度裁剪：在 world frame 上筛选高度接近 table+0~5cm 的点
        z_vals = pcd_world[:, 2]
        
        z_min = 0
        mask_height = (z_vals > z_min + 0.045) & (z_vals < z_min + 0.055)
        print(f"高度范围: z_min={z_min:.4f}, kept {np.count_nonzero(mask_height)} points in height band")

        # 定义HSV阈值 (根据你的相机稍微调)
        color_ranges = {
            'red':   [(0, 100, 60), (10, 255, 255)],
            'green': [(35, 80, 60), (85, 255, 255)],
            'blue':  [(90, 80, 60), (130, 255, 255)]
        }

        # --------------------------
        # 3. 对每种颜色提取并计算位姿
        # --------------------------
        blocks = {}
        for name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # 计算颜色掩码：确保沿最后一个通道合并，得到 (N,) 布尔数组
            mask_color = np.all((hsv >= lower) & (hsv <= upper), axis=1)   # axis=1 因为 hsv 已是 (N,3)

            # debug: 每种颜色匹配数
            print(f"{name} color matches: {np.count_nonzero(mask_color)}")

            # 组合掩码（高度 + 颜色）
            mask = mask_height & mask_color   # 两个都是 (N,) 布尔数组

            print(f"{name} combined mask true: {np.count_nonzero(mask)}")

            # 过滤点云与颜色 —— 注意这里使用的是 pcd_world 和 rgb_uint8（或 color_valid）
            pcd_filtered = pcd_world[mask]
            pcd_color_filtered = rgb_uint8[mask]

            pts = pcd_filtered
            if len(pts) < 10:
                print(f"[WARN] {name} block not found or too few points ({len(pts)})")
                continue

            center = np.mean(pts, axis=0)
            pts_centered = pts - center
            cov = np.cov(pts_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvecs = eigvecs[:, order]
            R = eigvecs
            blocks[name] = {'pos': center, 'rot': R}
            print(f"[INFO] Detected {name}: center={center}, pts={len(pts)}")

        return blocks

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
        self.target_positions = {
            'red': [0.0, 0.0, 0.05],      # 红色方块放在中间
            'green': [0.0, 0.0, 0.10],    # 绿色方块放在红色方块上面
            'blue': [0.0, 0.0, 0.15]      # 蓝色方块放在绿色方块上面
        }
        
        # 抓取姿态文件路径
        self.grasp_save_path = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three/block_grasps.pickle"
        
        # 初始化夹爪和机器人
        self.gripper = pg.PiperGripper()
        
    def create_grasp_collection(self, obj_path, save_path):
        """为方块创建抓取姿态集合"""
        print("正在生成方块抓取姿态集合...")
        
        # 创建3D环境
        base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
        mgm.gen_frame().attach_to(base)
        
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
        grasp_collection = grasp_collection.limit(20)
        print(f"实际获取抓取数量: {len(grasp_collection)}")
        
        # 保存抓取姿态
        grasp_collection.save_to_disk(file_name=save_path)
        print(f"抓取姿态已保存到: {save_path}")
        
        return grasp_collection
    
    def execute_pick_place(self, block_pos, target_pos, block_rot, target_rot, arm_controller, robot):
        """执行单个方块的抓取和放置"""
        print(f"执行抓取和放置: 从 {block_pos} 到 {target_pos}")
        
        target_rot = rm.rotmat_from_euler(0, 0, 0)
        # 创建3D环境
        base = wd.World(cam_pos=[1.2, .7, 1], lookat_pos=[.0, 0, .15])
        mgm.gen_frame().attach_to(base)
        
        # 设置旋转矩阵
        start_rot = block_rot
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
        
        # 生成Pick and Place运动
        mot_data = ppp_planner.gen_pick_and_place(
            obj_cmodel=block_start,
            end_jnt_values=start_conf,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            pick_approach_direction=rm.const.z_ax,
            place_approach_distance_list=[.05] * len(goal_pose_list),
            place_depart_distance_list=[.05] * len(goal_pose_list),
            pick_approach_distance=.05,
            pick_depart_distance=.05,
            pick_depart_direction=rm.const.z_ax,
            obstacle_list=[],
            use_rrt=True
        )
        
        if mot_data is None:
            print("错误：无法生成Pick-and-Place运动轨迹！")
            return False
        
        print("Pick-and-Place运动轨迹生成成功！")
        
        # 导出关节角轨迹
        trajectory, traj_path = self.export_joint_trajectory(mot_data)
        
        # 执行运动
        self.execute_motion(arm_controller, traj_path)
        
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
    
    def execute_motion(self, arm_controller, json_path):
        """从JSON文件读取关节角轨迹并执行"""
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"未找到轨迹文件: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict) or 'joint_trajectory' not in data:
            raise ValueError("轨迹文件格式不正确，缺少 'joint_trajectory' 字段")

        trajectory = data['joint_trajectory']
        print(f"读取到 {len(trajectory)} 个关节轨迹点，将依次执行 move_j（6轴）...")

        for i, jv in enumerate(trajectory):
            print(f"执行第 {i+1}/{len(trajectory)} 个点: {jv}")
            arm_controller.move_j(jv[:6], speed=10)
            time.sleep(0.2)

            if jv[6] >= 0.08:
                gripper_width = 0.04
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
        
        # 1. 检测方块位置

        blocks = self.detector.detect_blocks()
        if not blocks:
            print("未检测到方块，任务终止")
            return False
        
        # 2. 创建抓取姿态集合（如果不存在）
        if not os.path.exists(self.grasp_save_path):
            self.create_grasp_collection(self.block_path, self.grasp_save_path)
        
        # 3. 按顺序执行堆叠任务
        task_order = ['red', 'green', 'blue']
        
        for i, color in enumerate(task_order):
            if color not in blocks:
                print(f"未检测到{color}方块，跳过")
                continue
            
            print(f"\n=== 执行第{i+1}步：抓取{color}方块 ===")
            
            # 获取方块位置
            block_pos = blocks[color]['pos']
            block_rot = blocks[color]['rot']
            target_pos = self.target_positions[color]
            target_rot = (0,0,0)
            # 选择机械臂
            arm_controller, robot = self.choose_arm(block_pos)
            
            # 执行抓取和放置
            success = self.execute_pick_place(block_pos, target_pos, block_rot, target_rot, arm_controller, robot)
            
            if success:
                print(f"{color}方块放置成功！")
            else:
                print(f"{color}方块放置失败！")
                return False
        
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