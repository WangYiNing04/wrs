# Project RoboOrchard
#
# Copyright (c) 2024-2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
'''
需要wrs库
    export PYTHONPATH=$PYTHONPATH:~/PycharmProjects/wrs_tiaozhanbei
'''
import copy
import importlib
import os
import time
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import cv2

# from robo_orchard_lab.utils.build import build  # 不再需要，直接使用配置文件中的函数

# 导入真实机器人接口
try:
    from wrs.robot_con.piper.piper import PiperArmController
    PIPER_AVAILABLE = True
except ImportError:
    print("Warning: PiperArmController not available. Please install wrs package.")
    PIPER_AVAILABLE = False

# RealSense 相机接口
realsense = None
try:
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.drivers.devices.realsense.realsense_d400s import *
    realsense = "d400s"
except Exception:
    try:
        from wrs.drivers.devices.realsense.realsense_d400s import *
        realsense = "d400"
    except Exception:
        print("Warning: RealSense drivers not available. Please install wrs package.")
        realsense = None


def load_camera_extrinsics(config_path: str):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    extrinsics = {}
    for name, params in data.items():
        extrinsics[name] = np.array(params['extrinsic'], dtype=np.float32)
    return extrinsics

# 工具函数
def rotmat_to_euler_xyz(R):
    """旋转矩阵转欧拉角 (XYZ顺序)"""
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


def euler_to_rotmat(euler):
    """欧拉角转旋转矩阵 (XYZ顺序)"""
    x, y, z = euler
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]
    ])
    
    Ry = np.array([
        [math.cos(y), 0, math.sin(y)],
        [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]
    ])
    
    Rz = np.array([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


class MultiRobotInterface:
    """
    多机械臂机器人接口实现
    
    支持多个机械臂和多个相机的真实机器人接口
    """
    
    def __init__(self, robot_config: Dict[str, Any], camera_config_path: Optional[str] = None, home_state_path: Optional[str] = None):
        """
        初始化多机器人接口
        
        Args:
            robot_config: 机械臂配置字典
            camera_config_path: 相机配置文件路径
            home_state_path: 零位状态配置文件路径
        """
        self.is_connected = False
        self.arms = {}  # 存储多个机械臂
        self.cameras = {}
        self.camera_config = None
        self.robot_config = robot_config
        self.home_state = None
        self.home_state_path = home_state_path
  
        
        # 加载相机配置
        if camera_config_path and os.path.exists(camera_config_path):
            with open(camera_config_path, 'r') as file:
                self.camera_config = yaml.safe_load(file)

        # 挑战杯piper臂的三个相机:
        # head_camera:
        #   ID: '243322073422'
        # left_hand_camera:
        #   ID: '243322074546'
        # right_hand_camera:
        #   ID: '243322071033'

        # 默认相机配置
        if self.camera_config is None:
            self.camera_config = {
                'middle_camera': {'ID': '243322073422'},
                'left_camera': {'ID': '243322074546'},
                'right_camera': {'ID': '243322071033'}
            }
        
        # 相机角色映射
        self.camera_roles = {
            'middle': self.camera_config['middle_camera']['ID'],
            'left': self.camera_config['left_camera']['ID'],
            'right': self.camera_config['right_camera']['ID']
        }
        
        # 加载零位状态配置
        if self.home_state_path and os.path.exists(self.home_state_path):
            with open(self.home_state_path, 'r') as file:
                self.home_state = yaml.safe_load(file)
                print(f"Loaded home state from {self.home_state_path}")
        else:
            print("Warning: Home state file not found. Using default home positions.")
            # 默认零位状态
            self.home_state = {
                'left_arm': {
                    'gripper_effort': 0,
                    'gripper_opening': 0.2,
                    'joint_positions': [-0.04260348704118158, 0.12138764947620562, -0.005899212871740834, 
                                       0.11210249785559578, -0.04365068459237818, -0.20212658067346329]
                },
                
                'right_arm': {
                    'gripper_effort': 0,
                    'gripper_opening': 0.0,
                    'joint_positions': [-0.06995279641993273, 0.08845328649107262, -0.002530727415391778,
                                       0.0, -0.01996656664281513, 0.015742869852988853]
                }
            }
        
            # self.home_state = {
            #     'left_arm': {
            #         'gripper_effort': 0,
            #         'gripper_opening': 0.2,
            #         'joint_positions':[0.0,0.7656061296798325, -1.1507130291323813, -0.018779742751458987, 1.2317137597174384, 0.31133183197074854]
            #     },

            #     'right_arm': {
            #         'gripper_effort': 0,
            #         'gripper_opening': 0.2,
            #         'joint_positions':[-0.011903145498601329,0.7309264374427052, -1.1733499495307478, 0.0965865208053662, 1.2306840154587617, -0.019355701404617114]
            #     }
            # }


    def connect(self) -> bool:
        """
        连接到多个机器人
        
        Returns:
            连接是否成功
        """
        if not PIPER_AVAILABLE:
            print("Error: PiperArmController not available. Cannot connect to robot.")
            return False
        
        try:
            print("Connecting to multiple robots...")
            
            # 初始化多个机械臂
            for arm_name, arm_config in self.robot_config['arms'].items():
                print(f"Initializing {arm_name} arm...")
                can_name = arm_config.get('can_name', f'can{arm_name}')
                has_gripper = arm_config.get('has_gripper', True)
                auto_enable = arm_config.get('auto_enable', True)
                
                arm = PiperArmController(
                    can_name=can_name,
                    has_gripper=has_gripper,
                    auto_enable=auto_enable
                )
                self.arms[arm_name] = arm
                print(f"{arm_name} arm initialized successfully.")
            
            # 初始化相机
            if realsense is None:
                print("Warning: RealSense drivers not available. Camera functionality will be limited.")
            else:
                self._init_cameras()
            
            self.is_connected = True
            print("All robots connected successfully.")
            return True
            
        except Exception as e:
            print(f"Failed to connect to robots: {e}")
            return False
    
    def _init_cameras(self):
        """初始化相机"""
        try:
            # 查找实际连接的设备
            available_serials, ctx = find_devices()
            print("检测到设备:", available_serials)
            
            # 初始化相机（用字典存储，键为角色名称）
            for role, cam_id in self.camera_roles.items():
                if cam_id in available_serials:
                    print(f"正在初始化 {role} 相机 (ID: {cam_id})")
                    pipeline = RealSenseD400(device=cam_id)
                    pipeline.reset()
                    time.sleep(2)
                    pipeline = RealSenseD400(device=cam_id)  # 重新初始化
                    self.cameras[role] = pipeline
                else:
                    print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")
                    
        except Exception as e:
            print(f"Failed to initialize cameras: {e}")
    
    def disconnect(self):
        """断开机器人连接"""
        print("Disconnecting from robots...")
        
        # 断开所有机械臂
        for arm_name, arm in self.arms.items():
            try:
                arm.disable()
                print(f"{arm_name} arm disconnected.")
            except Exception as e:
                print(f"Error disconnecting {arm_name} arm: {e}")
        
        # 断开所有相机
        for cam_name, cam in self.cameras.items():
            try:
                cam.stop()
                print(f"{cam_name} camera disconnected.")
            except Exception as e:
                print(f"Error disconnecting {cam_name} camera: {e}")
        
        self.arms.clear()
        self.cameras.clear()
        self.is_connected = False
        print("All robots disconnected.")
    
    def get_joint_positions(arm):
        """获取单个机械臂的关节位置"""
        try:
            joint_positions = arm.get_joint_values()
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            joint_positions = None
        return joint_positions
    
    def get_color_image(self,cam_name):
        """获取单个相机彩色图像"""
        if cam_name not in self.cameras:
            raise ValueError(f"Camera {cam_name} not found. Available cameras: {list(self.cameras.keys())}")
        cam = self.cameras[cam_name] 
        try:
            if realsense == "d400s":
                return cam.get_color_img()
            else:
                return cam.get_color_img()
        except Exception as e:
            print(f"Error getting color image: {e}")
            return None


    def get_depth_image(self,cam_name):
        """获取单个相机深度图像"""
        if cam_name not in self.cameras:
            raise ValueError(f"Camera {cam_name} not found. Available cameras: {list(self.cameras.keys())}")
        cam = self.cameras[cam_name] 
        try:
            if realsense == "d400s":
                return cam.get_depth_img()
            else:
                return cam.get_depth_img()
        except Exception as e:
            print(f"Error getting depth image: {e}")
            return None


    def get_camera_intrinsics(self, cam_name):
        """获取单个相机相机内参，返回4×4矩阵"""
        if cam_name not in self.cameras:
            raise ValueError(f"Camera {cam_name} not found. Available cameras: {list(self.cameras.keys())}")
        
        cam = self.cameras[cam_name] 
        try:
            if realsense == "d400s":
                intrinsic = cam.intr_mat
            else:
                intrinsic = cam.intr_mat
            
            # 打印原始内参矩阵信息
            #print(f"Camera {cam_name} original intrinsic shape: {intrinsic.shape}")
            #print(f"Camera {cam_name} original intrinsic:\n{intrinsic}")
            
            # 如果内参矩阵是3x3，补齐为4x4
            if intrinsic.shape == (3, 3):
                # 创建4x4单位矩阵
                intrinsic_4x4 = np.eye(4, dtype=intrinsic.dtype)
                # 将3x3内参矩阵复制到左上角
                intrinsic_4x4[:3, :3] = intrinsic
                intrinsic = intrinsic_4x4
                #print(f"Converted to 4x4 intrinsic:\n{intrinsic}")
            elif intrinsic.shape != (4, 4):
                #print(f"Warning: Unexpected intrinsic shape {intrinsic.shape} for camera {cam_name}")
                # 创建默认4x4内参矩阵
                intrinsic = np.eye(4, dtype=np.float32)
            
            return intrinsic
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
            # 返回默认4x4内参矩阵
            return np.eye(4, dtype=np.float32)

    def get_camera_extrinsics(self, cam_name):
        """获取单个相机相机外参"""
        # if cam_name not in self.cameras:
        #     raise ValueError(f"Camera {cam_name} not found. Available cameras: {list(self.cameras.keys())}")       
        try:
            # 直接使用相机名称获取外参
            camera_extrinsics = load_camera_extrinsics('/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/config/camera_extrinsics_calib2.yaml')
            #print(camera_extrinsics)
            #print(cam_name)
            # 检查相机名称是否在配置中
            if cam_name in camera_extrinsics:
                extrinsic_matrix = camera_extrinsics[cam_name]
                return np.array(extrinsic_matrix, dtype=np.float32)
            else:
                print(f"Warning: Extrinsics not found for camera {cam_name}. Using identity matrix.")
                return np.eye(4, dtype=np.float32)
                
        except Exception as e:
            print(f"Error getting camera extrinsics: {e}")
            return np.eye(4, dtype=np.float32)

 
    # def resize_image(image, target_size=(320, 256)):
    #     """调整图像尺寸到目标大小"""
    #     return cv2.resize(image, target_size)

    # def resize_depth(depth, target_size=(320, 256)):
    #     """调整深度图尺寸到目标大小"""
    #     return cv2.resize(depth, target_size)

    
    def execute_action(self, actions: Dict[str, List[float]]) -> bool:
        """
        执行动作（包含关节和夹爪控制）
        
        Args:
            actions: 各机械臂动作字典，每个动作包含7个元素（6个关节+1个夹爪）
                
        Returns:
            执行是否成功
        """
        success = True
        
        for arm_name, action in actions.items():
            if arm_name in self.arms:
                try:
                    arm = self.arms[arm_name]
                    
                    # 检查动作格式
                    if len(action) != 7:
                        print(f"Warning: Invalid action format for {arm_name}. Expected 7 elements, got {len(action)}")
                        success = False
                        continue
                    
                    # 提取关节位置（前6个元素）
                    joint_positions = action[:6]
                    
                    # 提取夹爪开合度（第7个元素）
                    gripper_opening = action[6]
                    
                    # 打印调试信息
                    print(f"Executing {arm_name} action:")
                    print(f"  Joint positions: {joint_positions}")
                    print(f"  Gripper opening: {gripper_opening}")
                    
                    # 执行关节动作
                    try:
                        arm.move_j(joint_positions, speed=10, block=False)
                        print(f"  Joint movement command sent successfully")
                    except Exception as e:
                        print(f"  Error executing joint movement: {e}")
                        success = False
                    
                    # 执行夹爪动作
                    try:
                        arm.gripper_control(angle= gripper_opening, effort=0, enable=True)
                    except Exception as e:
                        print(f"  Error executing gripper movement: {e}")
                        success = False
                        
                except Exception as e:
                    print(f"Error executing action for {arm_name}: {e}")
                    success = False
            else:
                print(f"Warning: Arm {arm_name} not available")
                success = False
        
        return success
    
    def _world_to_robot_coords(self, coords: np.ndarray, arm_name: str) -> np.ndarray:
        """
        世界坐标系到机械臂坐标系的转换
        
        Args:
            coords: 世界坐标系下的坐标
            arm_name: 机械臂名称
            
        Returns:
            机械臂坐标系下的坐标
        """
        # 这里需要根据具体的机械臂配置来实现坐标变换
        # 暂时返回原坐标
        return coords
    
    def get_base_to_world_transform(self) -> Dict[str, np.ndarray]:
        """
        获取各机械臂基座到世界坐标系的变换
        
        Returns:
            各机械臂变换矩阵字典
        """
        transforms = {}
        
        for arm_name in self.arms.keys():
            # 这里需要根据具体的机械臂配置来获取变换矩阵
            # 暂时返回单位矩阵
            transforms[arm_name] = np.array([
                [0, -1, 0, 0],
                [1, 0, 0, -0.65],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        
        return transforms
    
    def move_to_home_position(self) -> bool:
        """
        移动所有机械臂到零位
        
        Returns:
            是否成功移动到零位
        """
        if not self.is_connected:
            print("Error: Robot not connected. Cannot move to home position.")
            return False
        
        success = True
        print("Moving all arms to home position...")
        
        for arm_name, arm in self.arms.items():
            try:
                if arm_name in self.home_state:
                    home_config = self.home_state[arm_name]
                    joint_positions = home_config['joint_positions']
                    gripper_opening = home_config['gripper_opening']
                    
                    print(f"Moving {arm_name} to home position...")
                    print(f"Target joint positions: {joint_positions}")
                    
                    # 移动到零位关节位置
                    arm.move_j(joint_positions,speed=10,block=False)
                    
                    # 设置夹爪状态
                    if hasattr(arm, 'set_gripper_opening'):
                        arm.set_gripper_opening(gripper_opening)
                    elif hasattr(arm, 'set_gripper_position'):
                        arm.set_gripper_position(gripper_opening)
                    
                    print(f"{arm_name} moved to home position successfully.")
                    
                else:
                    print(f"Warning: No home position configured for {arm_name}")
                    success = False
                    
            except Exception as e:
                print(f"Error moving {arm_name} to home position: {e}")
                success = False
        
        if success:
            print("All arms moved to home position successfully.")
        else:
            print("Some arms failed to move to home position.")
        
        return success
    
    def wait_for_home_position(self, timeout: float = 30.0) -> bool:
        """
        等待所有机械臂到达零位
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否成功到达零位
        """
        if not self.is_connected:
            return False
        
        print("Waiting for all arms to reach home position...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_at_home = True
            
            for arm_name, arm in self.arms.items():
                try:
                    if arm_name in self.home_state:
                        current_joints = arm.get_joint_positions()
                        target_joints = self.home_state[arm_name]['joint_positions']
                        
                        # 检查是否到达目标位置（允许小的误差）
                        joint_error = np.abs(np.array(current_joints) - np.array(target_joints))
                        max_error = np.max(joint_error)
                        
                        if max_error > 0.01:  # 1cm的误差阈值
                            all_at_home = False
                            break
                            
                except Exception as e:
                    print(f"Error checking {arm_name} position: {e}")
                    all_at_home = False
                    break
            
            if all_at_home:
                print("All arms reached home position.")
                return True
            
            time.sleep(0.1)  # 等待100ms后再次检查
        
        print(f"Timeout waiting for home position after {timeout}s")
        return False

def main():
    """
    主函数 - 演示如何使用多机械臂SEM策略
    """
    # head_camera:
    #   ID: '243322073422'
    # left_hand_camera:
    #   ID: '243322074546'
    # right_hand_camera:
    #   ID: '243322071033'
    # 机械臂配置参数
    camera_config_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml"  # 相机配置文件路径
    home_state_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/home_state2.yaml"  # 零位状态配置文件路径
    
    # 多机械臂配置
    robot_config = {
        'arms': {
            'left_arm': {
                'can_name': 'can0',
                'has_gripper': True,
                'auto_enable': True
            },
            'right_arm': {
                'can_name': 'can1',
                'has_gripper': True,
                'auto_enable': True
            }
        }
    }
    
    # 相机名称列表（使用所有3个相机）
    #camera_names = ["head", "left_hand", "right_hand"]
    

    try:
        # 初始化多机器人接口
        robot_interface = MultiRobotInterface(robot_config, camera_config_path, home_state_path)
        
        # 连接机器人
        if not robot_interface.connect():
            print("Failed to connect to robots.")
            return
        
        # 移动到零位
        print("Moving robots to home position before inference...")
        if not robot_interface.move_to_home_position():
            print("Failed to move to home position.")
            return
        
        # # 等待到达零位
        # if not robot_interface.wait_for_home_position(timeout=30.0):
        #     print("Timeout waiting for home position.")
        #     return
        
        print("All robots are now at home position. Ready for inference.")
           
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'robot_interface' in locals():
            robot_interface.disconnect()
        print("Cleanup completed.")

if __name__ == "__main__":
    main()