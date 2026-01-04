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
        from wrs.drivers.devices.realsense.realsense_d400 import *
        realsense = "d400"
    except Exception:
        print("Warning: RealSense drivers not available. Please install wrs package.")
        realsense = None


class MultiRobotSEMPolicy:
    """
    多机械臂SEM算法部署类
    
    支持多个机械臂和多个相机的SEM策略部署
    """
    
    def __init__(self, ckpt_path: str, config_file: str = "config_sem_robotwin.py"):
        """
        初始化多机械臂SEM策略
        
        Args:
            ckpt_path: 模型检查点路径
            config_file: 配置文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        self.cfg = self._load_config(config_file)
        
        # 构建数据变换
        _, self.transforms = self._build_transforms()
        # transforms现在已经是构建好的对象，不需要再次构建
            
        # 构建模型
        self.model = self._build_model()
        
        # 加载检查点
        self._load_checkpoint(ckpt_path)
        
        # 设置模型为评估模式
        self.model.eval()
        self.model.to(self.device)
        
        # 初始化状态
        self.action_history = []
        self.step_count = 0
        
        print(f"Multi-Robot SEM Policy initialized successfully on device: {self.device}")
    
    def _load_config(self, config_file: str):
        """加载配置文件"""
        assert config_file.endswith(".py")
        config_path = os.path.join(os.path.dirname(__file__), config_file)
        
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        return config_module.config
    
    def _build_transforms(self):
        """构建数据变换"""
        # 导入build_transforms函数
        import importlib.util
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 使用配置文件中的build_transforms函数
        train_transforms, val_transforms = config_module.build_transforms(self.cfg)
        return val_transforms, val_transforms
    
    def _build_model(self):
        """构建模型"""
        # 导入build_model函数
        import importlib.util
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), self.config_file)
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # 使用配置文件中的build_model函数
        model = config_module.build_model(self.cfg)
        return model
    
    def _load_checkpoint(self, ckpt_path: str):
        """加载检查点"""
        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path)
        else:
            state_dict = torch.load(ckpt_path, map_location=self.device)
        
        # 处理不同的检查点格式
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {ckpt_path}")
    
    def encode_multi_robot_obs(self, 
                              images: List[np.ndarray],
                              depths: List[np.ndarray], 
                              joint_positions: Dict[str, np.ndarray],
                              camera_intrinsics: List[np.ndarray],
                              camera_extrinsics: List[np.ndarray],
                              instruction: str,
                              base_to_world_transforms: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        编码多机器人观测数据
        
        Args:
            images: 图像列表
            depths: 深度图像列表
            joint_positions: 各机械臂关节位置字典
            camera_intrinsics: 相机内参列表
            camera_extrinsics: 相机外参列表
            instruction: 任务指令
            base_to_world_transforms: 各机械臂基座到世界坐标系的变换
            
        Returns:
            编码后的观测数据
        """
        # 处理图像数据
        processed_images = []
        for img in images:
            if img is not None:
                # 确保图像格式正确
                if len(img.shape) == 3 and img.shape[2] == 3:
                    processed_images.append(img)
                else:
                    print(f"Warning: Invalid image shape {img.shape}")
                    processed_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            else:
                processed_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # 处理深度数据
        processed_depths = []
        for depth in depths:
            if depth is not None:
                processed_depths.append(depth)
            else:
                processed_depths.append(np.zeros((480, 640), dtype=np.float32))
        
        # 合并所有机械臂的关节位置
        all_joint_positions = []
        for arm_name, joints in joint_positions.items():
            if joints is not None:
                all_joint_positions.extend(joints.tolist())
            else:
                # 填充默认值
                all_joint_positions.extend([0.0] * 7)  # 假设每个机械臂7个关节
        
        # 合并所有变换矩阵
        all_transforms = []
        for arm_name, transform in base_to_world_transforms.items():
            if transform is not None:
                all_transforms.extend(transform.flatten().tolist())
            else:
                # 填充单位矩阵
                all_transforms.extend(np.eye(4).flatten().tolist())
        
        # 构建投影矩阵
        projection_mats = self._build_projection_matrices(
            camera_intrinsics, camera_extrinsics, base_to_world_transforms
        )
        
        # 构建观测数据
        obs_data = {
            'images': processed_images,
            'depths': processed_depths,
            'joint_positions': np.array(all_joint_positions),
            'camera_intrinsics': camera_intrinsics,
            'camera_extrinsics': camera_extrinsics,
            'instruction': instruction,
            'base_to_world_transforms': np.array(all_transforms),
            'projection_mat': projection_mats,
            'image_wh': np.array([[320, 256]] * len(processed_images))  # 假设图像尺寸为320x256
        }
        
        return obs_data
    
    def _build_projection_matrices(self, 
                                  camera_intrinsics: List[np.ndarray],
                                  camera_extrinsics: List[np.ndarray], 
                                  base_to_world_transforms: Dict[str, np.ndarray]) -> np.ndarray:
        """
        构建投影矩阵，用于将3D点投影到像素坐标
        
        Args:
            camera_intrinsics: 相机内参列表 [3x3]
            camera_extrinsics: 相机外参列表 [4x4] (世界坐标系到相机坐标系)
            base_to_world_transforms: 基座到世界坐标系的变换字典 [4x4]
            
        Returns:
            投影矩阵 [batch_size, num_cameras, 3, 4]
            用于求解: projection_mat @ [x, y, z, 1]^T = [u, v, w]^T
        """
        num_cameras = len(camera_intrinsics)
        projection_mats = []
        
        for i in range(num_cameras):
            # 获取相机内参 [3x3]
            intrinsic = camera_intrinsics[i] if camera_intrinsics[i] is not None else np.eye(3)
            
            # 获取相机外参 [4x4] (世界坐标系到相机坐标系)
            extrinsic = camera_extrinsics[i] if camera_extrinsics[i] is not None else np.eye(4)
            
            # 获取基座到世界坐标系的变换 [4x4]
            arm_names = list(base_to_world_transforms.keys())
            if i < len(arm_names):
                base_to_world = base_to_world_transforms[arm_names[i]]
            else:
                base_to_world = np.eye(4)
            
            # 构建投影矩阵: P = K @ T_world2cam @ T_base2world
            # 其中:
            # - K: 相机内参矩阵 [3x3]
            # - T_world2cam: 世界坐标系到相机坐标系的变换 [3x4] (取extrinsic前3行)
            # - T_base2world: 基座坐标系到世界坐标系的变换 [4x4]
            T_world2cam = extrinsic[:3, :]  # [3x4]
            T_base2world = base_to_world    # [4x4]
            
            # 投影矩阵计算: P = K @ T_world2cam @ T_base2world
            # 这个矩阵用于将基座坐标系中的3D点投影到像素坐标
            projection_mat = intrinsic @ T_world2cam @ T_base2world  # [3x4]
            
            projection_mats.append(projection_mat)
        
        # 转换为numpy数组并添加batch维度
        # 形状: [batch_size, num_cameras, 3, 4]
        projection_mats = np.array(projection_mats)  # [num_cameras, 3, 4]
        projection_mats = projection_mats[None, ...]  # [1, num_cameras, 3, 4]
        
        return projection_mats
    
    def predict_action(self, 
                      images: List[np.ndarray],
                      depths: List[np.ndarray],
                      joint_positions: Dict[str, np.ndarray],
                      camera_intrinsics: List[np.ndarray],
                      camera_extrinsics: List[np.ndarray],
                      instruction: str,
                      base_to_world_transforms: Dict[str, np.ndarray]) -> np.ndarray:
        """
        预测动作
        
        Args:
            images: 图像列表
            depths: 深度图像列表
            joint_positions: 各机械臂关节位置字典
            camera_intrinsics: 相机内参列表
            camera_extrinsics: 相机外参列表
            instruction: 任务指令
            base_to_world_transforms: 各机械臂基座到世界坐标系的变换
            
        Returns:
            预测的动作序列
        """
        # 编码观测数据
        obs_data = self.encode_multi_robot_obs(
            images, depths, joint_positions, camera_intrinsics, 
            camera_extrinsics, instruction, base_to_world_transforms
        )
        
        # 应用数据变换
        if self.transforms is not None:
            for transform in self.transforms:
                obs_data = transform(obs_data)
        
        # 转换为张量
        obs_tensor = self._obs_to_tensor(obs_data)
        
        # 推理
        with torch.no_grad():
            obs_tensor = obs_tensor.to(self.device)
            actions = self.model(obs_tensor)
            
            if isinstance(actions, dict):
                actions = actions['actions']
            
            # 转换为numpy数组
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            
            # 确保输出格式正确
            if len(actions.shape) == 1:
                actions = actions.reshape(1, -1)
        
        # 更新历史
        self.action_history.append(actions[0])
        self.step_count += 1
        
        return actions
    
    def _obs_to_tensor(self, obs_data: Dict[str, Any]) -> torch.Tensor:
        """将观测数据转换为张量"""
        # 这里需要根据具体的模型输入格式来实现
        # 暂时返回一个占位符
        return torch.randn(1, 100)  # 占位符
    
    def reset(self):
        """重置策略状态"""
        self.action_history = []
        self.step_count = 0


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


def get_color_image(cam):
    """获取彩色图像"""
    try:
        if realsense == "d400s":
            return cam.get_color_image()
        else:
            return cam.get_color_image()
    except Exception as e:
        print(f"Error getting color image: {e}")
        return None


def get_depth_image(cam):
    """获取深度图像"""
    try:
        if realsense == "d400s":
            return cam.get_depth_image()
        else:
            return cam.get_depth_image()
    except Exception as e:
        print(f"Error getting depth image: {e}")
        return None


def get_camera_intrinsics(cam):
    """获取相机内参"""
    try:
        if realsense == "d400s":
            return cam.get_intrinsics()
        else:
            return cam.get_intrinsics()
    except Exception as e:
        print(f"Error getting camera intrinsics: {e}")
        return np.eye(3, dtype=np.float32)


def get_camera_extrinsics(cam):
    """获取相机外参"""
    try:
        if realsense == "d400s":
            return cam.get_extrinsics()
        else:
            return cam.get_extrinsics()
    except Exception as e:
        print(f"Error getting camera extrinsics: {e}")
        return np.eye(4, dtype=np.float32)


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
        
        # 默认相机配置
        if self.camera_config is None:
            self.camera_config = {
                'head_camera': {'ID': 'head_cam'},
                'left_hand_camera': {'ID': 'left_hand_cam'},
                'right_hand_camera': {'ID': 'right_hand_cam'}
            }
        
        # 相机角色映射
        self.camera_roles = {
            'head': self.camera_config['head_camera']['ID'],
            'left_hand': self.camera_config['left_hand_camera']['ID'],
            'right_hand': self.camera_config['right_hand_camera']['ID']
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
                    'gripper_opening': 0.0,
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
    
    def get_joint_positions(self) -> Dict[str, np.ndarray]:
        """
        获取所有机械臂的关节位置
        
        Returns:
            各机械臂关节位置字典
        """
        joint_positions = {}
        
        for arm_name, arm in self.arms.items():
            try:
                joints = arm.get_joint_positions()
                if joints is not None:
                    joint_positions[arm_name] = np.array(joints)
                else:
                    joint_positions[arm_name] = np.zeros(7)  # 默认7个关节
            except Exception as e:
                print(f"Error getting joint positions for {arm_name}: {e}")
                joint_positions[arm_name] = np.zeros(7)
        
        return joint_positions
    
    def get_camera_data(self, camera_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        获取相机数据
        
        Args:
            camera_names: 相机名称列表
            
        Returns:
            相机数据字典
        """
        camera_data = {}
        
        for cam_name in camera_names:
            try:
                if cam_name in self.cameras:
                    cam = self.cameras[cam_name]
                    
                    # 获取RGB图像
                    rgb_img = get_color_image(cam)
                    if rgb_img is not None and rgb_img.dtype == np.float32:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    
                    # 获取深度图像
                    depth_img = get_depth_image(cam)
                    
                    # 获取相机参数
                    intrinsic = get_camera_intrinsics(cam)
                    extrinsic = get_camera_extrinsics(cam)
                    
                    camera_data[cam_name] = {
                        'rgb': rgb_img if rgb_img is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                        'depth': depth_img if depth_img is not None else np.zeros((480, 640), dtype=np.float32),
                        'intrinsic': intrinsic,
                        'extrinsic': extrinsic
                    }
                else:
                    print(f"Warning: Camera {cam_name} not available")
                    camera_data[cam_name] = {
                        'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                        'depth': np.zeros((480, 640), dtype=np.float32),
                        'intrinsic': np.eye(3, dtype=np.float32),
                        'extrinsic': np.eye(4, dtype=np.float32)
                    }
                    
            except Exception as e:
                print(f"Error getting camera data for {cam_name}: {e}")
                camera_data[cam_name] = {
                    'rgb': np.zeros((480, 640, 3), dtype=np.uint8),
                    'depth': np.zeros((480, 640), dtype=np.float32),
                    'intrinsic': np.eye(3, dtype=np.float32),
                    'extrinsic': np.eye(4, dtype=np.float32)
                }
        
        return camera_data
    
    def execute_action(self, actions: Dict[str, List[float]]) -> bool:
        """
        执行动作
        
        Args:
            actions: 各机械臂动作字典
            
        Returns:
            执行是否成功
        """
        success = True
        
        for arm_name, action in actions.items():
            if arm_name in self.arms:
                try:
                    arm = self.arms[arm_name]
                    
                    # 执行动作（这里需要根据具体的动作格式来实现）
                    if len(action) >= 6:
                        # 假设前6个是位置和旋转
                        position = action[:3]
                        rotation = action[3:6]
                        
                        # 转换为机械臂坐标系
                        position_robot = self._world_to_robot_coords(position, arm_name)
                        rotation_robot = self._world_to_robot_coords(rotation, arm_name)
                        
                        # 执行动作
                        arm.move_to_position(position_robot, rotation_robot)
                    else:
                        print(f"Warning: Invalid action format for {arm_name}")
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
                    arm.move_to_joint_positions(joint_positions)
                    
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
    # 配置参数

    #可选
    ckpt_path = "path/to/your/model.safetensors"  # 替换为实际的模型路径

    #可选
    config_file = "/home/fafurobot/wrs_tiaozhanbei/wrs/robot_con/piper/policy/smolvla/sem/config_sem_robotwin.py"  # SEM配置文件路径

    #可选

    #可选
    instruction = "place the empty cup on the table"  # 任务指令

    #可选
    camera_config_path = "/home/fafurobot/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml"  # 相机配置文件路径

    #可选
    home_state_path = "/home/fafurobot/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/home_state1.yaml"  # 零位状态配置文件路径
    


    # 多机械臂配置 可以修改can name
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
    camera_names = ["head", "left_hand", "right_hand"]
    
    # 控制参数
    step_hz = 5.0  # 控制频率
    dt = 1.0 / step_hz
    max_xyz_step = 0.01  # 每步最大位置增量 (m)
    max_rpy_step = np.deg2rad(2.0)  # 每步最大旋转增量 (rad)
    
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
        
        # 等待到达零位
        # if not robot_interface.wait_for_home_position(timeout=30.0):
        #     print("Timeout waiting for home position.")
        #     return
        
        print("All robots are now at home position. Ready for inference.")
        
        # 初始化多机械臂SEM策略
        print("Initializing Multi-Robot SEM policy...")
        policy = MultiRobotSEMPolicy(ckpt_path, config_file)
        
        # 主循环
        print("Starting inference loop...")
        print(f"Control frequency: {step_hz} Hz")
        print(f"Max position step: {max_xyz_step} m")
        print(f"Max rotation step: {np.rad2deg(max_rpy_step)} deg")
        print("Press Ctrl+C to stop...")
        
        step_count = 0
        max_steps = 1000  # 最大执行步数
        
        while step_count < max_steps:
            try:
                print(f"\nStep {step_count + 1}/{max_steps}")
                
                # 获取机器人状态
                joint_positions = robot_interface.get_joint_positions()
                camera_data = robot_interface.get_camera_data(camera_names)
                base_to_world_transforms = robot_interface.get_base_to_world_transform()
                
                # 准备数据
                images = [camera_data[cam_name]['rgb'] for cam_name in camera_names]
                depths = [camera_data[cam_name]['depth'] for cam_name in camera_names]
                camera_intrinsics = [camera_data[cam_name]['intrinsic'] for cam_name in camera_names]
                camera_extrinsics = [camera_data[cam_name]['extrinsic'] for cam_name in camera_names]
                
                # 预测动作
                start_time = time.time()
                actions = policy.predict_action(
                    images=images,
                    depths=depths,
                    joint_positions=joint_positions,
                    camera_intrinsics=camera_intrinsics,
                    camera_extrinsics=camera_extrinsics,
                    instruction=instruction,
                    base_to_world_transforms=base_to_world_transforms
                )
                inference_time = time.time() - start_time
                
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Predicted actions shape: {actions.shape}")
                
                # 执行动作（需要将动作分配给不同的机械臂）
                if len(actions) > 0:
                    first_action = actions[0].tolist()
                    
                    # 限幅处理（避免大步跳变）
                    if len(first_action) >= 6:
                        # 位置限幅
                        first_action[:3] = np.clip(first_action[:3], -max_xyz_step, max_xyz_step)
                        # 旋转限幅
                        first_action[3:6] = np.clip(first_action[3:6], -max_rpy_step, max_rpy_step)
                    
                    # 将动作分配给不同的机械臂
                    arm_actions = {
                        'left_arm': first_action[:6],  # 前6个参数给左臂
                        'right_arm': first_action[6:12] if len(first_action) >= 12 else [0] * 6  # 后6个参数给右臂
                    }
                    
                    success = robot_interface.execute_action(arm_actions)
                    
                    if not success:
                        print("Failed to execute action.")
                        break
                    
                    print(f"Actions executed: {arm_actions}")
                
                # 控制频率
                time.sleep(dt)
                step_count += 1
                
            except KeyboardInterrupt:
                print("\nStopping inference loop...")
                break
            except Exception as e:
                print(f"Error in step {step_count + 1}: {e}")
                import traceback
                traceback.print_exc()
                # 继续执行，不中断循环
                time.sleep(1.0)
                step_count += 1
        
        print("Inference loop completed.")
        
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
