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

from robo_orchard_lab.utils.build import build

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


class RealRobotSEMPolicy:
    """
    SEM算法在真实机器人上的部署类
    
    该类基于原始的SEMPolicy类，但针对真实机器人环境进行了适配：
    1. 移除了对RoboTwin环境的依赖
    2. 提供了真实机器人数据接口的模板
    3. 支持实时推理和动作执行
    """
    
    def __init__(self, ckpt_path: str, config_file: str = "config_sem_robotwin.py"):
        """
        初始化SEM策略
        
        Args:
            ckpt_path: 模型检查点路径
            config_file: 配置文件路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        self.cfg = self._load_config(config_file)
        
        # 构建数据变换
        _, self.transforms = self._build_transforms()
        if self.transforms is not None:
            self.transforms = [build(x) for x in self.transforms]
            
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
        
        print(f"SEM Policy initialized successfully on device: {self.device}")
    
    def _load_config(self, config_file: str):
        """加载配置文件"""
        assert config_file.endswith(".py")
        module_name = os.path.split(config_file)[-1][:-3]
        spec = importlib.util.spec_from_file_location(module_name, config_file)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config
    
    def _build_transforms(self):
        """构建数据变换"""
        if hasattr(self.cfg, "build_transforms"):
            print("Building transforms...")
            train_transforms, val_transforms = self.cfg.build_transforms(self.cfg.config)
            return train_transforms, val_transforms
        else:
            print("Warning: Config file does not contain a 'build_transforms' function.")
            return None, None
    
    def _build_model(self):
        """构建模型"""
        if hasattr(self.cfg, "build_model"):
            print("Building model...")
            model = self.cfg.build_model(self.cfg.config)
            return model
        else:
            raise AttributeError("Config file must contain a 'build_model' function.")
    
    def _load_checkpoint(self, ckpt_path: str):
        """加载模型检查点"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        print(f"Loading checkpoint from: {ckpt_path}")
        try:
            from safetensors.torch import load_model as load_safetensors
            load_safetensors(self.model, ckpt_path, device=str(self.device))
        except ImportError:
            print("safetensors not found, using torch.load.")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            print(f"num of missing_keys: {len(missing_keys)}, num of unexpected_keys: {len(unexpected_keys)}")
            if len(missing_keys) > 0:
                print(f"missing_keys:\n {missing_keys}")
            if len(unexpected_keys) > 0:
                print(f"unexpected_keys:\n {unexpected_keys}")
        
        print("Checkpoint loaded successfully.")
    
    def encode_real_robot_obs(self, 
                             images: List[np.ndarray], 
                             depths: List[np.ndarray],
                             joint_positions: List[float],
                             camera_intrinsics: List[np.ndarray],
                             camera_extrinsics: List[np.ndarray],
                             instruction: str,
                             base_to_world_transform: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        编码真实机器人的观测数据
        
        Args:
            images: RGB图像列表 (H, W, 3)
            depths: 深度图像列表 (H, W) - 单位：米
            joint_positions: 关节位置列表 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            camera_intrinsics: 相机内参矩阵列表 (3, 3)
            camera_extrinsics: 相机外参矩阵列表 (4, 4) - 世界坐标系到相机坐标系
            instruction: 任务指令文本
            base_to_world_transform: 机器人基座到世界坐标系的变换矩阵 (4, 4)
            
        Returns:
            编码后的数据字典
        """
        # 确保输入数据格式正确
        assert len(images) == len(depths) == len(camera_intrinsics) == len(camera_extrinsics)
        assert len(joint_positions) == 14, f"Expected 14 joint positions, got {len(joint_positions)}"
        
        # 处理图像数据
        processed_images = []
        processed_depths = []
        
        for img, depth in zip(images, depths):
            # 确保图像格式正确
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img)
            
            # 处理深度图像
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            processed_depths.append(depth)
        
        # 堆叠图像数据
        images_stack = np.stack(processed_images)
        depths_stack = np.stack(processed_depths)
        
        # 处理相机参数
        intrinsics = []
        extrinsics = []
        
        for intrinsic, extrinsic in zip(camera_intrinsics, camera_extrinsics):
            # 扩展内参矩阵到4x4
            intrinsic_4x4 = np.eye(4, dtype=np.float32)
            intrinsic_4x4[:3, :3] = intrinsic.astype(np.float32)
            intrinsics.append(intrinsic_4x4)
            
            # 确保外参矩阵格式正确
            extrinsic_4x4 = extrinsic.astype(np.float32)
            extrinsics.append(extrinsic_4x4)
        
        intrinsics_stack = np.stack(intrinsics)
        extrinsics_stack = np.stack(extrinsics)
        
        # 处理关节状态
        joint_state = np.array(joint_positions, dtype=np.float32)
        
        # 默认的基座到世界坐标系变换
        if base_to_world_transform is None:
            base_to_world_transform = np.array([
                [0, -1, 0, 0],
                [1, 0, 0, -0.65],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        
        # 构建数据字典
        data = {
            "imgs": images_stack,
            "depths": depths_stack,
            "joint_state": joint_state[None],  # 添加batch维度
            "intrinsic": intrinsics_stack,
            "T_world2cam": extrinsics_stack,
            "T_base2world": base_to_world_transform,
            "text": instruction,
            "step_index": self.step_count,
        }
        
        # 应用数据变换
        if self.transforms is not None:
            for transform in self.transforms:
                if transform is None:
                    continue
                data = transform(data)
        
        # 添加batch维度
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v[None]
            else:
                data[k] = [v]
        
        return data
    
    def get_action(self, data: Dict[str, Any]) -> np.ndarray:
        """
        获取动作预测
        
        Args:
            data: 编码后的观测数据
            
        Returns:
            动作序列 (T, 14) - T为预测步数，14为关节数
        """
        with torch.no_grad():
            # 将数据移动到设备
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(self.device)
            
            # 模型推理
            outputs = self.model(data)
            actions = outputs[0]["pred_actions"][0]
            
            # 提取有效动作步数
            valid_action_step = 32
            actions = actions[:valid_action_step, :, 0].cpu().numpy()
            
            return actions
    
    def predict_action(self, 
                      images: List[np.ndarray], 
                      depths: List[np.ndarray],
                      joint_positions: List[float],
                      camera_intrinsics: List[np.ndarray],
                      camera_extrinsics: List[np.ndarray],
                      instruction: str,
                      base_to_world_transform: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测动作的便捷接口
        
        Args:
            images: RGB图像列表
            depths: 深度图像列表
            joint_positions: 关节位置列表
            camera_intrinsics: 相机内参矩阵列表
            camera_extrinsics: 相机外参矩阵列表
            instruction: 任务指令
            base_to_world_transform: 基座到世界坐标系变换
            
        Returns:
            动作序列
        """
        # 编码观测数据
        data = self.encode_real_robot_obs(
            images=images,
            depths=depths,
            joint_positions=joint_positions,
            camera_intrinsics=camera_intrinsics,
            camera_extrinsics=camera_extrinsics,
            instruction=instruction,
            base_to_world_transform=base_to_world_transform
        )
        
        # 获取动作
        actions = self.get_action(data)
        
        # 更新状态
        self.step_count += 1
        self.action_history.append(actions[0])  # 保存第一个动作
        
        return actions
    
    def reset(self):
        """重置策略状态"""
        self.action_history = []
        self.step_count = 0
        print("Policy state reset.")


def rotmat_to_euler_xyz(R):
    """将旋转矩阵转换为XYZ欧拉角"""
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.atan2(sy, cy)
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float32)

def euler_to_rotmat(rpy):
    """将XYZ欧拉角转换为旋转矩阵"""
    cr, sr = math.cos(rpy[0]), math.sin(rpy[0])
    cp, sp = math.cos(rpy[1]), math.sin(rpy[1])
    cy, sy = math.cos(rpy[2]), math.sin(rpy[2])
    
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], dtype=np.float32)
    return R

def get_color_image(cam):
    """获取彩色图像"""
    # 兼容常见命名：get_rgb_image / get_color_image / get_color / get_color_img
    for name in ["get_rgb_image", "get_color_image", "get_color", "get_color_img"]:
        if hasattr(cam, name):
            img = getattr(cam, name)()
            # 确保输出是 HxWx3，且为 float32 [0,1] 或 uint8 [0,255]
            if img is None:
                continue
            img = np.asarray(img)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            return img
    raise RuntimeError("RealSense 实例上未找到彩色图像方法（get_rgb_image/get_color_image/get_color/get_color_img）")

def get_depth_image(cam):
    """获取深度图像"""
    # 兼容常见命名：get_depth_image / get_depth / get_depth_img
    for name in ["get_depth_image", "get_depth", "get_depth_img"]:
        if hasattr(cam, name):
            depth = getattr(cam, name)()
            if depth is None:
                continue
            depth = np.asarray(depth)
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
            return depth
    raise RuntimeError("RealSense 实例上未找到深度图像方法")

def get_camera_intrinsics(cam):
    """获取相机内参"""
    if hasattr(cam, 'get_intrinsics'):
        return cam.get_intrinsics()
    elif hasattr(cam, 'intrinsics'):
        return cam.intrinsics
    else:
        # 默认内参
        return np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

def get_camera_extrinsics(cam):
    """获取相机外参"""
    if hasattr(cam, 'get_extrinsics'):
        return cam.get_extrinsics()
    elif hasattr(cam, 'extrinsics'):
        return cam.extrinsics
    else:
        # 默认外参（单位矩阵）
        return np.eye(4, dtype=np.float32)


class RealRobotInterface:
    """
    真实机器人接口实现
    
    基于PiperArmController和RealSense相机的真实机器人接口
    """
    
    def __init__(self, camera_config_path: Optional[str] = None):
        """
        初始化机器人接口
        
        Args:
            camera_config_path: 相机配置文件路径
        """
        self.is_connected = False
        self.arm = None
        self.cameras = {}
        self.camera_config = None
        
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
        
    def connect(self) -> bool:
        """
        连接到机器人
        
        Returns:
            连接是否成功
        """
        if not PIPER_AVAILABLE:
            print("Error: PiperArmController not available. Cannot connect to robot.")
            return False
        
        try:
            print("Connecting to robot...")
            # 初始化机械臂
            self.arm = PiperArmController(can_name="can0", has_gripper=True, auto_enable=True)
            
            # 初始化相机
            if realsense is None:
                print("Warning: RealSense drivers not available. Camera functionality will be limited.")
            else:
                self._init_cameras()
            
            self.is_connected = True
            print("Robot connected successfully.")
            return True
            
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
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
        print("Disconnecting from robot...")
        
        # 断开机械臂
        if self.arm is not None:
            try:
                self.arm.disable()
            except Exception as e:
                print(f"Error disabling arm: {e}")
        
        # 断开相机
        for pipeline in self.cameras.values():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"Error stopping camera: {e}")
        
        self.is_connected = False
        self.arm = None
        self.cameras = {}
        print("Robot disconnected.")
    
    def get_joint_positions(self) -> List[float]:
        """
        获取当前关节位置
        
        Returns:
            关节位置列表 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
        """
        if not self.is_connected or self.arm is None:
            print("Robot not connected")
            return [0.0] * 14
        
        try:
            # 获取机械臂关节位置
            joint_positions = self.arm.get_joint_positions()
            
            # 扩展为14维（左右臂各6个关节 + 左右夹爪各1个）
            if len(joint_positions) == 12:  # 只有12个关节
                # 添加夹爪位置（假设为0.0）
                joint_positions = list(joint_positions) + [0.0, 0.0]
            elif len(joint_positions) == 14:  # 已经有14个关节
                joint_positions = list(joint_positions)
            else:
                # 其他情况，补齐到14维
                joint_positions = list(joint_positions) + [0.0] * (14 - len(joint_positions))
            
            return joint_positions
            
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return [0.0] * 14
    
    def get_camera_data(self, camera_names: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        获取相机数据
        
        Args:
            camera_names: 相机名称列表
            
        Returns:
            相机数据字典，格式为：
            {
                'camera_name': {
                    'rgb': np.ndarray,  # RGB图像 (H, W, 3)
                    'depth': np.ndarray,  # 深度图像 (H, W)
                    'intrinsic': np.ndarray,  # 内参矩阵 (3, 3)
                    'extrinsic': np.ndarray,  # 外参矩阵 (4, 4)
                }
            }
        """
        camera_data = {}
        
        for cam_name in camera_names:
            try:
                if cam_name in self.cameras:
                    cam = self.cameras[cam_name]
                    
                    # 获取RGB图像
                    rgb_img = get_color_image(cam)
                    if rgb_img.dtype == np.float32:
                        rgb_img = (rgb_img * 255).astype(np.uint8)
                    
                    # 获取深度图像
                    depth_img = get_depth_image(cam)
                    
                    # 获取相机参数
                    intrinsic = get_camera_intrinsics(cam)
                    extrinsic = get_camera_extrinsics(cam)
                    
                    camera_data[cam_name] = {
                        'rgb': rgb_img,
                        'depth': depth_img,
                        'intrinsic': intrinsic,
                        'extrinsic': extrinsic
                    }
                else:
                    # 如果相机不可用，返回示例数据
                    print(f"Warning: Camera {cam_name} not available, using dummy data")
                    camera_data[cam_name] = {
                        'rgb': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                        'depth': np.random.rand(480, 640).astype(np.float32),
                        'intrinsic': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
                        'extrinsic': np.eye(4, dtype=np.float32)
                    }
                    
            except Exception as e:
                print(f"Error getting camera data for {cam_name}: {e}")
                # 返回示例数据
                camera_data[cam_name] = {
                    'rgb': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                    'depth': np.random.rand(480, 640).astype(np.float32),
                    'intrinsic': np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32),
                    'extrinsic': np.eye(4, dtype=np.float32)
                }
        
        return camera_data
    
    def execute_action(self, action: List[float]) -> bool:
        """
        执行动作
        
        Args:
            action: 动作列表 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            
        Returns:
            执行是否成功
        """
        if not self.is_connected or self.arm is None:
            print("Robot not connected")
            return False
        
        try:
            # 将14维动作转换为机械臂控制指令
            if len(action) >= 6:
                # 使用前6维作为末端位姿增量
                delta_pos = np.array(action[:3], dtype=np.float32)
                delta_rpy = np.array(action[3:6], dtype=np.float32)
                
                # 获取当前位姿
                current_pos, current_rotmat = self.arm.get_pose()
                
                # 计算目标位姿
                target_pos = current_pos + delta_pos
                current_rpy = rotmat_to_euler_xyz(current_rotmat)
                target_rpy = current_rpy + delta_rpy
                target_rotmat = euler_to_rotmat(target_rpy)
                
                # 执行运动
                self.arm.move_l(target_pos, target_rotmat, is_euler=False, speed=10, block=False)
                
                print(f"Executing action: pos_delta={delta_pos}, rpy_delta={delta_rpy}")
                return True
            else:
                print(f"Invalid action dimension: {len(action)}, expected at least 6")
                return False
                
        except Exception as e:
            print(f"Error executing action: {e}")
            return False
    
    def get_base_to_world_transform(self) -> np.ndarray:
        """
        获取基座到世界坐标系的变换矩阵
        
        Returns:
            变换矩阵 (4, 4)
        """
        # 默认变换矩阵（根据实际机器人配置调整）
        return np.array([
            [0, -1, 0, 0],
            [1, 0, 0, -0.65],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)


def main():
    """
    主函数 - 演示如何使用真实机器人SEM策略
    """
    # 配置参数
    ckpt_path = "path/to/your/model.safetensors"  # 替换为实际的模型路径
    config_file = "config_sem_robotwin.py"  # 替换为实际的配置文件路径
    instruction = "place the empty cup on the table"  # 任务指令
    camera_config_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml"  # 相机配置文件路径
    
    # 相机名称列表（使用实际的角色名称）
    camera_names = ["left_hand", "right_hand"]  # 根据实际相机配置修改
    
    # 控制参数
    step_hz = 5.0  # 控制频率
    dt = 1.0 / step_hz
    max_xyz_step = 0.01  # 每步最大位置增量 (m)
    max_rpy_step = np.deg2rad(2.0)  # 每步最大旋转增量 (rad)
    
    try:
        # 初始化机器人接口
        robot_interface = RealRobotInterface(camera_config_path=camera_config_path)
        
        # 连接机器人
        if not robot_interface.connect():
            print("Failed to connect to robot.")
            return
        
        # 初始化SEM策略
        print("Initializing SEM policy...")
        policy = RealRobotSEMPolicy(ckpt_path, config_file)
        
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
                base_to_world_transform = robot_interface.get_base_to_world_transform()
                
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
                    base_to_world_transform=base_to_world_transform
                )
                inference_time = time.time() - start_time
                
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Predicted actions shape: {actions.shape}")
                
                # 执行第一个动作
                if len(actions) > 0:
                    first_action = actions[0].tolist()
                    
                    # 限幅处理（避免大步跳变）
                    if len(first_action) >= 6:
                        # 位置限幅
                        first_action[:3] = np.clip(first_action[:3], -max_xyz_step, max_xyz_step)
                        # 旋转限幅
                        first_action[3:6] = np.clip(first_action[3:6], -max_rpy_step, max_rpy_step)
                    
                    success = robot_interface.execute_action(first_action)
                    
                    if not success:
                        print("Failed to execute action.")
                        break
                    
                    print(f"Action executed: {first_action[:6]}")
                
                # 控制频率
                time.sleep(dt)
                step_count += 1
                
                # 检查任务完成条件（需要根据具体任务实现）
                # if task_completed:
                #     print("Task completed!")
                #     break
                
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
