#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三相机双机械臂数据构建器
用于构建符合MultiArmManipulationInput格式的输入数据
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional
from dataclasses import dataclass

from wrs.robot_con.piper.MultiRobotInterface import MultiRobotInterface
from wrs.robot_con.piper.MultiRobotInterface import *


@dataclass
class CameraData:
    """相机数据结构"""
    # RGB图像: [H, W, 3] - uint8类型，范围0-255
    image: np.ndarray  # shape: [H, W, 3], dtype: uint8
    
    # 深度图像: [H, W] - float32类型，单位为米
    depth: np.ndarray  # shape: [H, W], dtype: float32
    
    # 相机内参矩阵: [4, 4] - float64类型
    intrinsic: np.ndarray  # shape: [4, 4], dtype: float64
    
    # 世界坐标系到相机坐标系的变换矩阵: [4, 4] - float64类型
    t_world2cam: np.ndarray  # shape: [4, 4], dtype: float64


@dataclass
class RobotArmData:
    """机械臂数据结构"""
    # 关节状态历史: [hist_steps, num_joints] - float64类型
    # hist_steps: 历史步数，通常为1
    # num_joints: 关节数量，双机械臂通常为12-14个关节
    history_joint_state: List[np.ndarray]  # List[shape: [num_joints], dtype: float64]


class MultiArmDataBuilder:
    """三相机双机械臂数据构建器"""
    
    def __init__(
        self,
        interface: Optional[MultiRobotInterface] = None,
        image_height: int = 480,
        image_width: int = 640,
        num_joints_per_arm: int = 7,
        hist_steps: int = 1
    ):
        """
        初始化数据构建器
        
        Args:
            image_height: 图像高度，默认480
            image_width: 图像宽度，默认640
            num_joints_per_arm: 每只机械臂的关节数，默认7
            hist_steps: 历史步数，默认1
        """
        self.image_height = image_height
        self.image_width = image_width
        self.num_joints_per_arm = num_joints_per_arm
        self.hist_steps = hist_steps
        self.num_joints_total = num_joints_per_arm * 2  # 双机械臂总关节数
        self.interface= interface
        
        # 相机名称列表
        self.camera_names = ["middle", "left", "right"]
        
    def create_camera_data(
        self,
        camera_name: str,
        image: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        intrinsic: Optional[np.ndarray] = None,
        t_world2cam: Optional[np.ndarray] = None
    ) -> CameraData:
        """
        创建单个相机的数据
        
        Args:
            camera_name: 相机名称
            image: RGB图像，如果为None则创建随机图像
            depth: 深度图像，如果为None则创建随机深度
            intrinsic: 内参矩阵，如果为None则创建默认内参
            t_world2cam: 外参矩阵，如果为None则创建默认外参
            
        Returns:
            CameraData: 相机数据对象
        """
        # 创建默认RGB图像: [H, W, 3] - uint8类型
        if image is None:
            try:
                image = self.interface.get_color_image(camera_name)
            except Exception as e:
                print(e)
                image = np.random.randint(0, 256, (self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # 创建默认深度图像: [H, W] - float32类型，范围0.1-5.0米
        if depth is None:
            try:
                depth = self.interface.get_depth_image(camera_name)
            except Exception as e:
                print(e)
                depth = np.random.uniform(0.1, 5.0, (self.image_height, self.image_width)).astype(np.float32)
        
        # 创建默认内参矩阵: [4, 4] - float64类型
        if intrinsic is None:
            try:
                intrinsic = self.interface.get_camera_intrinsics(camera_name)
            except Exception as e:
                print(e)
                intrinsic = self._create_default_intrinsic()
        
        # 创建默认外参矩阵: [4, 4] - float64类型
        if t_world2cam is None:
            if camera_name in ["left", "right"]:
                # 获取相机到末端的变换 (ee2cam)
                ee2cam = self.interface.get_camera_extrinsics(camera_name)
                
                if camera_name == "left":
                    # 左臂（基座与世界坐标系重合）
                    left_ee_pos = self.interface.arms['left_arm'].get_pose()
                    
                    T_base_ee = np.eye(4)
                    T_base_ee[:3, :3] = left_ee_pos[1]  # 旋转部分
                    T_base_ee[:3, 3] = left_ee_pos[0]   # 平移部分
                    
                    # 计算 world2cam = ee2cam @ inv(ee2base)
                    t_world2cam = T_base_ee @ ee2cam
                    
                elif camera_name == "right":
                    # 右臂（基座在世界坐标系下Y轴-0.6m处）
                    right_ee_pos = self.interface.arms['right_arm'].get_pose()
                    
                    # 构建 ee2base_right 矩阵（末端到右臂基座）
                    T_base_ee = np.eye(4)
                    T_base_ee[:3, :3] = right_ee_pos[1]  # 旋转部分
                    T_base_ee[:3, 3] = right_ee_pos[0]   # 平移部分
                    
                    t_base2cam = T_base_ee @ ee2cam

                    # 构造 T_world_base（世界坐标系到机器人基坐标系的变换）
                    T_world_base = np.eye(4)
                    T_world_base[1, 3] = -0.6  # Y 轴负方向 0.6 米

                    # 计算 T_world_cam
                    t_world2cam = T_world_base @ t_base2cam
            
            elif camera_name == "middle":
                cam2base = self.interface.get_camera_extrinsics(camera_name)
                t_world2cam = cam2base
        

        return CameraData(
            image=image,
            depth=depth,
            intrinsic=intrinsic,
            t_world2cam=t_world2cam
        )

    def create_robot_arm_data(
        self,
        left_arm_joints: Optional[List[np.ndarray]] = None,
        right_arm_joints: Optional[List[np.ndarray]] = None
    ) -> RobotArmData:
        """
        创建双机械臂数据
        
        Args:
            left_arm_joints: 左臂关节状态历史，如果为None则创建随机数据
            right_arm_joints: 右臂关节状态历史，如果为None则创建随机数据
            
        Returns:
            RobotArmData: 机械臂数据对象
        """
        # 创建默认左臂关节状态: [hist_steps, num_joints_per_arm] - float64类型
        if left_arm_joints is None:
            try:
                left_arm_joints = self.interface.arms['left_arm'].get_joint_values()
               
            except Exception as e:
                print(e)
                left_arm_joints = [
                    np.random.uniform(-np.pi, np.pi, self.num_joints_per_arm).astype(np.float64)
                    for _ in range(self.hist_steps)
                ]

            left_gripper_status = self.interface.arms['left_arm'].get_gripper_status()
            left_gripper_opening = left_gripper_status[0]

            left_arm_joints = np.append(left_arm_joints, left_gripper_opening)

            print(f'left_arm_joints:{left_arm_joints}')
        
        # 创建默认右臂关节状态: [hist_steps, num_joints_per_arm] - float64类型
        if right_arm_joints is None:
            try:
                right_arm_joints = self.interface.arms['right_arm'].get_joint_values()
                
            except Exception as e:
                print(e)
                right_arm_joints = [
                    np.random.uniform(-np.pi, np.pi, self.num_joints_per_arm).astype(np.float64)
                    for _ in range(self.hist_steps)
                ]

            right_gripper_status = self.interface.arms['right_arm'].get_gripper_status()
            right_gripper_opening = right_gripper_status[0]

            right_arm_joints = np.append(right_arm_joints, right_gripper_opening)

            print(f'right_arm_joints:{right_arm_joints}')

        # 合并双机械臂关节状态: [hist_steps, num_joints_total] - float64类型
        history_joint_state = []
        combined_joints = np.concatenate([left_arm_joints, right_arm_joints])
        history_joint_state.append(combined_joints)
        
        return RobotArmData(history_joint_state=history_joint_state)
    
    def build_multi_arm_data(
        self,
        instruction: str = "Pick up the red object with both arms",
        camera_data_dict: Optional[Dict[str, CameraData]] = None,
        robot_arm_data: Optional[RobotArmData] = None,
        t_robot2world: Optional[np.ndarray] = None,
        step_index: int = 0 
    ) -> Dict:
        """
        构建完整的多机械臂数据
        
        Args:
            instruction: 任务指令文本
            camera_data_dict: 相机数据字典，如果为None则创建默认数据
            robot_arm_data: 机械臂数据，如果为None则创建默认数据
            t_robot2world: 机器人基座到世界坐标系的变换矩阵，如果为None则创建单位矩阵
            step_index: 当前时间步索引，默认为0
            
        Returns:
            Dict: 符合MultiArmManipulationInput格式的数据字典
        """
        # 创建默认相机数据
        if camera_data_dict is None:
            camera_data_dict = {}
            for cam_name in self.camera_names:
                camera_data_dict[cam_name] = self.create_camera_data(cam_name)
        
        # 创建默认机械臂数据
        if robot_arm_data is None:
            robot_arm_data = self.create_robot_arm_data()
        
        # 创建默认机器人基座变换矩阵: [4, 4] - float64类型
        if t_robot2world is None:
            t_robot2world = np.eye(4, dtype=np.float64)
        
        # 构建数据字典
        data = {
            # 图像数据: from camera_data_dict
            "imgs": np.stack([camera_data.image for camera_data in camera_data_dict.values()]),  # shape: [num_cameras, H, W, 3], dtype: uint8
            
            # 深度数据: from camera_data_dict
            "depths": np.stack([camera_data.depth for camera_data in camera_data_dict.values()]),  # shape: [num_cameras, H, W], dtype: float32
            
            # 内参矩阵: from camera_data_dict
            "intrinsic": np.stack([camera_data.intrinsic.astype(np.float64) for camera_data in camera_data_dict.values()]),  # shape: [num_cameras, 4, 4], dtype: float64
            
            # 外参矩阵: from camera_data_dict
            "T_world2cam": np.stack([camera_data.t_world2cam.astype(np.float64) for camera_data in camera_data_dict.values()]),  # shape: [num_cameras, 4, 4], dtype: float64
            
            # 机器人基座变换矩阵: from t_robot2world
            "T_base2world": t_robot2world.astype(np.float64),  # shape: [4, 4], dtype: float64
            
            # 关节状态: from robot_arm_data
            "joint_state": np.array(robot_arm_data.history_joint_state),  # shape: [hist_steps, num_joints_total], dtype: float64
            
            # 任务指令: from instruction
            "text": instruction,
            
            # 当前时间步索引 from step_index
            "step_index": step_index,
            
            # 添加任务名称字段 (可选)
            "task_name": "your_task_name_here",
            
            # 添加帧索引字段 (可选)
            "frame_index": 0,
            
            # 添加情景索引字段 (可选)
            "episode_index": 0
        }
        
        return data
    
    def _create_default_intrinsic(self) -> np.ndarray:
        """创建默认内参矩阵: [4, 4] - float64类型"""
        intrinsic = np.eye(4, dtype=np.float64)
        
        # 设置焦距和主点（假设相机参数）
        fx = fy = 500.0  # 焦距
        cx = self.image_width / 2.0  # 主点x坐标
        cy = self.image_height / 2.0  # 主点y坐标
        
        intrinsic[0, 0] = fx  # fx
        intrinsic[1, 1] = fy  # fy
        intrinsic[0, 2] = cx  # cx
        intrinsic[1, 2] = cy  # cy
        
        return intrinsic
    
    def _create_default_extrinsic(self, camera_name: str) -> np.ndarray:
        """创建默认外参矩阵: [4, 4] - float64类型"""
        extrinsic = np.eye(4, dtype=np.float64)
        
        # 根据相机名称设置不同的位置
        if camera_name == "camera_left":
            extrinsic[0, 3] = -0.5  # 左相机位置
            extrinsic[1, 3] = 0.0
            extrinsic[2, 3] = 1.0
        elif camera_name == "camera_center":
            extrinsic[0, 3] = 0.0  # 中相机位置
            extrinsic[1, 3] = 0.0
            extrinsic[2, 3] = 1.0
        elif camera_name == "camera_right":
            extrinsic[0, 3] = 0.5  # 右相机位置
            extrinsic[1, 3] = 0.0
            extrinsic[2, 3] = 1.0
        
        return extrinsic


def create_example_data() -> Dict:
    """
    创建示例数据
    
    Returns:
        Dict: 符合MultiArmManipulationInput格式的示例数据
    """
    # 创建数据构建器
    builder = MultiArmDataBuilder(
        image_height=480,
        image_width=640,
        num_joints_per_arm=7,
        hist_steps=1
    )
    
    # 构建数据
    data = builder.build_multi_arm_data(
        instruction="Pick up the red object with both arms and place it in the container"
    )
    
    return data


def print_data_info(data: Dict):
    """打印数据信息"""
    print("=== 三相机双机械臂数据信息 ===")
    print(f"任务指令: {data['instruction']}")
    print(f"相机数量: {len(data['image'])}")
    print(f"相机名称: {list(data['image'].keys())}")
    
    # 打印图像信息
    for cam_name, images in data['image'].items():
        img_shape = images[0].shape
        print(f"{cam_name} 图像形状: {img_shape}, 数据类型: {images[0].dtype}")
    
    # 打印深度信息
    for cam_name, depths in data['depth'].items():
        depth_shape = depths[0].shape
        print(f"{cam_name} 深度形状: {depth_shape}, 数据类型: {depths[0].dtype}")
    
    # 打印内参信息
    for cam_name, intrinsic in data['intrinsic'].items():
        print(f"{cam_name} 内参形状: {intrinsic.shape}, 数据类型: {intrinsic.dtype}")
    
    # 打印外参信息
    for cam_name, extrinsic in data['t_world2cam'].items():
        print(f"{cam_name} 外参形状: {extrinsic.shape}, 数据类型: {extrinsic.dtype}")
    
    # 打印机械臂信息
    joint_state_shape = data['history_joint_state'][0].shape
    print(f"关节状态形状: {joint_state_shape}, 数据类型: {data['history_joint_state'][0].dtype}")
    print(f"历史步数: {len(data['history_joint_state'])}")
    
    # 打印机器人基座变换信息
    print(f"机器人基座变换形状: {data['t_robot2world'].shape}, 数据类型: {data['t_robot2world'].dtype}")


if __name__ == "__main__":
    # 创建示例数据
    example_data = create_example_data()
    
    # 打印数据信息
    print_data_info(example_data)
    
    print("\n=== 数据构建完成 ===")
    print("数据已准备好用于processor.pre_process()处理")
