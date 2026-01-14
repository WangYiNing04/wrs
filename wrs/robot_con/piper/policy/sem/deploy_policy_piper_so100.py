'''
该文件用于部署Piper的SEM策略
Obs:
- 包含彩色图像、深度图像、相机内参、相机外参、关节位置

Action:
- 包含关节位置、夹爪位置

Piper API:
- get_color_image(cam) -> np.ndarray       获得真实世界彩色图像
- get_depth_image(cam) -> np.ndarray       获得真实世界深度图像
- get_camera_intrinsics(cam) -> np.ndarray       获得相机内参
- get_camera_extrinsics(cam) -> np.ndarray       获得相机内参
- get_joint_positions(cam) -> np.ndarray       获得关节位置
- get_gripper_position(cam) -> np.ndarray       获得夹爪位置
- execute_action(action) -> bool       执行动作

PYTHONPATH:
/home/wyn/RoboOrchardLab-master/projects/sem/robotwin/onnx_scripts:
:/home/wyn/wrs-main:/home/wyn/RoboOrchardLab-master/projects/sem/robotwin:
/~/PycharmProjects/wrs_tiaozhanbei:/home/wyn/PycharmProjects/wrs_tiaozhanbei


使用前测试，确保相机和机械臂能正常工作
'''

import copy
import importlib
import os
import time
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any


import copy
import importlib
import os
import time
import math
import yaml
from typing import Dict, List, Optional, Tuple, Any
import onnxruntime as ort
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
#import cv2



from wrs.robot_con.piper.MultiRobotInterface import *



def pose_to_homogeneous(xyz, quat):
    """
    将 (x, y, z, qx, qy, qz, qw) 转换为 4x4 齐次变换矩阵

    参数:
        xyz: [x, y, z]
        quat: [qx, qy, qz, qw]  # 注意顺序，w 在最后

    返回:
        4x4 numpy.ndarray
    """
    # 1. 计算旋转矩阵
    rotation = R.from_quat(quat).as_matrix()  # 3x3

    # 2. 构造 4x4 齐次矩阵
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = xyz
    return T

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

# 获取彩色图像
def get_color_image(cam):
    """获取彩色图像"""
    try:
        return cam.get_color_image()
    except Exception as e:
        print(f"Error getting color image: {e}")
        return None

# 获取深度图像
def get_depth_image(cam):
    """获取深度图像"""
    try:
        return cam.get_depth_image()
    except Exception as e:
        print(f"Error getting depth image: {e}")
        return None

# 获取相机内参
def get_camera_intrinsics(cam):
    """获取相机内参"""
    try:
        return cam.get_intrinsics()
    except Exception as e:
        print(f"Error getting camera intrinsics: {e}")
        return np.eye(3, dtype=np.float32)

# 获取相机外参
def get_camera_extrinsics(cam):
    """获取相机外参"""
    try:
        return cam.get_extrinsics()
    except Exception as e:
        print(f"Error getting camera extrinsics: {e}")
        return np.eye(4, dtype=np.float32)



import sys

# 添加onnx_scripts到Python路径 - 必须在导入其他模块之前
onnx_scripts_path = os.path.join(os.path.dirname(__file__), "..", "onnx_scripts")
sys.path.insert(0, os.path.abspath(onnx_scripts_path))

import sys
sys.path.append('/home/wyn/RoboOrchardLab-master/projects/sem/robotwin/onnx_scripts')


# 内联函数定义，替代外部依赖
import json
from contextlib import contextmanager
from typing import Any, Dict, Type, Union, Literal
from accelerate import Accelerator
from safetensors.torch import load_model as safetensors_load_model, save_model as safetensors_save_model

# 内联 in_cwd 函数
@contextmanager
def in_cwd(destination: str):
    """Context manager to temporarily change the current working directory."""
    try:
        original_path = os.getcwd()
        os.chdir(destination)
        yield destination
    finally:
        os.chdir(original_path)

# 内联 load_config_class 函数
def load_config_class(config_str: str) -> Type:
    """Load a class from configuration string."""
    try:
        config_dict = json.loads(config_str)
        if 'class_name' in config_dict:
            # 动态导入类
            module_path = config_dict['class_name']
            module_name, class_name = module_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls
        else:
            # 如果没有class_name，返回一个简单的配置类
            class SimpleConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            return SimpleConfig
    except Exception as e:
        print(f"Error loading config class: {e}")
        # 返回一个默认的配置类
        class DefaultConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return DefaultConfig

# 内联 ModelMixin 类
class ModelMixin:
    """A simplified ModelMixin for loading models."""
    
    @staticmethod
    def load_model(
        directory: str,
        load_weight: bool = True,
        strict: bool = True,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads a model from a local directory."""
        
        directory = os.path.abspath(directory)
        
        if not os.path.exists(directory):
            raise FileNotFoundError(f"checkpoint {directory} does not exists!")
        
        config_file = os.path.join(directory, f"{model_prefix}.config.json")
        
        with open(config_file, "r") as f:
            cfg = load_config_class(f.read())
        
        with in_cwd(directory):
            model = cfg()
        
        if load_weight:
            model.load_weights(
                directory=directory,
                strict=strict,
                device=device,
                device_map=device_map,
                model_prefix=model_prefix,
                load_impl=load_impl,
            )
        return model
    
    def load_weights(
        self,
        directory: str,
        device: str | None = "cpu",
        device_map: str | dict[str, int | str | torch.device] | None = None,
        strict: bool = True,
        model_prefix: str = "model",
        load_impl: Literal["native", "accelerate"] = "accelerate",
    ):
        """Loads pretrained weights from a directory."""
        
        if load_impl == "native":
            if device_map is not None:
                raise ValueError(
                    "device_map is not supported when load_impl is 'native'."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"{ckpt_path} does not exists!")
            missing, unexpected = safetensors_load_model(
                self, filename=ckpt_path, strict=strict
            )
            if len(missing) > 0:
                print(f"Warning: Some weights are missing: {missing}")
            if len(unexpected) > 0:
                print(f"Warning: Some unexpected weights found: {unexpected}")
        elif load_impl == "accelerate":
            from accelerate import load_checkpoint_and_dispatch
            
            if device is not None and device_map is not None:
                raise ValueError(
                    "Only one of device or device_map can be specified."
                )
            ckpt_path = os.path.join(directory, f"{model_prefix}.safetensors")
            if os.path.exists(ckpt_path):
                load_checkpoint_and_dispatch(
                    self, ckpt_path, strict=strict, device_map=device_map
                )
            else:
                load_checkpoint_and_dispatch(
                    self, directory, strict=strict, device_map=device_map
                )
        else:
            raise ValueError(
                f"Invalid load_impl: {load_impl}, "
                f"expected 'native' or 'accelerate'."
            )
        
        if device is not None:
            self.to(device=device)

from data_builder import MultiArmDataBuilder

def main():
    """
    主函数 - 演示如何使用多机械臂SEM策略

    开始前通过插拔can口确认左臂为can0,右臂为can1
    
    """

     # 检查GPU可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # 配置参数
    ckpt_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/policy/sem/model/stack_blocks_three/model.safetensors"  # 替换为实际的模型路径
    config_file = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/policy/sem/config_sem_robotwin.py"  # SEM配置文件路径
    instruction = "Place red block, green block, and blue block in the center, then arrange green block on red block and blue block on green block"  # 任务指令
    camera_config_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml"  # 相机配置文件路径
    home_state_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/home_state1.yaml"  # 零位状态配置文件路径
    
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
    camera_names = ["left", "middle", "right"]
    
    # 控制参数
    step_hz = 20.0  # 控制频率
    dt = 1.0 / step_hz
    max_joint_step = 0.1  # 最大关节位置步长
    
    
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

        print("Initializing data builder...")
        # 初始化数据构建器
        builder = MultiArmDataBuilder(
            interface=robot_interface,
            image_height=320,
            image_width=256,
            num_joints_per_arm=7,
            hist_steps=1
        )

        data = builder.build_multi_arm_data(
            instruction="Your task instruction",
            camera_data_dict=None,
            robot_arm_data=None,
            t_robot2world=None,
            step_index=None
        )
        
        # 初始化多机械臂SEM策略
        print("Initializing Multi-Robot SEM policy...")
        
        
        # 主循环
        print("Starting inference loop...")
        print(f"Control frequency: {step_hz} Hz")
        print(f"Max joint step: {max_joint_step} radians")
        print("Press Ctrl+C to stop...")
        
    
    
        # 定义模型路径 - 指向workspace目录
        output_path = os.path.join(os.path.dirname(__file__), "workspace")

        

        print("Loading ONNX model...")
        import onnxruntime as ort
        print(f"ONNX Runtime available devices: {ort.get_device()}")
        
        # 加载ONNX模型配置
        model_config_path = os.path.join(output_path, "model.config.json")
        with open(model_config_path, 'r') as f:
            model_config = json.load(f)
        
        # 查找ONNX模型文件 - 支持encoder和decoder两个模型
        onnx_files = [f for f in os.listdir(output_path) if f.endswith('.onnx')]
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model found in {output_path}")
        
        # 查找encoder和decoder模型文件
        encoder_model_path = None
        decoder_model_path = None
        
        for onnx_file in onnx_files:
            if 'encoder' in onnx_file.lower():
                encoder_model_path = os.path.join(output_path, onnx_file)
            elif 'decoder' in onnx_file.lower():
                decoder_model_path = os.path.join(output_path, onnx_file)
        
        # 如果没有找到明确的encoder/decoder文件，使用第一个文件作为默认模型
        if not encoder_model_path and not decoder_model_path:
            print("Warning: No encoder/decoder specific ONNX files found, using single model approach")
            single_model_path = os.path.join(output_path, onnx_files[0])
            encoder_model_path = single_model_path
            decoder_model_path = single_model_path
        elif not encoder_model_path:
            print("Warning: No encoder ONNX file found, using decoder model for both")
            encoder_model_path = decoder_model_path
        elif not decoder_model_path:
            print("Warning: No decoder ONNX file found, using encoder model for both")
            decoder_model_path = encoder_model_path
        
        print(f"Loading Encoder ONNX model from: {encoder_model_path}")
        print(f"Loading Decoder ONNX model from: {decoder_model_path}")
        
        # 配置ONNX Runtime providers
        providers = []
        if device.type == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # 创建ONNX推理会话
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # 根据模型配置设置推理步数
        num_inference_timesteps = model_config.get('num_inference_timesteps', 10)
        print(f"Using {num_inference_timesteps} inference timesteps")
        
        # 创建encoder和decoder会话
        encoder_session = ort.InferenceSession(
            encoder_model_path, 
            providers=providers,
            sess_options=session_options
        )
        
        decoder_session = ort.InferenceSession(
            decoder_model_path, 
            providers=providers,
            sess_options=session_options
        )
        
        # 获取encoder输入输出信息
        encoder_input_names = [input.name for input in encoder_session.get_inputs()]
        encoder_output_names = [output.name for output in encoder_session.get_outputs()]
        
        # 获取decoder输入输出信息
        decoder_input_names = [input.name for input in decoder_session.get_inputs()]
        decoder_output_names = [output.name for output in decoder_session.get_outputs()]
        
        print(f"Encoder Model input names: {encoder_input_names}")
        print(f"Encoder Model output names: {encoder_output_names}")
        print(f"Decoder Model input names: {decoder_input_names}")
        print(f"Decoder Model output names: {decoder_output_names}")
        print(f"ONNX Model providers: {encoder_session.get_providers()}")
        
        # 加载数据预处理器配置
        data_preprocessor_config = model_config.get('data_preprocessor', {})
        if data_preprocessor_config:
            print(f"Data preprocessor config loaded: {data_preprocessor_config.get('type', 'Unknown')}")
        
        # 加载ONNX模型类（用于后处理）
        try:
            from onnx_model import OnnxSEMModel
            onnx_model_class = OnnxSEMModel(model_config)
            print("ONNX model class loaded successfully")
        except ImportError as e:
            print(f"Warning: Could not import OnnxSEMModel: {e}")
            onnx_model_class = None
        
        # 定义ONNX推理辅助函数
        def run_onnx_inference(data_dict, input_names, output_names, session):
            """执行ONNX推理的辅助函数"""
            # 准备输入数据
            onnx_inputs = {}
            missing_inputs = []
            
            for name in input_names:
                if name in data_dict:
                    if isinstance(data_dict[name], torch.Tensor):
                        onnx_inputs[name] = data_dict[name].cpu().numpy()
                    else:
                        onnx_inputs[name] = data_dict[name]
                else:
                    missing_inputs.append(name)
            
            if missing_inputs:
                print(f"Warning: Missing inputs: {missing_inputs}")
                print(f"Available data keys: {list(data_dict.keys())}")
                return None
            
            try:
                # 执行推理
                onnx_outputs = session.run(output_names, onnx_inputs)
                
                # 转换输出格式
                model_outs = {}
                for i, output_name in enumerate(output_names):
                    if i < len(onnx_outputs):
                        model_outs[output_name] = torch.from_numpy(onnx_outputs[i]).to(device)
                
                return model_outs
            except Exception as e:
                print(f"ONNX inference error: {e}")
                print(f"Input shapes: {[(name, inp.shape if hasattr(inp, 'shape') else type(inp)) for name, inp in onnx_inputs.items()]}")
                return None
        
        # 定义双模型推理函数
        def run_dual_onnx_inference(data_dict, encoder_session, decoder_session, 
                                   encoder_input_names, encoder_output_names,
                                   decoder_input_names, decoder_output_names):
            """执行encoder-decoder双模型推理"""
            # 第一步：执行encoder推理
            encoder_start = time.time()
            encoder_outs = run_onnx_inference(data_dict, encoder_input_names, encoder_output_names, encoder_session)
            encoder_time = time.time() - encoder_start
            inference_stats['encoder_inference_time'] += encoder_time
            
            if encoder_outs is None:
                print("Encoder inference failed")
                return None
            
            # 第二步：准备decoder输入（将encoder输出作为decoder输入）
            decoder_data = data_dict.copy()
            decoder_data.update(encoder_outs)
            
            # 第三步：执行decoder推理
            decoder_start = time.time()
            decoder_outs = run_onnx_inference(decoder_data, decoder_input_names, decoder_output_names, decoder_session)
            decoder_time = time.time() - decoder_start
            inference_stats['decoder_inference_time'] += decoder_time
            
            if decoder_outs is None:
                print("Decoder inference failed")
                return None
            
            # 合并encoder和decoder输出
            final_outs = encoder_outs.copy()
            final_outs.update(decoder_outs)
            
            return final_outs
        
        # 推理统计
        inference_stats = {
            'total_inferences': 0,
            'successful_inferences': 0,
            'failed_inferences': 0,
            'total_inference_time': 0.0,
            'encoder_inference_time': 0.0,
            'decoder_inference_time': 0.0
        } 

        #model = torch.compile(model, mode="reduce-overhead", fullgraph=True)



        # 加载处理器（只执行一次）
        processor_cfg = load_config_class(
            open(f"{output_path}/processor.json").read()
        )
        with in_cwd(output_path):
            processor = processor_cfg()

       
    
        # 预热ONNX模型（第一次推理通常较慢）
        print("Warming up ONNX models...")
        dummy_data = builder.build_multi_arm_data(
            instruction="dummy",
            camera_data_dict=None,
            robot_arm_data=None,
            t_robot2world=None,
            step_index=0
        )
        dummy_data = processor.pre_process(dummy_data)
        
        # 检查是否使用双模型推理
        use_dual_model = (encoder_model_path != decoder_model_path)
        
        if use_dual_model:
            print("Using dual model (encoder + decoder) inference")
            warmup_result = run_dual_onnx_inference(
                dummy_data, encoder_session, decoder_session,
                encoder_input_names, encoder_output_names,
                decoder_input_names, decoder_output_names
            )
        else:
            print("Using single model inference")
            warmup_result = run_onnx_inference(dummy_data, encoder_input_names, encoder_output_names, encoder_session)
        
        if warmup_result is not None:
            print("ONNX model(s) warmup completed.")
        else:
            print("ONNX model(s) warmup failed.")
            print("Available keys in dummy_data:", list(dummy_data.keys()))
            if use_dual_model:
                print("Required encoder input names:", encoder_input_names)
                print("Required decoder input names:", decoder_input_names)
            else:
                print("Required input names:", encoder_input_names)

        step_count = 0
        max_steps = 1000  # 最大执行步数
        
        while step_count < max_steps:
            try:
                print(f"\nStep {step_count + 1}/{max_steps}")
                time.sleep(3) #保证相机稳定
                data_start = time.time()
                data = builder.build_multi_arm_data(
                    instruction="Your task instruction",
                    camera_data_dict=None,
                    robot_arm_data=None,
                    t_robot2world=None,
                    step_index=step_count
                )
                data_build_time = time.time() - data_start

                #输出data
                #print(data)

                # 预测动作
                start_time = time.time()
                



                # init data dict with imgs, depths, text, intrinsic, joint_state

                preprocess_start = time.time()
                data = processor.pre_process(data)
                preprocess_time = time.time() - preprocess_start

                
                # ONNX推理
                inference_start = time.time()
                
                # 使用双模型或单模型推理
                inference_stats['total_inferences'] += 1
                if use_dual_model:
                    model_outs = run_dual_onnx_inference(
                        data, encoder_session, decoder_session,
                        encoder_input_names, encoder_output_names,
                        decoder_input_names, decoder_output_names
                    )
                else:
                    model_outs = run_onnx_inference(data, encoder_input_names, encoder_output_names, encoder_session)
                
                if model_outs is None:
                    inference_stats['failed_inferences'] += 1
                    print("ONNX inference failed, skipping this step")
                    continue
                
                inference_stats['successful_inferences'] += 1
                inference_time = time.time() - inference_start
                inference_stats['total_inference_time'] += inference_time

                postprocess_start = time.time()
    
                # 使用ONNX模型类进行后处理（如果可用）
                if onnx_model_class is not None:
                    try:
                        # 使用ONNX模型类进行后处理
                        actions = onnx_model_class.post_process(
                            batch=1, 
                            model_outputs=model_outs
                        ).action
                    except Exception as e:
                        print(f"ONNX model post-processing failed: {e}")
                        # 回退到原始processor后处理
                        actions = processor.post_process(batch=1, model_outputs=model_outs).action
                else:
                    # 使用原始processor进行后处理
                    actions = processor.post_process(batch=1, model_outputs=model_outs).action

                postprocess_time = time.time() - postprocess_start
                #print(actions)
                # 定期清理缓存
             
                inference_time = time.time() - start_time
                
                # 打印时间统计
                print(f"Data build: {data_build_time:.4f}s | "
                      f"Preprocess: {preprocess_time:.4f}s | "
                      f"Inference: {inference_time:.4f}s | "
                      f"Postprocess: {postprocess_time:.4f}s")
                
                print(f"Inference time: {inference_time:.3f}s")
                print(f"Predicted actions shape: {actions.shape}")
                
                # 每100步打印推理统计
                if step_count % 100 == 0 and step_count > 0:
                    success_rate = inference_stats['successful_inferences'] / inference_stats['total_inferences'] * 100
                    avg_inference_time = inference_stats['total_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
                    
                    if use_dual_model:
                        avg_encoder_time = inference_stats['encoder_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
                        avg_decoder_time = inference_stats['decoder_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
                        print(f"ONNX Dual Model Stats - Total: {inference_stats['total_inferences']}, "
                              f"Success: {inference_stats['successful_inferences']}, "
                              f"Failed: {inference_stats['failed_inferences']}, "
                              f"Success Rate: {success_rate:.1f}%, "
                              f"Avg Total Time: {avg_inference_time:.4f}s, "
                              f"Avg Encoder: {avg_encoder_time:.4f}s, "
                              f"Avg Decoder: {avg_decoder_time:.4f}s")
                    else:
                        print(f"ONNX Single Model Stats - Total: {inference_stats['total_inferences']}, "
                              f"Success: {inference_stats['successful_inferences']}, "
                              f"Failed: {inference_stats['failed_inferences']}, "
                              f"Success Rate: {success_rate:.1f}%, "
                              f"Avg Time: {avg_inference_time:.4f}s")
                
                print("First action command:")
                #print(actions[0])

                # 执行完整的64步动作序列

                #continue
                #actions.shape[0]
                for i in range(64):
                    if step_count >= max_steps:
                        break
                        
                    action = actions[i].tolist()
                    
                    # 检查动作长度
                    if len(action) != 14:
                        print(f"Warning: Expected 14 action elements, got {len(action)}")
                        # 如果动作长度不正确，使用零动作
                        action = [0] * 14
                    
                    # # 限幅处理（避免大步跳变）
                    # # 前6个是左臂关节位置
                    # action[:6] = np.clip(action[:6], -max_joint_step, max_joint_step)
                    # # 第7个是左臂夹爪开合度
                    # action[6] = np.clip(action[6], 0, 1)  # 假设夹爪开合度在0-1之间
                    
                    # # 后6个是右臂关节位置
                    # action[7:13] = np.clip(action[7:13], -max_joint_step, max_joint_step)
                    # # 第14个是右臂夹爪开合度
                    # action[13] = np.clip(action[13], 0, 1)  # 假设夹爪开合度在0-1之间

                    # 获取当前关节位置
                    current_joints_left = robot_interface.arms['left_arm'].get_joint_values()
                    current_joints_right = robot_interface.arms['right_arm'].get_joint_values()

                    # 计算目标位置与当前位置的差值
                    delta_left = np.array(action[:6]) - current_joints_left[:6]
                    delta_right = np.array(action[7:13]) - current_joints_right[:6]

                    # if (abs(delta_left) > 0.1).any():  # 任一元素 > 0.1 则返回 True
                    #     print("大步限位工作中")

                    # if (abs(delta_left) > 0.1).any():  # 任一元素 > 0.1 则返回 True
                    #     print("大步限位工作中")
                    # 限制变化量大小
                    delta_left = np.clip(delta_left, -max_joint_step, max_joint_step)
                    delta_right = np.clip(delta_right, -max_joint_step, max_joint_step)

                    # 计算平滑后的目标位置
                    target_left = current_joints_left[:6] + delta_left
                    target_right = current_joints_right[:6] + delta_right

                    # 夹爪处理
                    gripper_left = np.clip(action[6], 0, 1)
                    gripper_right = np.clip(action[13], 0, 1)

                    # 将动作分配给不同的机械臂
                    # 组合成最终动作
                    arm_actions = {
                        'left_arm': np.concatenate([target_left, [gripper_left]]),
                        'right_arm': np.concatenate([target_right, [gripper_right]])
                    }
        
                    # 执行动作
                    success = robot_interface.execute_action(arm_actions)
                    
                    if not success:
                        print(f"Failed to execute action at step {step_count + 1}")
                        break
                    
                    print(f"Executed action {i+1}/64: {arm_actions}")
                    
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
        
        # 打印最终推理统计
        if inference_stats['total_inferences'] > 0:
            success_rate = inference_stats['successful_inferences'] / inference_stats['total_inferences'] * 100
            avg_inference_time = inference_stats['total_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
            
            print(f"\n=== Final ONNX Inference Statistics ===")
            print(f"Model type: {'Dual Model (Encoder + Decoder)' if use_dual_model else 'Single Model'}")
            print(f"Total inferences: {inference_stats['total_inferences']}")
            print(f"Successful: {inference_stats['successful_inferences']}")
            print(f"Failed: {inference_stats['failed_inferences']}")
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average total inference time: {avg_inference_time:.4f}s")
            print(f"Total inference time: {inference_stats['total_inference_time']:.2f}s")
            
            if use_dual_model:
                avg_encoder_time = inference_stats['encoder_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
                avg_decoder_time = inference_stats['decoder_inference_time'] / inference_stats['successful_inferences'] if inference_stats['successful_inferences'] > 0 else 0
                print(f"Average encoder time: {avg_encoder_time:.4f}s")
                print(f"Average decoder time: {avg_decoder_time:.4f}s")
                print(f"Total encoder time: {inference_stats['encoder_inference_time']:.2f}s")
                print(f"Total decoder time: {inference_stats['decoder_inference_time']:.2f}s")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        if 'robot_interface' in locals():
            #robot_interface.disconnect()
            pass

        print("Cleanup completed.")

        print("Moving robots to home position before inference...")
        if not robot_interface.move_to_home_position():
            print("Failed to move to home position.")
            return

if __name__ == "__main__":
    main()