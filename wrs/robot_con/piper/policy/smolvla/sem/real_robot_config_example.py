# 真实机器人SEM策略配置示例
# 请根据实际环境修改以下配置

# 模型和配置文件路径
MODEL_CONFIG = {
    "ckpt_path": "path/to/your/model.safetensors",  # SEM模型检查点路径
    "config_file": "config_sem_robotwin.py",  # SEM配置文件路径
}

# 机器人配置
ROBOT_CONFIG = {
    "can_name": "can0",  # CAN总线名称
    "has_gripper": True,  # 是否有夹爪
    "auto_enable": True,  # 自动启用
}

# 相机配置
CAMERA_CONFIG = {
    "config_path": "/path/to/camera_correspondence.yaml",  # 相机配置文件路径
    "camera_names": ["left_hand", "right_hand"],  # 使用的相机名称
    "realsense_type": "d400s",  # RealSense类型: "d400" 或 "d400s"
}

# 控制参数
CONTROL_CONFIG = {
    "step_hz": 5.0,  # 控制频率 (Hz)
    "max_xyz_step": 0.01,  # 每步最大位置增量 (m)
    "max_rpy_step": 2.0,  # 每步最大旋转增量 (度)
    "max_steps": 1000,  # 最大执行步数
}

# 任务配置
TASK_CONFIG = {
    "instruction": "place the empty cup on the table",  # 任务指令
    "visualize_cameras": False,  # 是否显示相机画面
}

# 坐标系变换配置
TRANSFORM_CONFIG = {
    "base_to_world": [
        [0, -1, 0, 0],
        [1, 0, 0, -0.65],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],  # 基座到世界坐标系的变换矩阵
}

# 安全配置
SAFETY_CONFIG = {
    "enable_limits": True,  # 启用限幅
    "emergency_stop_key": "q",  # 紧急停止按键
    "log_level": "INFO",  # 日志级别
}