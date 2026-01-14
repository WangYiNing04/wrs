# 真实机器人SEM策略部署指南

本文档介绍如何在真实机器人上部署SEM（Sequential Environment Modeling）策略。

## 功能特性

- 支持PiperArmController机械臂控制
- 支持RealSense D400/D400s相机
- 多相机配置支持
- 实时推理和控制
- 安全限幅和错误处理
- 坐标系变换支持

## 依赖要求

### 硬件要求
- Piper机械臂
- RealSense D400/D400s相机
- CAN总线连接

### 软件依赖
```bash
# 安装wrs包（包含PiperArmController和RealSense驱动）
pip install wrs

# 安装RoboOrchardLab
pip install -e .

# 其他依赖
pip install torch torchvision
pip install numpy opencv-python
pip install pyyaml
```

## 配置说明

### 1. 相机配置

创建相机配置文件 `camera_correspondence.yaml`：
```yaml
head_camera:
  ID: "your_head_camera_id"
left_hand_camera:
  ID: "your_left_hand_camera_id"
right_hand_camera:
  ID: "your_right_hand_camera_id"
```

### 2. 模型配置

确保有以下文件：
- SEM模型检查点文件（.safetensors或.pth）
- SEM配置文件（.py）

### 3. 机器人配置

修改 `real_robot_config_example.py` 中的配置：
```python
# 模型路径
MODEL_CONFIG = {
    "ckpt_path": "path/to/your/model.safetensors",
    "config_file": "config_sem_robotwin.py",
}

# 相机配置
CAMERA_CONFIG = {
    "config_path": "/path/to/camera_correspondence.yaml",
    "camera_names": ["left_hand", "right_hand"],
}
```

## 使用方法

### 1. 基本使用

```python
from real_robot_deploy import RealRobotSEMPolicy, RealRobotInterface

# 初始化机器人接口
robot_interface = RealRobotInterface(camera_config_path="camera_correspondence.yaml")

# 连接机器人
if robot_interface.connect():
    # 初始化SEM策略
    policy = RealRobotSEMPolicy("model.safetensors", "config.py")
    
    # 执行任务
    # ... 具体实现见main函数
```

### 2. 运行主程序

```bash
python real_robot_deploy.py
```

### 3. 自定义任务

修改 `main()` 函数中的任务指令：
```python
instruction = "pick up the red block"  # 修改为你的任务
```

## API接口

### RealRobotInterface类

#### 主要方法

- `connect()`: 连接机器人
- `disconnect()`: 断开连接
- `get_joint_positions()`: 获取关节位置
- `get_camera_data(camera_names)`: 获取相机数据
- `execute_action(action)`: 执行动作
- `get_base_to_world_transform()`: 获取坐标系变换

#### 相机数据格式

```python
camera_data = {
    'camera_name': {
        'rgb': np.ndarray,      # RGB图像 (H, W, 3)
        'depth': np.ndarray,    # 深度图像 (H, W)
        'intrinsic': np.ndarray, # 内参矩阵 (3, 3)
        'extrinsic': np.ndarray, # 外参矩阵 (4, 4)
    }
}
```

### RealRobotSEMPolicy类

#### 主要方法

- `predict_action(...)`: 预测动作
- `encode_real_robot_obs(...)`: 编码观测数据
- `get_action(data)`: 获取动作
- `reset()`: 重置策略状态

## 安全注意事项

1. **限幅控制**: 默认启用位置和旋转限幅，避免大步跳变
2. **紧急停止**: 支持Ctrl+C中断
3. **错误处理**: 自动处理连接错误和相机错误
4. **资源清理**: 自动清理机械臂和相机资源

## 故障排除

### 常见问题

1. **机械臂连接失败**
   - 检查CAN总线连接
   - 确认PiperArmController权限
   - 检查机械臂电源状态

2. **相机连接失败**
   - 检查RealSense驱动安装
   - 确认相机ID配置正确
   - 检查USB连接

3. **模型加载失败**
   - 检查模型文件路径
   - 确认配置文件格式
   - 检查CUDA/CPU兼容性

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 添加新相机

1. 在相机配置文件中添加新相机ID
2. 在 `camera_names` 列表中添加相机名称
3. 确保相机驱动支持

### 自定义动作执行

重写 `execute_action` 方法：
```python
def execute_action(self, action):
    # 自定义动作执行逻辑
    pass
```

### 添加安全监控

在控制循环中添加安全检查：
```python
# 检查关节限制
if self._check_joint_limits(joint_positions):
    print("Warning: Joint limits exceeded")
    return False
```

## 性能优化

1. **推理优化**: 使用GPU加速
2. **控制频率**: 根据任务调整控制频率
3. **图像处理**: 优化图像预处理流程
4. **内存管理**: 及时释放不需要的数据

## 许可证

本项目遵循Apache License 2.0许可证。