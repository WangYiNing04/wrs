# 三个方块堆叠任务

这个目录包含了三个方块堆叠任务的完整实现，基于 `empty_cup_place` 目录的代码结构。

## 任务描述

按照以下顺序执行方块堆叠：
1. **红色方块** → 放到中间位置 (0, 0, 0.05)
2. **绿色方块** → 放到红色方块上面 (0, 0, 0.08)  
3. **蓝色方块** → 放到绿色方块上面 (0, 0, 0.11)

## 文件说明

- `deploy.py`: 主要的任务执行脚本
- `README.md`: 使用说明文档

## 功能特性

### 1. 智能方块检测
- 使用YOLO模型检测红、绿、蓝三个方块
- 自动获取方块的3D位置坐标
- 支持置信度阈值过滤

### 2. 双臂协调控制
- 根据方块位置自动选择左臂或右臂
- 支持双臂机器人的协调工作
- 智能避障和路径规划

### 3. 抓取姿态规划
- 自动生成方块的抓取姿态集合
- 使用优化后的抓取参数（min_dist_between_sampled_contact_points=0.03）
- 支持Piper夹爪的精确抓取

### 4. 运动轨迹生成
- 自动生成Pick-and-Place运动轨迹
- 导出关节角轨迹到JSON文件
- 支持真实机械臂执行

## 使用方法

### 基本使用
```bash
cd /home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/stack_blocks_three
python deploy.py
```

### 前置条件

1. **硬件要求**：
   - 双臂Piper机器人
   - RealSense相机
   - 三个不同颜色的方块（红、绿、蓝）

2. **软件依赖**：
   - wrs机器人仿真库
   - ultralytics (YOLO)
   - opencv-python
   - numpy
   - pyrealsense2

3. **配置文件**：
   - 相机配置文件：`/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/config/camera_correspondence.yaml`
   - 方块模型文件：`/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/block.stl`

## 主要类和方法

### `BlockDetector` 类
- `detect_blocks()`: 检测三个方块的位置
- 使用YOLO模型进行物体检测
- 自动转换相机坐标到世界坐标

### `BlockStackingTask` 类
- `run_stacking_task()`: 执行完整的堆叠任务
- `execute_pick_place()`: 执行单个方块的抓取和放置
- `choose_arm()`: 根据位置选择机械臂
- `create_grasp_collection()`: 生成抓取姿态集合

## 工作流程

1. **初始化**：
   - 初始化双臂机械臂控制器
   - 初始化相机和YOLO模型
   - 设置目标位置和抓取参数

2. **检测阶段**：
   - 使用相机获取点云和图像数据
   - YOLO模型检测三个方块
   - 计算方块的世界坐标位置

3. **规划阶段**：
   - 生成方块抓取姿态集合
   - 为每个方块规划Pick-and-Place轨迹
   - 选择合适的机械臂

4. **执行阶段**：
   - 按顺序执行三个方块的抓取和放置
   - 实时控制机械臂运动
   - 监控任务执行状态

## 参数配置

### 目标位置设置
```python
self.target_positions = {
    'red': [0.0, 0.0, 0.05],      # 红色方块放在中间
    'green': [0.0, 0.0, 0.08],    # 绿色方块放在红色方块上面
    'blue': [0.0, 0.0, 0.11]      # 蓝色方块放在绿色方块上面
}
```

### 抓取参数优化
```python
min_dist_between_sampled_contact_points=.03  # 使用优化后的参数
```

### 机械臂选择逻辑
```python
if block_pos[1] > -0.3:  # Y坐标大于-0.3使用左臂
    return self.left_arm_con, psa.PiperSglArm()
else:  # 否则使用右臂
    return self.right_arm_con, psa.PiperSglArm(pos=[0, -0.6, 0])
```

## 输出文件

- `block_grasps.pickle`: 方块抓取姿态集合
- `exported/joint_trajectory_blocks_*.json`: 关节角轨迹文件

## 故障排除

### 常见问题

1. **相机初始化失败**：
   - 检查相机连接
   - 确认配置文件路径正确
   - 检查相机权限

2. **YOLO检测失败**：
   - 确认模型文件存在
   - 检查图像质量和光照条件
   - 调整置信度阈值

3. **机械臂控制失败**：
   - 检查机械臂连接
   - 确认CAN总线通信
   - 检查安全限制

4. **抓取规划失败**：
   - 检查方块模型文件
   - 调整抓取参数
   - 检查碰撞检测

### 调试建议

1. **启用详细日志**：
   ```python
   print(f"检测到{color_name}方块: 位置={world_point}, 置信度={conf:.2f}")
   ```

2. **可视化检测结果**：
   - 在检测阶段添加图像显示
   - 保存检测结果图像

3. **轨迹验证**：
   - 在仿真环境中验证轨迹
   - 检查关节角度限制

## 扩展功能

### 支持更多方块
- 修改 `color_mapping` 字典
- 添加新的目标位置
- 更新任务执行顺序

### 自定义抓取策略
- 修改抓取参数
- 添加抓取姿态过滤
- 实现动态抓取选择

### 增强检测能力
- 训练专用的YOLO模型
- 添加颜色检测
- 实现形状识别

## 注意事项

1. **安全第一**：确保工作区域安全，避免碰撞
2. **参数调优**：根据实际环境调整检测和抓取参数
3. **错误处理**：实现完善的异常处理机制
4. **性能优化**：根据实际需求调整检测频率和精度

## 技术支持

如有问题，请联系：
- 邮箱：wangyining0408@outlook.com
- 作者：wang yining

