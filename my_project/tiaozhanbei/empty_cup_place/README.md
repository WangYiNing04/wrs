# 集成抓取规划和Pick-and-Place任务

这个目录包含了将抓取规划和Pick-and-Place任务结合在一起的完整解决方案。

## 文件说明

- `integrated_pick_place.py`: 主要的集成文件，包含完整的抓取规划和Pick-and-Place任务流程

## 功能特性

### 1. 自动抓取姿态生成
- 为指定物体自动生成抓取姿态集合
- 支持Piper夹爪的抓取规划
- 自动保存和加载抓取姿态数据

### 2. 完整的Pick-and-Place任务
- 从起始位置抓取物体
- 移动到目标位置
- 放置物体到指定位置
- 支持障碍物避障

### 3. 可视化演示
- 3D环境可视化
- 动画演示运动轨迹
- 交互式控制（空格键逐步播放）

## 使用方法

### 基本使用
```python
python integrated_pick_place.py
```

### 自定义参数
可以修改 `main()` 函数中的参数：

```python
# 物体路径
obj_path = r"path/to/your/object.stl"

# 起始和目标位置
start_pos = [x1, y1, z1]  # 起始位置
goal_pos = [x2, y2, z2]   # 目标位置

# 旋转矩阵
start_rot = rm.rotmat_from_euler(rx, ry, rz)  # 起始旋转
goal_rot = rm.rotmat_from_euler(rx, ry, rz)   # 目标旋转
```

## 主要函数

### `create_grasp_collection(obj_path, save_path, gripper=None)`
为指定物体创建抓取姿态集合

### `run_pick_place_task(obj_path, grasp_collection_path, start_pos, goal_pos, ...)`
执行Pick-and-Place任务

### `animate_motion(mot_data, base)`
动画显示运动轨迹

## 依赖项

- wrs (机器人仿真库)
- numpy
- direct (Panda3D)

## 注意事项

1. 确保物体STL文件路径正确
2. 首次运行会自动生成抓取姿态文件
3. 后续运行会使用已保存的抓取姿态文件
4. 按空格键可以逐步播放动画
