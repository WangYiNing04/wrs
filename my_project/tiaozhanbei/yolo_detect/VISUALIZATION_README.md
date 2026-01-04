# 3D点云可视化功能说明

## 功能概述

本程序新增了3D点云可视化功能，可以在实时YOLO检测过程中直观地查看筛选后的点云数据。

## 主要特性

### 1. 高度筛选可视化
- 自动筛选高度在7.4-7.6cm范围内的点云
- 在3D散点图中显示筛选后的点云
- 根据高度进行颜色映射，便于观察

### 2. 实时可视化
- 支持实时显示检测结果
- 可控制可视化频率（默认每2秒最多显示一次）
- 支持运行时切换可视化开关

### 3. 交互式控制
- 按 `v` 键：切换3D可视化开关
- 按 `h` 键：显示帮助信息
- 按 `ESC` 键：退出程序

## 使用方法

### 1. 启用可视化功能

在创建检测器时启用可视化：

```python
detector = RealTimeYOLODetector(
    yolo_model_path='your_model.pt',
    check_3D_keypoints=True,
    save_pointcloud=True,
    visualize_3d=True  # 启用3D可视化
)
```

### 2. 运行时控制

程序运行后，可以使用以下键盘控制：

- **ESC**: 退出程序
- **v**: 切换3D可视化开关
- **h**: 显示帮助信息

### 3. 可视化参数

可视化功能支持以下参数：

- `height_min`: 最小高度阈值（默认0.074m = 7.4cm）
- `height_max`: 最大高度阈值（默认0.076m = 7.6cm）
- `camera_role`: 相机标识（用于区分不同相机的点云）

## 可视化特性

### 1. 颜色映射
- 点云根据Z坐标（高度）进行颜色映射
- 使用Viridis颜色方案
- 颜色条显示高度范围

### 2. 坐标轴
- X轴：水平方向（米）
- Y轴：水平方向（米）
- Z轴：高度方向（米）

### 3. 显示信息
- 图表标题包含相机信息和点云数量
- 显示高度筛选范围
- 实时更新点云统计

## 测试功能

运行测试脚本验证可视化功能：

```bash
python test_visualization.py
```

这将生成1000个测试点云点并显示3D可视化。

## 依赖库

确保安装以下Python库：

```bash
pip install matplotlib numpy
```

可选（用于交互式可视化）：
```bash
pip install plotly
```

## 注意事项

1. **性能考虑**: 可视化功能会增加计算开销，建议根据需要开启
2. **频率控制**: 默认每2秒最多显示一次可视化，避免过于频繁
3. **内存使用**: 大量点云数据可能占用较多内存
4. **显示模式**: 使用非阻塞显示模式，不会影响主程序运行

## 故障排除

### 1. 可视化不显示
- 检查matplotlib是否正确安装
- 确认可视化开关已启用
- 检查是否有检测到的目标

### 2. 性能问题
- 减少可视化频率
- 降低点云密度
- 关闭不必要的功能

### 3. 显示异常
- 检查点云数据是否有效
- 确认坐标系统正确
- 查看控制台错误信息

## 扩展功能

### 1. 交互式可视化
程序支持Plotly交互式可视化（需要安装plotly）：

```python
# 使用交互式可视化
detector.visualize_pointcloud_3d_interactive(pointcloud, ...)
```

### 2. 自定义可视化
可以修改可视化参数：

```python
# 自定义高度范围
bbox_pcd_world, original_count, filtered_count = detector.extract_bbox_pointcloud(
    results[0].boxes, 
    pcd, 
    color_img.shape,
    height_min=0.070,  # 7.0cm
    height_max=0.080   # 8.0cm
)
```

## 技术细节

- 使用matplotlib的3D散点图功能
- 支持颜色映射和交互式缩放
- 自动调整坐标轴比例
- 非阻塞显示模式




