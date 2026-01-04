# RGB图片采集脚本使用说明

基于DataCollector.py中的相机相关代码，我为你创建了两个RGB图片采集脚本：

## 脚本文件

1. **rgb_image_collector.py** - 完整版采集器（需要配置文件）
2. **simple_rgb_collector.py** - 简化版采集器（自动检测相机）

## 功能特性

### 主要功能
- 支持多相机同时采集RGB图片
- 实时预览所有相机画面
- 键盘控制采集操作
- 自动保存图片到指定目录
- 支持单帧采集和连续采集模式

### 键盘控制
- **空格键**: 采集当前帧
- **'c'键**: 开始连续采集模式（每秒采集一次）
- **'s'键**: 停止连续采集模式
- **ESC键**: 退出程序

## 使用方法

### 方法1：使用完整版采集器（推荐）

```bash
python rgb_image_collector.py
```

**要求**：
- 需要 `./config/camera_correspondence.yaml` 配置文件
- 配置文件格式：
```yaml
head_camera:
  ID: "your_head_camera_id"
left_hand_camera:
  ID: "your_left_hand_camera_id"
right_hand_camera:
  ID: "your_right_hand_camera_id"
```

### 方法2：使用简化版采集器

```bash
python simple_rgb_collector.py
```

**特点**：
- 自动检测所有可用相机
- 不需要配置文件
- 适合快速测试和简单采集

## 输出文件结构

```
collected_images/
├── head/                    # 头部相机图片
│   ├── head_20231201_143022_001_000001.jpg
│   └── ...
├── left_hand/              # 左手相机图片
│   ├── left_hand_20231201_143022_001_000001.jpg
│   └── ...
└── right_hand/             # 右手相机图片
    ├── right_hand_20231201_143022_001_000001.jpg
    └── ...
```

## 依赖库

确保安装以下Python库：
```bash
pip install opencv-python
pip install pyrealsense2
pip install pynput
pip install pyyaml
pip install numpy
```

## 注意事项

1. **相机初始化**：脚本会自动检测和初始化RealSense相机
2. **图片格式**：保存为JPG格式，自动处理RGB到BGR的转换
3. **文件命名**：使用时间戳和计数器确保文件名唯一
4. **资源清理**：程序退出时会自动清理相机资源
5. **错误处理**：包含完善的异常处理机制

## 故障排除

### 常见问题

1. **找不到相机设备**
   - 检查RealSense相机是否正确连接
   - 确认相机驱动已安装
   - 尝试使用简化版采集器

2. **配置文件错误**
   - 检查YAML文件格式是否正确
   - 确认相机ID是否匹配实际设备
   - 使用简化版采集器绕过配置文件

3. **权限问题**
   - 确保有权限访问相机设备
   - 在Linux系统上可能需要添加用户到video组

4. **内存不足**
   - 连续采集模式会占用较多内存
   - 可以调整采集频率或使用单帧采集

## 扩展功能

可以根据需要添加以下功能：
- 深度图像采集
- 点云数据保存
- 图像预处理（去噪、增强等）
- 批量处理功能
- 网络传输功能

## 技术支持

如有问题，请检查：
1. 相机连接状态
2. 依赖库版本
3. 系统权限设置
4. 错误日志输出

