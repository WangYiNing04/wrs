import os
import cv2
import time
import yaml
import threading
import queue
from datetime import datetime
from pathlib import Path
import numpy as np
from wrs.drivers.devices.realsense.realsense_d400s import *
from ultralytics import YOLO  # 导入YOLO模型
from typing import Optional, List, Tuple
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RealTimeYOLODetector:
    """实时YOLO检测器，支持多相机同时预览和YOLO实时推理"""
    
<<<<<<< HEAD
    def __init__(self, config_path=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/config/camera_correspondence.yaml', 
=======
    def __init__(self, config_path=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\yolo_detect\config\camera_correspondence.yaml', 
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
                 yolo_model_path='yolov8n.pt',check_3D_keypoints=False, save_pointcloud=False, visualize_3d=False):
        """
        初始化实时检测器
        
        Args:
            config_path: 相机配置文件路径
            yolo_model_path: YOLO模型路径
        """
        self.config_path = config_path
        self.rs_pipelines = {}
        self.detection_active = False
        self.key_press_times = {}
        self.key_repeat_threshold = 0.1  # 100ms内重复按键视为重复
        self.check_3D_keypoints = check_3D_keypoints
        self.save_pointcloud = save_pointcloud
        self.visualize_3d = visualize_3d
        self.last_visualization_time = 0  # 用于控制可视化频率
        
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        print(f"已加载YOLO模型: {yolo_model_path}")
        
        # 初始化相机
        self.initialize_cameras()

        if self.check_3D_keypoints:
            print("3D关键点检测已启用")

        #middle camera hand-eye matrix
        self._init_calib_mat = np.array([[0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
                                            [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
                                            [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
                                            [0.0, 0.0, 0.0, 1.0]]
                                        )

    def initialize_cameras(self):
        """初始化相机"""
        try:
            # 读取YAML配置文件
            with open(self.config_path, 'r') as file:
                camera_config = yaml.safe_load(file)

            # 从配置中提取相机ID
            camera_roles = {
                'middle': camera_config['middle_camera']['ID'],
                #'left': camera_config['left_camera']['ID'],
                #'right': camera_config['right_camera']['ID']
            }

            # 查找实际连接的设备
            available_serials, ctx = find_devices()
            print("检测到设备:", available_serials)

            # 初始化相机（用字典存储，键为角色名称）
            for role, cam_id in camera_roles.items():
                if cam_id in available_serials:
                    print(f"正在初始化 {role} 相机 (ID: {cam_id})")
                    pipeline = RealSenseD400(device=cam_id)
                    pipeline.reset()
                    time.sleep(2)
                    pipeline = RealSenseD400(device=cam_id)  # 重新初始化
                    self.rs_pipelines[role] = pipeline
                    print(f"{role} 相机初始化成功")
                else:
                    print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")
                    
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise
    
    def run_yolo_inference(self, image):
        """
        使用YOLO模型进行推理
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 带标注的图像
            results: 推理结果
        """
        if image is None:
            return None, None
            
        # 运行推理，设置置信度阈值为0.9
        results = self.yolo_model(image, conf=0.9)

    
        # 渲染结果
        annotated_image = results[0].plot(
            boxes=True,
            labels=True,
        )
        
        return annotated_image, results
    
    def start_detection_mode(self):
        """
        启动检测模式，支持实时预览和键盘控制
        控制说明：
        - ESC键：退出程序
        """
        print("启动实时YOLO检测模式...")
        print("控制说明：")
        print("- ESC键：退出程序")
        
        self.detection_active = True
        
        try:
            # 实时显示画面
            while self.detection_active:
                display_images = {}
                
                for role, pipeline in self.rs_pipelines.items():
                    try:
                        pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                        # 使用YOLO进行推理
                        annotated_img, results = self.run_yolo_inference(color_img)
                        
                        #world2cam
                        pcd = self.align_pcd(pcd)
                        
                        # 裁剪世界坐标系下的点云到指定范围
                        cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
                            pcd, 
                            x_range=(0, 0.6), 
                            y_range=(0, -0.6), 
                            z_range=(0.07, 0.08)
                        )
                        
                        # 打印裁剪后的点云（按高度排序）并计算中心点
                        if cropped_pcd is not None and len(cropped_pcd) > 0:
                            self.print_cropped_pointcloud_with_center(cropped_pcd, role)
                            
                            # 可视化裁剪后的点云（控制频率，避免过于频繁）
                            current_time = time.time()
                            if current_time - self.last_visualization_time > 3.0:  # 每3秒最多可视化一次
                                self.visualize_pointcloud_3d(
                                    cropped_pcd, 
                                    title=f"裁剪后点云 - {role}相机",
                                    height_min=0.07,
                                    height_max=0.08,
                                    camera_role=role
                                )
                                self.last_visualization_time = current_time
                        
        
                    except Exception as e:
                        print(f"从 {role} 相机获取图像失败: {e}")
                        display_images[role] = None
                
        except Exception as e:
            print(f"检测过程中发生错误: {e}")
        finally:
            # 清理资源
            self.cleanup()

   

    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 停止所有相机
        for pipeline in self.rs_pipelines.values():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"停止相机时出错: {e}")
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("程序已退出")

      
    def align_pcd(self, pcd):
        c2w_mat = self._init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
    def print_cropped_pointcloud_with_center(self, cropped_pcd, camera_role):
        """
        打印裁剪后的点云，按高度排序，并计算中心点
        
        Args:
            cropped_pcd: 裁剪后的点云数据 (N, 3)
            camera_role: 相机角色名称
        """
        if cropped_pcd is None or len(cropped_pcd) == 0:
            print(f"[{camera_role}相机] 没有裁剪后的点云数据")
            return
            
        print(f"\n=== [{camera_role}相机] 裁剪后点云详细信息 ===")
        print(f"点云数量: {len(cropped_pcd)}")
        
        # 按高度（Z坐标）降序排序，优先显示高度高的点
        sorted_indices = np.argsort(cropped_pcd[:, 2])[::-1]  # 降序排序
        sorted_pcd = cropped_pcd[sorted_indices]
        
        # 打印前20个最高点（避免输出过多）
        print(f"\n前20个最高点（按高度降序）:")
        print("序号    X坐标(m)    Y坐标(m)    Z坐标(m)    高度(cm)")
        print("-" * 60)
        for i in range(min(20, len(sorted_pcd))):
            point = sorted_pcd[i]
            print(f"{i+1:3d}    {point[0]:8.4f}    {point[1]:8.4f}    {point[2]:8.4f}    {point[2]*100:6.2f}")
        
        if len(sorted_pcd) > 20:
            print(f"... (还有 {len(sorted_pcd) - 20} 个点未显示)")
        
        # 计算中心点
        center_point = np.mean(cropped_pcd, axis=0)
        print(f"\n点云中心点:")
        print(f"  X: {center_point[0]:.4f} m")
        print(f"  Y: {center_point[1]:.4f} m") 
        print(f"  Z: {center_point[2]:.4f} m ({center_point[2]*100:.2f} cm)")
        
        # 计算点云范围
        min_coords = np.min(cropped_pcd, axis=0)
        max_coords = np.max(cropped_pcd, axis=0)
        print(f"\n点云范围:")
        print(f"  X: [{min_coords[0]:.4f}, {max_coords[0]:.4f}] m (跨度: {max_coords[0]-min_coords[0]:.4f} m)")
        print(f"  Y: [{min_coords[1]:.4f}, {max_coords[1]:.4f}] m (跨度: {max_coords[1]-min_coords[1]:.4f} m)")
        print(f"  Z: [{min_coords[2]:.4f}, {max_coords[2]:.4f}] m (跨度: {max_coords[2]-min_coords[2]:.4f} m)")
        
        # 计算高度统计
        heights = cropped_pcd[:, 2]
        print(f"\n高度统计:")
        print(f"  平均高度: {np.mean(heights):.4f} m ({np.mean(heights)*100:.2f} cm)")
        print(f"  最高点: {np.max(heights):.4f} m ({np.max(heights)*100:.2f} cm)")
        print(f"  最低点: {np.min(heights):.4f} m ({np.min(heights)*100:.2f} cm)")
        print(f"  高度标准差: {np.std(heights):.4f} m ({np.std(heights)*100:.2f} cm)")
        
        print("=" * 50)

    def crop_pointcloud_world(self, pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
        """
        在世界坐标系下裁剪点云到指定范围
        
        Args:
            pcd_world: 世界坐标系下的点云数据 (N, 3)
            x_range: X轴范围 (min, max)，默认(0, 0.6)
            y_range: Y轴范围 (min, max)，默认(0, -0.6) 
            z_range: Z轴范围 (min, max)，默认(0.07, 0.08)
            
        Returns:
            tuple: (裁剪后的点云数据, 原始点云数量, 裁剪后点云数量)
        """
        if pcd_world is None or len(pcd_world) == 0:
            print("输入点云为空")
            return None, 0, 0
            
        # 记录原始点云数量
        original_count = len(pcd_world)
        
        # 提取坐标
        x = pcd_world[:, 0]
        y = pcd_world[:, 1] 
        z = pcd_world[:, 2]
        
        # 创建裁剪掩码
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        y_mask = (y >= y_range[1]) & (y <= y_range[0])  # 注意Y轴范围是(0, -0.6)，所以是y >= -0.6 and y <= 0
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        
        # 组合所有掩码
        combined_mask = x_mask & y_mask & z_mask
        
        # 应用掩码裁剪点云
        cropped_pcd = pcd_world[combined_mask]
        cropped_count = len(cropped_pcd)
        
        # 打印统计信息
        print(f"点云裁剪统计:")
        print(f"  原始点云数量: {original_count}")
        print(f"  裁剪后点云数量: {cropped_count}")
        print(f"  裁剪范围: X[{x_range[0]}, {x_range[1]}], Y[{y_range[1]}, {y_range[0]}], Z[{z_range[0]}, {z_range[1]}]")
        print(f"  保留比例: {cropped_count/original_count*100:.2f}%")
        
        if cropped_count > 0:
            # 打印裁剪后点云的坐标范围
            print(f"  裁剪后点云范围:")
            print(f"    X: [{cropped_pcd[:, 0].min():.4f}, {cropped_pcd[:, 0].max():.4f}]")
            print(f"    Y: [{cropped_pcd[:, 1].min():.4f}, {cropped_pcd[:, 1].max():.4f}]")
            print(f"    Z: [{cropped_pcd[:, 2].min():.4f}, {cropped_pcd[:, 2].max():.4f}]")
        else:
            print("  警告: 裁剪后没有剩余点云")
            
        return cropped_pcd, original_count, cropped_count
    
   
    
    def save_bbox_pointcloud(self, pointcloud, camera_role, save_dir="pointcloud_output"):
        """
        保存目标框内的点云到文件
        
        Args:
            pointcloud: 点云数据 (N, 3)
            camera_role: 相机角色名称
            save_dir: 保存目录
        """
        import os
        from datetime import datetime
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_role}_bbox_pointcloud_{timestamp}.ply"
        filepath = os.path.join(save_dir, filename)
        
        # 保存为PLY格式
        self.save_ply(pointcloud, filepath)
        print(f"点云已保存到: {filepath}")
    
    def save_ply(self, points, filename):
        """
        保存点云为PLY格式
        
        Args:
            points: 点云数据 (N, 3)
            filename: 保存路径
        """
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            
            for point in points:
                f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n')
    
    def visualize_pointcloud_3d(self, pointcloud, title="3D Point Cloud Visualization", 
                               height_min=0.074, height_max=0.076, camera_role="unknown"):
        """
        在3D散点图中可视化点云数据
        
        Args:
            pointcloud: 点云数据 (N, 3)
            title: 图表标题
            height_min: 最小高度阈值 (米)
            height_max: 最大高度阈值 (米)
            camera_role: 相机角色标识
        """
        if pointcloud is None or len(pointcloud) == 0:
            print("没有点云数据可以可视化")
            return
            
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取坐标
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        
        # 根据高度着色
        colors = z  # 使用Z坐标作为颜色映射
        
        # 绘制3D散点图
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1, alpha=0.6)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 设置标题
        ax.set_title(f'{title}\n相机: {camera_role}, 点云数量: {len(pointcloud)}\n高度范围: {height_min*100:.1f}-{height_max*100:.1f}cm')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('高度 (m)')
        
        # 设置相等的坐标轴比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 显示图形
        plt.tight_layout()
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.1)  # 短暂暂停以确保图形显示
        
        print(f"3D可视化已显示: {len(pointcloud)} 个点云点")
    
    def visualize_pointcloud_3d_interactive(self, pointcloud, title="3D Point Cloud Visualization", 
                                          height_min=0.074, height_max=0.076, camera_role="unknown"):
        """
        交互式3D点云可视化（可选功能）
        
        Args:
            pointcloud: 点云数据 (N, 3)
            title: 图表标题
            height_min: 最小高度阈值 (米)
            height_max: 最大高度阈值 (米)
            camera_role: 相机角色标识
        """
        if pointcloud is None or len(pointcloud) == 0:
            print("没有点云数据可以可视化")
            return
            
        try:
            # 尝试使用plotly进行交互式可视化
            import plotly.graph_objects as go
            import plotly.express as px
            
            # 创建交互式3D散点图
            fig = go.Figure(data=[go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=pointcloud[:, 2],  # 根据高度着色
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="高度 (m)")
                ),
                text=[f'点 {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})' for i, p in enumerate(pointcloud)],
                hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )])
            
            # 更新布局
            fig.update_layout(
                title=f'{title}<br>相机: {camera_role}, 点云数量: {len(pointcloud)}<br>高度范围: {height_min*100:.1f}-{height_max*100:.1f}cm',
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='cube'
                ),
                width=800,
                height=600
            )
            
            # 显示图形
            fig.show()
            print(f"交互式3D可视化已显示: {len(pointcloud)} 个点云点")
            
        except ImportError:
            print("Plotly未安装，使用matplotlib进行可视化")
            self.visualize_pointcloud_3d(pointcloud, title, height_min, height_max, camera_role)
        except Exception as e:
            print(f"交互式可视化失败: {e}")
            print("回退到matplotlib可视化")
            self.visualize_pointcloud_3d(pointcloud, title, height_min, height_max, camera_role)



def main():
    """主函数"""
    try:
        # 创建检测器
        detector = RealTimeYOLODetector(
<<<<<<< HEAD
            yolo_model_path=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/empty_cup_place/best.pt',  # 可以替换为您自己的模型路径
=======
            yolo_model_path=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\yolo_detect\model\empty_cup_place\best.pt',  # 可以替换为您自己的模型路径
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            check_3D_keypoints=False,
            save_pointcloud=False,  # 启用点云保存功能
            visualize_3d=False  # 启用3D可视化功能
        )
        
        # 启动检测模式
        detector.start_detection_mode()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()