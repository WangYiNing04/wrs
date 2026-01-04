import os
import time
import yaml
import numpy as np
import wrs.basis.robot_math as rm
from wrs.drivers.devices.realsense.realsense_d400s import *


class PointCloudProcessor:
    """最小化点云处理器，实现世界坐标系转换和裁剪功能"""
    
    def __init__(self, config_path=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\yolo_detect\config\camera_correspondence.yaml'):
        # middle camera hand-eye matrix (相机到世界的变换矩阵)
        self._init_calib_mat = np.array([
            [0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
            [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
            [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 相机相关属性
        self.config_path = config_path
        self.rs_pipelines = {}
        self.camera_active = False
        
        # 初始化相机
        self.initialize_cameras()
    
    def align_pcd(self, pcd):
        """
        将点云从相机坐标系转换到世界坐标系
        
        Args:
            pcd: 相机坐标系下的点云数据 (N, 3)
            
        Returns:
            np.ndarray: 世界坐标系下的点云数据 (N, 3)
        """
        c2w_mat = self._init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
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
    
    def process_pointcloud(self, pcd_camera):
        """
        完整的点云处理流程：相机坐标系 -> 世界坐标系 -> 裁剪
        
        Args:
            pcd_camera: 相机坐标系下的点云数据 (N, 3)
            
        Returns:
            tuple: (裁剪后的点云数据, 原始点云数量, 裁剪后点云数量)
        """
        # 步骤1: 转换到世界坐标系
        pcd_world = self.align_pcd(pcd_camera)
        
        # 步骤2: 裁剪到指定范围
        cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
            pcd_world, 
            x_range=(0, 0.6), 
            y_range=(0, -0.6), 
            z_range=(0.07, 0.08)
        )
        
        return cropped_pcd, original_count, cropped_count
    
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
    
    def get_camera_data(self, role='middle'):
        """
        从指定相机获取点云和图像数据
        
        Args:
            role: 相机角色名称，默认'middle'
            
        Returns:
            tuple: (点云数据, 彩色点云, 深度图, 彩色图) 或 (None, None, None, None) 如果失败
        """
        if role not in self.rs_pipelines:
            print(f"错误: 未找到 {role} 相机")
            return None, None, None, None
            
        try:
            pcd, pcd_color, depth_img, color_img = self.rs_pipelines[role].get_pcd_texture_depth()
            return pcd, pcd_color, depth_img, color_img
        except Exception as e:
            print(f"从 {role} 相机获取数据失败: {e}")
            return None, None, None, None
    
    def start_camera_stream(self):
        """启动相机流并开始实时处理"""
        print("启动相机流...")
        print("控制说明：")
        print("- Ctrl+C: 退出程序")
        
        self.camera_active = True
        
        try:
            while self.camera_active:
                for role, pipeline in self.rs_pipelines.items():
                    try:
                        # 获取相机数据
                        pcd, pcd_color, depth_img, color_img = self.get_camera_data(role)
                        
                        if pcd is not None:
                            # 处理点云：相机坐标系 -> 世界坐标系 -> 裁剪
                            cropped_pcd, original_count, cropped_count = self.process_pointcloud(pcd)
                            
                            if cropped_pcd is not None and len(cropped_pcd) > 0:
                                print(f"[{role}相机] 处理完成: {len(cropped_pcd)} 个点")
                            else:
                                print(f"[{role}相机] 没有符合条件的点云")
                        else:
                            print(f"[{role}相机] 获取点云失败")
                            
                    except Exception as e:
                        print(f"处理 {role} 相机数据时出错: {e}")
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"相机流处理过程中发生错误: {e}")
        finally:
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
        
        self.camera_active = False
        print("程序已退出")


def main():
    """主函数 - 启动真实相机流"""
    try:
        # 创建处理器（会自动初始化相机）
        processor = PointCloudProcessor()
        
        # 启动相机流
        processor.start_camera_stream()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()
