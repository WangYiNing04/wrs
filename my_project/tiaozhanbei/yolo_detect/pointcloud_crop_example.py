#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云裁剪功能使用示例
展示如何在检测流程中使用crop_pointcloud_world函数
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def demonstrate_crop_usage():
    """演示点云裁剪功能的使用方法"""
    print("=== 点云裁剪功能使用示例 ===\n")
    
    # 模拟RealTimeYOLODetector类中的crop_pointcloud_world方法
    def crop_pointcloud_world(pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
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
        y_mask = (y >= y_range[1]) & (y <= y_range[0])  # 注意Y轴范围是(0, -0.6)
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
    
    # 示例1: 基本使用
    print("1. 基本使用示例:")
    print("-" * 40)
    
    # 生成测试点云
    np.random.seed(42)
    n_points = 5000
    x = np.random.uniform(-0.5, 1.0, n_points)
    y = np.random.uniform(-1.0, 0.5, n_points)
    z = np.random.uniform(0.05, 0.15, n_points)
    test_pcd = np.column_stack([x, y, z])
    
    # 使用默认参数裁剪
    cropped_pcd, orig_count, crop_count = crop_pointcloud_world(test_pcd)
    
    print(f"\n2. 自定义裁剪范围示例:")
    print("-" * 40)
    
    # 使用自定义参数裁剪
    cropped_pcd2, orig_count2, crop_count2 = crop_pointcloud_world(
        test_pcd,
        x_range=(0.1, 0.5),    # X: 0.1 到 0.5
        y_range=(0, -0.4),     # Y: 0 到 -0.4
        z_range=(0.08, 0.12)   # Z: 0.08 到 0.12
    )
    
    print(f"\n3. 在检测流程中的使用:")
    print("-" * 40)
    print("在detect.py中的使用方式:")
    print("""
    # 获取点云并转换到世界坐标系
    pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
    pcd = self.align_pcd(pcd)  # 转换到世界坐标系
    
    # 裁剪点云到指定范围
    cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
        pcd, 
        x_range=(0, 0.6), 
        y_range=(0, -0.6), 
        z_range=(0.07, 0.08)
    )
    
    # 在YOLO检测中使用裁剪后的点云
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        bbox_pcd_world, bbox_original_count, bbox_filtered_count = self.extract_bbox_pointcloud(
            results[0].boxes, 
            cropped_pcd if cropped_pcd is not None else pcd,  # 使用裁剪后的点云
            color_img.shape,
            height_min=0.074,
            height_max=0.076
        )
    """)
    
    print(f"\n4. 可视化结果:")
    print("-" * 40)
    
    # 可视化
    if cropped_pcd is not None and len(cropped_pcd) > 0:
        fig = plt.figure(figsize=(12, 4))
        
        # 原始点云
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(test_pcd[:, 0], test_pcd[:, 1], test_pcd[:, 2], 
                   c=test_pcd[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax1.set_title(f'原始点云\n{len(test_pcd)} 个点')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        
        # 默认裁剪
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(cropped_pcd[:, 0], cropped_pcd[:, 1], cropped_pcd[:, 2], 
                   c=cropped_pcd[:, 2], cmap='viridis', s=1, alpha=0.6)
        ax2.set_title(f'默认裁剪\n{len(cropped_pcd)} 个点')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_zlabel('Z (m)')
        
        # 自定义裁剪
        if cropped_pcd2 is not None and len(cropped_pcd2) > 0:
            ax3 = fig.add_subplot(133, projection='3d')
            ax3.scatter(cropped_pcd2[:, 0], cropped_pcd2[:, 1], cropped_pcd2[:, 2], 
                       c=cropped_pcd2[:, 2], cmap='viridis', s=1, alpha=0.6)
            ax3.set_title(f'自定义裁剪\n{len(cropped_pcd2)} 个点')
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_zlabel('Z (m)')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n=== 示例完成 ===")
    print(f"成功演示了点云裁剪功能的使用方法")

if __name__ == "__main__":
    demonstrate_crop_usage()




