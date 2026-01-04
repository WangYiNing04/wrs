#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裁剪后点云可视化测试脚本
测试crop_pointcloud_world函数和可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def test_cropped_pointcloud_visualization():
    """测试裁剪后点云的可视化功能"""
    print("=== 裁剪后点云可视化测试 ===\n")
    
    # 模拟RealTimeYOLODetector类中的crop_pointcloud_world方法
    def crop_pointcloud_world(pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
        """模拟点云裁剪函数"""
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
    
    # 模拟visualize_pointcloud_3d方法
    def visualize_pointcloud_3d(pointcloud, title="3D Point Cloud Visualization", 
                               height_min=0.07, height_max=0.08, camera_role="unknown"):
        """3D点云可视化函数"""
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
    
    # 生成测试点云数据
    print("1. 生成测试点云数据...")
    np.random.seed(42)
    n_points = 20000
    
    # 生成在较大范围内的随机点云
    x = np.random.uniform(-0.8, 1.2, n_points)  # X: -0.8 到 1.2
    y = np.random.uniform(-1.2, 0.8, n_points)  # Y: -1.2 到 0.8  
    z = np.random.uniform(0.05, 0.15, n_points) # Z: 0.05 到 0.15
    
    # 创建点云数组
    pointcloud = np.column_stack([x, y, z])
    
    print(f"生成了 {len(pointcloud)} 个测试点云点")
    print(f"原始点云范围:")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Z: [{z.min():.3f}, {z.max():.3f}]")
    
    # 测试不同的裁剪范围
    test_cases = [
        {
            "name": "默认裁剪范围",
            "x_range": (0, 0.6),
            "y_range": (0, -0.6),
            "z_range": (0.07, 0.08)
        },
        {
            "name": "更严格的裁剪范围",
            "x_range": (0.1, 0.5),
            "y_range": (0, -0.4),
            "z_range": (0.075, 0.085)
        },
        {
            "name": "宽松的裁剪范围",
            "x_range": (0, 0.8),
            "y_range": (0, -0.8),
            "z_range": (0.06, 0.09)
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n{i+2}. 测试 {test_case['name']}:")
        print("-" * 50)
        
        # 裁剪点云
        cropped_pcd, original_count, cropped_count = crop_pointcloud_world(
            pointcloud,
            x_range=test_case['x_range'],
            y_range=test_case['y_range'],
            z_range=test_case['z_range']
        )
        
        # 可视化裁剪后的点云
        if cropped_pcd is not None and len(cropped_pcd) > 0:
            print(f"\n正在显示 {test_case['name']} 的可视化结果...")
            visualize_pointcloud_3d(
                cropped_pcd, 
                title=f"裁剪后点云 - {test_case['name']}",
                height_min=test_case['z_range'][0],
                height_max=test_case['z_range'][1],
                camera_role="测试相机"
            )
            
            # 等待用户确认
            input("按回车键继续下一个测试...")
        else:
            print(f"警告: {test_case['name']} 裁剪后没有剩余点云")
    
    print(f"\n=== 测试完成 ===")
    print("所有裁剪后点云可视化测试已完成")

def demonstrate_keyboard_controls():
    """演示键盘控制功能"""
    print("\n=== 键盘控制说明 ===")
    print("在detect.py运行时，您可以使用以下键盘控制:")
    print("  ESC: 退出程序")
    print("  v: 切换YOLO检测框3D可视化")
    print("  c: 手动触发裁剪后点云可视化")
    print("  h: 显示帮助信息")
    print("\n裁剪后点云可视化特点:")
    print("  - 自动每3秒显示一次裁剪后的点云")
    print("  - 按'c'键可随时手动触发可视化")
    print("  - 显示裁剪范围和统计信息")
    print("  - 根据高度进行颜色映射")

if __name__ == "__main__":
    test_cropped_pointcloud_visualization()
    demonstrate_keyboard_controls()




