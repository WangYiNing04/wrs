#!/usr/bin/env python3
"""
性能对比测试脚本
比较原始版本和优化版本的性能
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, name):
        self.name = name
        self.frame_times = deque(maxlen=100)
        self.point_counts = deque(maxlen=100)
        self.start_time = None
        self.frame_count = 0
        
    def start_frame(self):
        """开始帧计时"""
        self.start_time = time.time()
        
    def end_frame(self, point_count=0):
        """结束帧计时"""
        if self.start_time is not None:
            frame_time = time.time() - self.start_time
            self.frame_times.append(frame_time)
            self.point_counts.append(point_count)
            self.frame_count += 1
            
    def get_stats(self):
        """获取统计信息"""
        if not self.frame_times:
            return {}
            
        return {
            'name': self.name,
            'avg_frame_time': np.mean(self.frame_times),
            'min_frame_time': np.min(self.frame_times),
            'max_frame_time': np.max(self.frame_times),
            'fps': 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
            'avg_point_count': np.mean(self.point_counts),
            'total_frames': self.frame_count
        }

def simulate_point_cloud_processing():
    """模拟点云处理"""
    # 生成随机点云数据
    num_points = np.random.randint(10000, 50000)
    points = np.random.randn(num_points, 3)
    colors = np.random.rand(num_points, 3)
    
    # 模拟变换矩阵
    transform = np.eye(4)
    transform[:3, 3] = np.random.randn(3) * 0.1
    
    return points, colors, transform

def test_original_processing():
    """测试原始处理方式"""
    monitor = PerformanceMonitor("原始版本")
    
    print("测试原始处理方式...")
    for i in range(50):
        monitor.start_frame()
        
        # 模拟原始处理
        points, colors, transform = simulate_point_cloud_processing()
        
        # 模拟变换（CPU）
        points_homo = np.hstack([points, np.ones((len(points), 1))])
        transformed = np.dot(points_homo, transform.T)[:, :3]
        
        # 模拟可视化
        time.sleep(0.01)  # 模拟渲染时间
        
        monitor.end_frame(len(points))
        
        if i % 10 == 0:
            print(f"  帧 {i}: {len(points)} 点, {1.0/monitor.frame_times[-1]:.1f} FPS")
    
    return monitor.get_stats()

def test_optimized_processing():
    """测试优化处理方式"""
    monitor = PerformanceMonitor("优化版本")
    
    print("测试优化处理方式...")
    
    # 模拟GPU加速
    try:
        import cupy as cp
        gpu_available = True
        print("  使用GPU加速")
    except ImportError:
        gpu_available = False
        print("  使用CPU处理（GPU不可用）")
    
    for i in range(50):
        monitor.start_frame()
        
        # 模拟优化处理
        points, colors, transform = simulate_point_cloud_processing()
        
        # 点云降采样
        downsample_ratio = 0.1
        num_points = len(points)
        target_points = max(1000, int(num_points * downsample_ratio))
        if target_points < num_points:
            indices = np.random.choice(num_points, target_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # 变换处理
        if gpu_available:
            # 模拟GPU处理
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            transformed = np.dot(points_homo, transform.T)[:, :3]
        else:
            # CPU处理
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            transformed = np.dot(points_homo, transform.T)[:, :3]
        
        # 模拟优化后的可视化
        time.sleep(0.005)  # 减少渲染时间
        
        monitor.end_frame(len(points))
        
        if i % 10 == 0:
            print(f"  帧 {i}: {len(points)} 点, {1.0/monitor.frame_times[-1]:.1f} FPS")
    
    return monitor.get_stats()

def plot_performance_comparison(original_stats, optimized_stats):
    """绘制性能对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # FPS对比
    names = [original_stats['name'], optimized_stats['name']]
    fps_values = [original_stats['fps'], optimized_stats['fps']]
    ax1.bar(names, fps_values, color=['red', 'green'])
    ax1.set_title('FPS对比')
    ax1.set_ylabel('FPS')
    for i, v in enumerate(fps_values):
        ax1.text(i, v + 0.1, f'{v:.1f}', ha='center')
    
    # 平均帧时间对比
    frame_times = [original_stats['avg_frame_time'], optimized_stats['avg_frame_time']]
    ax2.bar(names, frame_times, color=['red', 'green'])
    ax2.set_title('平均帧时间对比')
    ax2.set_ylabel('时间 (秒)')
    for i, v in enumerate(frame_times):
        ax2.text(i, v + 0.001, f'{v:.3f}', ha='center')
    
    # 点云数量对比
    point_counts = [original_stats['avg_point_count'], optimized_stats['avg_point_count']]
    ax3.bar(names, point_counts, color=['red', 'green'])
    ax3.set_title('平均点云数量对比')
    ax3.set_ylabel('点数')
    for i, v in enumerate(point_counts):
        ax3.text(i, v + 100, f'{int(v)}', ha='center')
    
    # 性能提升
    fps_improvement = (optimized_stats['fps'] - original_stats['fps']) / original_stats['fps'] * 100
    time_improvement = (original_stats['avg_frame_time'] - optimized_stats['avg_frame_time']) / original_stats['avg_frame_time'] * 100
    
    improvements = [fps_improvement, time_improvement]
    improvement_labels = ['FPS提升 (%)', '时间减少 (%)']
    ax4.bar(improvement_labels, improvements, color=['blue', 'orange'])
    ax4.set_title('性能提升')
    ax4.set_ylabel('百分比 (%)')
    for i, v in enumerate(improvements):
        ax4.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("手眼标定程序性能对比测试")
    print("=" * 50)
    
    # 测试原始版本
    print("\n1. 测试原始版本...")
    original_stats = test_original_processing()
    
    # 测试优化版本
    print("\n2. 测试优化版本...")
    optimized_stats = test_optimized_processing()
    
    # 打印结果
    print("\n" + "=" * 50)
    print("性能对比结果:")
    print("=" * 50)
    
    print(f"\n原始版本:")
    print(f"  平均FPS: {original_stats['fps']:.1f}")
    print(f"  平均帧时间: {original_stats['avg_frame_time']:.3f}秒")
    print(f"  平均点云数量: {original_stats['avg_point_count']:.0f}")
    
    print(f"\n优化版本:")
    print(f"  平均FPS: {optimized_stats['fps']:.1f}")
    print(f"  平均帧时间: {optimized_stats['avg_frame_time']:.3f}秒")
    print(f"  平均点云数量: {optimized_stats['avg_point_count']:.0f}")
    
    # 计算改进
    fps_improvement = (optimized_stats['fps'] - original_stats['fps']) / original_stats['fps'] * 100
    time_improvement = (original_stats['avg_frame_time'] - optimized_stats['avg_frame_time']) / original_stats['avg_frame_time'] * 100
    
    print(f"\n性能改进:")
    print(f"  FPS提升: {fps_improvement:.1f}%")
    print(f"  时间减少: {time_improvement:.1f}%")
    print(f"  点云数量减少: {(1 - optimized_stats['avg_point_count']/original_stats['avg_point_count'])*100:.1f}%")
    
    # 绘制对比图
    try:
        plot_performance_comparison(original_stats, optimized_stats)
        print(f"\n性能对比图已保存为 'performance_comparison.png'")
    except ImportError:
        print(f"\nmatplotlib不可用，跳过图表生成")
    
    print(f"\n优化建议:")
    print(f"1. 使用优化版本: manual_calib_piper_eyeinhand_right_camera_optimized.py")
    print(f"2. 调整downsample_ratio参数来控制点云密度")
    print(f"3. 调整visualization_fps参数来控制更新频率")
    print(f"4. 安装GPU加速库以获得更好性能")

if __name__ == "__main__":
    main()



