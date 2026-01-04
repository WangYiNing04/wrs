#!/usr/bin/env python3
"""
测试优化版本的手眼标定程序
"""
import numpy as np
import time
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_acceleration():
    """测试GPU加速功能"""
    print("测试GPU加速功能...")
    
    try:
        import cupy as cp
        print("✓ CuPy可用")
        
        # 测试GPU计算
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([2, 3, 4, 5, 6])
        c = a + b
        result = cp.asnumpy(c)
        print(f"✓ GPU计算测试通过: {result}")
        return True
    except ImportError:
        print("✗ CuPy不可用，将使用CPU处理")
        return False

def test_point_cloud_processing():
    """测试点云处理功能"""
    print("\n测试点云处理功能...")
    
    try:
        from manual_calib_piper_eyeinhand_right_camera_optimized import PointCloudProcessor
        
        # 创建处理器
        processor = PointCloudProcessor(downsample_ratio=0.1, use_gpu=True)
        
        # 生成测试点云
        num_points = 10000
        points = np.random.randn(num_points, 3)
        colors = np.random.rand(num_points, 3)
        
        print(f"原始点云: {len(points)} 点")
        
        # 测试降采样
        start_time = time.time()
        points_ds, colors_ds = processor.downsample_pointcloud(points, colors)
        downsample_time = time.time() - start_time
        
        print(f"降采样后: {len(points_ds)} 点")
        print(f"降采样时间: {downsample_time:.3f}秒")
        
        # 测试变换
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = [0.1, 0.2, 0.3]
        
        start_time = time.time()
        transformed_points = processor.transform_points_gpu(points_ds, transform_matrix)
        transform_time = time.time() - start_time
        
        print(f"变换时间: {transform_time:.3f}秒")
        print("✓ 点云处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 点云处理测试失败: {e}")
        return False

def test_async_capture():
    """测试异步捕获功能"""
    print("\n测试异步捕获功能...")
    
    try:
        from manual_calib_piper_eyeinhand_right_camera_optimized import AsyncPointCloudCapture
        
        # 创建模拟传感器
        class MockSensor:
            def get_pcd_texture_depth(self):
                points = np.random.randn(1000, 3)
                colors = np.random.rand(1000, 3)
                return points, colors, None, None
        
        # 创建异步捕获器
        sensor = MockSensor()
        capture = AsyncPointCloudCapture(sensor, max_queue_size=3)
        
        # 启动捕获
        capture.start()
        time.sleep(0.1)  # 等待一些数据
        
        # 获取最新数据
        latest_data = capture.get_latest_pcd()
        if latest_data is not None:
            print("✓ 异步捕获测试通过")
            capture.stop()
            return True
        else:
            print("✗ 异步捕获测试失败")
            capture.stop()
            return False
            
    except Exception as e:
        print(f"✗ 异步捕获测试失败: {e}")
        return False

def test_performance():
    """测试性能提升"""
    print("\n测试性能提升...")
    
    try:
        from manual_calib_piper_eyeinhand_right_camera_optimized import PointCloudProcessor
        
        processor = PointCloudProcessor(downsample_ratio=0.1, use_gpu=True)
        
        # 生成大量点云数据
        num_points = 50000
        points = np.random.randn(num_points, 3)
        colors = np.random.rand(num_points, 3)
        transform_matrix = np.eye(4)
        
        # 测试处理时间
        start_time = time.time()
        processed_points, processed_colors = processor.process_pointcloud(
            points, colors, transform_matrix
        )
        processing_time = time.time() - start_time
        
        print(f"处理 {num_points} 个点 -> {len(processed_points)} 个点")
        print(f"处理时间: {processing_time:.3f}秒")
        print(f"处理速度: {num_points/processing_time:.0f} 点/秒")
        
        # 检查性能是否合理
        if processing_time < 1.0:  # 应该在1秒内完成
            print("✓ 性能测试通过")
            return True
        else:
            print("✗ 性能测试失败（处理时间过长）")
            return False
            
    except Exception as e:
        print(f"✗ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("优化版本手眼标定程序测试")
    print("=" * 50)
    
    tests = [
        ("GPU加速", test_gpu_acceleration),
        ("点云处理", test_point_cloud_processing),
        ("异步捕获", test_async_capture),
        ("性能测试", test_performance),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}测试:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！优化版本可以正常使用。")
        print("\n使用说明:")
        print("1. 运行优化版本: python manual_calib_piper_eyeinhand_right_camera_optimized.py")
        print("2. 调整参数以获得最佳性能")
        print("3. 如果遇到问题，请检查GPU驱动和依赖库")
    else:
        print(f"\n⚠️  有 {total-passed} 个测试失败，请检查环境配置")
        print("\n故障排除:")
        print("1. 安装必要的依赖库")
        print("2. 检查GPU驱动和CUDA")
        print("3. 确保Python环境正确")

if __name__ == "__main__":
    main()



