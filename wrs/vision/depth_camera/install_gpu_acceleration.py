#!/usr/bin/env python3
"""
安装GPU加速库的脚本
"""
import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装 {package} 失败: {e}")
        return False

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用，版本: {torch.version.cuda}")
            print(f"✓ GPU设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("✗ CUDA不可用")
            return False
    except ImportError:
        print("✗ PyTorch未安装")
        return False

def main():
    """主安装函数"""
    print("开始安装GPU加速库...")
    
    # 检查CUDA
    print("\n1. 检查CUDA环境...")
    cuda_available = check_cuda()
    
    # 安装基础包
    print("\n2. 安装基础包...")
    basic_packages = [
        "numpy",
        "opencv-python",
        "open3d",
    ]
    
    for package in basic_packages:
        install_package(package)
    
    # 安装GPU加速包
    if cuda_available:
        print("\n3. 安装GPU加速包...")
        gpu_packages = [
            "cupy-cuda12x",  # 根据CUDA版本调整
            "torch",
            "torchvision",
        ]
        
        for package in gpu_packages:
            install_package(package)
    else:
        print("\n3. 跳过GPU包安装（CUDA不可用）")
    
    # 验证安装
    print("\n4. 验证安装...")
    
    # 测试基础包
    try:
        import numpy as np
        print("✓ NumPy可用")
    except ImportError:
        print("✗ NumPy不可用")
    
    try:
        import cv2
        print("✓ OpenCV可用")
    except ImportError:
        print("✗ OpenCV不可用")
    
    try:
        import open3d as o3d
        print("✓ Open3D可用")
    except ImportError:
        print("✗ Open3D不可用")
    
    # 测试GPU包
    if cuda_available:
        try:
            import cupy as cp
            print("✓ CuPy可用")
        except ImportError:
            print("✗ CuPy不可用")
        
        try:
            import torch
            print("✓ PyTorch可用")
        except ImportError:
            print("✗ PyTorch不可用")
    
    print("\n安装完成！")
    print("\n使用说明：")
    print("1. 运行优化版本: python manual_calib_piper_eyeinhand_right_camera_optimized.py")
    print("2. 如果GPU不可用，程序会自动回退到CPU处理")
    print("3. 可以通过调整downsample_ratio和visualization_fps参数来进一步优化性能")

if __name__ == "__main__":
    main()



