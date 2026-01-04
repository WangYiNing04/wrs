#!/usr/bin/env python3
"""
简单的固定眼相机标定测试
"""
import numpy as np
import json

def test_basic_functionality():
    """测试基本功能"""
    print("开始基本功能测试...")
    
    # 测试1: 创建标定矩阵
    calib_matrix = np.eye(4)
    calib_matrix[:3, 3] = [0.0, 0.0, 1.0]  # 相机在(0, 0, 1)位置
    print("✓ 标定矩阵创建成功")
    
    # 测试2: 点云变换
    test_points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    points_homo = np.hstack([test_points, np.ones((test_points.shape[0], 1))])
    world_points_homo = (calib_matrix @ points_homo.T).T
    world_points = world_points_homo[:, :3]
    print("✓ 点云变换成功")
    
    # 测试3: JSON保存和加载
    test_data = {'affine_mat': calib_matrix.tolist()}
    with open('test.json', 'w') as f:
        json.dump(test_data, f)
    
    with open('test.json', 'r') as f:
        loaded_data = json.load(f)
    
    loaded_matrix = np.array(loaded_data['affine_mat'])
    assert np.allclose(calib_matrix, loaded_matrix), "矩阵保存/加载失败"
    print("✓ JSON保存/加载成功")
    
    # 清理
    import os
    os.remove('test.json')
    
    print("🎉 所有基本功能测试通过！")

if __name__ == "__main__":
    test_basic_functionality()



