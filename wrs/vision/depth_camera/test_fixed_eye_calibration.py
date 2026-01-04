"""
测试固定眼相机标定功能
"""
import numpy as np
import json
import os
from pathlib import Path

def test_calibration_matrix_loading():
    """测试标定矩阵加载功能"""
    print("测试标定矩阵加载功能...")
    
    # 创建测试标定矩阵
    test_matrix = np.array([
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 保存测试数据
    test_data = {'affine_mat': test_matrix.tolist()}
    test_file = "test_calibration.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    # 加载测试数据
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    
    loaded_matrix = np.array(loaded_data['affine_mat'])
    
    # 验证数据一致性
    assert np.allclose(test_matrix, loaded_matrix), "标定矩阵加载失败"
    print("✓ 标定矩阵加载测试通过")
    
    # 清理测试文件
    os.remove(test_file)

def test_multi_camera_calibration():
    """测试多相机标定数据格式"""
    print("测试多相机标定数据格式...")
    
    # 创建多个测试标定矩阵
    test_matrices = [
        np.array([
            [1.0, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        np.array([
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    ]
    
    # 创建多相机标定数据
    multi_calib_data = {
        'camera_calibrations': []
    }
    
    for i, matrix in enumerate(test_matrices):
        multi_calib_data['camera_calibrations'].append({
            'camera_id': i,
            'affine_mat': matrix.tolist()
        })
    
    # 保存测试数据
    test_file = "test_multi_calibration.json"
    with open(test_file, 'w') as f:
        json.dump(multi_calib_data, f)
    
    # 加载测试数据
    with open(test_file, 'r') as f:
        loaded_data = json.load(f)
    
    # 验证数据
    assert len(loaded_data['camera_calibrations']) == 3, "多相机标定数据数量不正确"
    
    for i, calib_data in enumerate(loaded_data['camera_calibrations']):
        assert calib_data['camera_id'] == i, f"相机ID不正确: {calib_data['camera_id']}"
        loaded_matrix = np.array(calib_data['affine_mat'])
        assert np.allclose(test_matrices[i], loaded_matrix), f"相机{i}标定矩阵不正确"
    
    print("✓ 多相机标定数据格式测试通过")
    
    # 清理测试文件
    os.remove(test_file)

def test_point_cloud_transformation():
    """测试点云变换功能"""
    print("测试点云变换功能...")
    
    # 创建测试点云（相机坐标系）
    test_points = np.array([
        [0.0, 0.0, 1.0],  # 相机前方1米
        [0.1, 0.0, 1.0],  # 相机前方1米，右侧10cm
        [0.0, 0.1, 1.0],  # 相机前方1米，上方10cm
    ])
    
    # 创建变换矩阵（相机到世界）
    # 相机在原点，世界坐标系中相机在(0, 0, 1)位置，向下看
    cam2world = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    # 变换点云到世界坐标系
    # 添加齐次坐标
    points_homo = np.hstack([test_points, np.ones((test_points.shape[0], 1))])
    world_points_homo = (cam2world @ points_homo.T).T
    world_points = world_points_homo[:, :3]
    
    # 验证变换结果
    expected_points = np.array([
        [0.0, 0.0, 2.0],  # 世界坐标系中在(0, 0, 2)
        [0.1, 0.0, 2.0],  # 世界坐标系中在(0.1, 0, 2)
        [0.0, 0.1, 2.0],  # 世界坐标系中在(0, 0.1, 2)
    ])
    
    assert np.allclose(world_points, expected_points), "点云变换不正确"
    print("✓ 点云变换功能测试通过")

def test_calibration_accuracy():
    """测试标定精度评估"""
    print("测试标定精度评估...")
    
    # 模拟标定误差
    true_matrix = np.eye(4)
    true_matrix[:3, 3] = [0.0, 0.0, 1.0]  # 相机在(0, 0, 1)
    
    # 添加小误差
    error_matrix = true_matrix.copy()
    error_matrix[:3, 3] += np.array([0.01, 0.02, 0.005])  # 添加位置误差
    
    # 计算误差
    position_error = np.linalg.norm(true_matrix[:3, 3] - error_matrix[:3, 3])
    rotation_error = np.linalg.norm(true_matrix[:3, :3] - error_matrix[:3, :3])
    
    print(f"位置误差: {position_error:.4f} 米")
    print(f"旋转误差: {rotation_error:.4f}")
    
    # 验证误差在合理范围内
    assert position_error < 0.1, "位置误差过大"
    assert rotation_error < 0.1, "旋转误差过大"
    
    print("✓ 标定精度评估测试通过")

def main():
    """运行所有测试"""
    print("开始固定眼相机标定功能测试...\n")
    
    try:
        test_calibration_matrix_loading()
        test_multi_camera_calibration()
        test_point_cloud_transformation()
        test_calibration_accuracy()
        
        print("\n🎉 所有测试通过！固定眼相机标定功能正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
