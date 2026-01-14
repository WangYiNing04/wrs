#!/usr/bin/env python3
"""
测试平滑运动控制器的脚本
用于验证机械臂移动流畅度优化效果
"""

import time
import numpy as np
from wrs.robot_con.piper.collect_data.DataCollector import DataCollector

def test_smooth_motion():
    """测试平滑运动控制"""
    print("初始化数据采集器...")
    collector = DataCollector()
    
    try:
        print("测试左臂运动控制器...")
        left_controller = collector.left_motion_controller
        
        # 获取当前关节位置
        current_joints = collector.left_arm_con.get_joint_values()
        print(f"当前关节位置: {current_joints}")
        
        # 测试连续移动命令
        print("发送连续移动命令...")
        for i in range(10):
            # 创建小幅度的关节移动
            target_joints = current_joints + np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) * i
            success = left_controller.queue_move_j(target_joints, speed=5)
            print(f"命令 {i+1}: {'成功' if success else '被忽略'}")
            time.sleep(0.1)  # 快速发送命令
        
        # 等待所有命令执行完成
        print("等待命令执行完成...")
        while left_controller.is_busy():
            time.sleep(0.1)
        
        print("测试完成！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    finally:
        # 清理资源
        collector.cleanup()
        print("资源清理完成")

if __name__ == "__main__":
    test_smooth_motion()











