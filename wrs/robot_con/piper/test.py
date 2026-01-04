"""
Created on 2025/10/5 
Author: Hao Chen (chen960216@gmail.com)
"""
import time
import numpy as np
from wrs.robot_con.piper.piper import PiperArmController

arm1 = PiperArmController(can_name="can0", has_gripper=True, )
#arm2 = PiperArmController(can_name="can1", has_gripper=True, )

# # 获取当前角度（弧度）
# current_angles = arm1.get_joint_values()  # np.array([0.1, -0.2, ...])
# print(arm1.get_joint_values_raw())
# # 定义目标角度（在当前角度基础上偏移）
# target_angles = current_angles + np.array([0.0, 0.1, -0.1, 0.0, 0.0, 0.0])
#
# # 移动到目标角度（阻塞模式）
# arm1.move_j(target_angles, speed=50, block=True)

print("arm1 joint values:", arm1.get_joint_values_raw(),arm1.get_joint_values())
#print("arm2 joint values:", arm2.get_joint_values_raw())

#arm1.move_j([0, 0, 0, 0, 0, 1.3], block=True,is_radians=True)

# 示例：控制所有关节到特定角度（弧度）
joint_angles_rad = []  # 6个关节的弧度值

# 转换为整数（毫弧度）
factor = 57324.840764
joint_angles_millirad = [int(angle * factor) for angle in joint_angles_rad]

# 调用底层接口
arm1.interface.JointCtrl(
    joint_angles_millirad[0],  # joint1
    joint_angles_millirad[1],  # joint2
    joint_angles_millirad[2],  # joint3
    joint_angles_millirad[3],  # joint4
    joint_angles_millirad[4],  # joint5
    joint_angles_millirad[5]   # joint6
)


arm1.interface.JointCtrl(0, 0, 0, 0, 0, 1300)
#arm2.move_j([0, 0, 0, 0, 0, -1.3], block=True)
time.sleep(1)
arm1.move_j([0, 0, 0, 0, 0, 0], block=False,is_radians=True)
#arm2.move_j([0, 0, 0, 0, 0, 0], block=False)
