'''
Author: wang yining
Date: 2025-11-07 14:33:26
LastEditTime: 2025-11-07 14:40:54
FilePath: /wrs_tiaozhanbei/my_project/joycon/piper3.py
Description: 
e-mail: wangyining0408@outlook.com
'''
from joyconrobotics import JoyconRobotics
import time
from wrs.robot_con.piper.piper import PiperArmController
from wrs.robot_sim.manipulators.piper.piper import Piper
import numpy as np
import wrs.basis.robot_math as rm
piper_right = PiperArmController(can_name='can0', has_gripper=True)
piper_right_sim = Piper(enable_cc=True)
joyconrobotics_right = JoyconRobotics("right")

try:
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
    time.sleep(1)
    while True:
        # 1️⃣ 实时读取机械臂当前状态
        current_position, current_rot = piper_right.get_pose()

        # 2️⃣ 获取 JoyCon 控制输入（增量）
        result = joyconrobotics_right.get_control()
        if result is None:
            raise ValueError("JoyCon 返回 None，可能断开连接或读取失败")

        pose, gripper, control_button = result
        print(f'{pose=}, {gripper=}, {control_button=}')

        # === 添加死区过滤 ===
        pose = np.array(pose)
        deadzone_pos = 0.002        # 平移死区 (m)
        deadzone_rot = np.radians(5)  # 旋转死区 (约5度)

        if np.all(np.abs(pose[:3]) < deadzone_pos) and np.all(np.abs(pose[3:]) < deadzone_rot):
            time.sleep(0.05)
            continue

        # 3️⃣ 计算当前增量（非累积）
        factor = 0.1
        delta_pos = pose[:3]
        delta_rot = rm.rotmat_from_euler(
            pose[3] * factor,
            pose[4] * factor,
            pose[5] * factor
        )

        # 4️⃣ 更新目标姿态（基于当前机械臂状态）
        target_pos = current_position + delta_pos
        target_rot = current_rot @ delta_rot

        print("目标位置:", target_pos)
        print("目标旋转:\n", target_rot)

        # 5️⃣ 执行运动
        piper_right.move_l(target_pos, target_rot, speed=10)
        # 6️⃣ 小延时避免过快循环
        time.sleep(0.05)
except KeyboardInterrupt:
    print("程序终止，关闭机械臂连接")
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
except Exception as e:
    print("发生错误:", e)
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
finally:
    print("机械臂连接已关闭")
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)