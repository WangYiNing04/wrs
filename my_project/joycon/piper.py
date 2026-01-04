'''
Author: wang yining
Date: 2025-11-05 19:26:27
LastEditTime: 2025-11-08 11:14:13
FilePath: /wrs_tiaozhanbei/my_project/joycon/piper.py
Description: 
e-mail: wangyining0408@outlook.com
'''
'''
1.get_pose()
2.delta_pose
3.update_pose
4.ik
5.toppra
6.excute
'''
from joyconrobotics import JoyconRobotics
import time
from wrs.robot_con.piper.piper import PiperArmController
from wrs.robot_sim.manipulators.piper.piper import Piper
import numpy as np
import wrs.basis.robot_math as rm
piper_right = PiperArmController(can_name='can0', has_gripper=True)
piper_right_sim = Piper(enable_cc=True)
joyconrobotics_right = JoyconRobotics("right" , close_y = True)

# Initial pose [x, y, z, roll, pitch, yaw]
init_gpos = [0.210, -0.4, -0.047, -3.1, -1.45, -1.5]

# Pose limits: [[min], [max]]
glimit = [
    [0.210, -0.4, -0.047, -3.1, -1.45, -1.5],
    [0.420, 0.4, 0.30, 3.1, 1.45, 1.5]
]

offset_position_m = init_gpos[:3]

def safe_move_l(piper, piper_con, target_pos, target_rot, speed=10, search_dx=0.01, search_steps=5):
    """
    安全移动函数：
    当末端目标姿态在Z方向抬升时IK无解，允许X方向偏移以寻找可行解
    """
    jnts = piper.ik(target_pos, target_rot)
    if jnts is not None:
        piper_con.move_l(target_pos, target_rot, speed=speed)
        return True
    
    # === IK 无解，沿 X 方向搜索 ===
    print("⚠️ IK 无解，开始X方向微调搜索...")
    for dx in np.linspace(-search_dx, search_dx, search_steps):
        test_pos = target_pos.copy()
        test_pos[0] += dx
        jnts = piper.ik(test_pos, target_rot)
        if jnts is not None:
            print(f"✅ 在 X 偏移 {dx:.3f} m 处找到可行解")
            piper_con.move_l(test_pos, target_rot, speed=speed)
            return True

    print("❌ 搜索失败，无法找到可行IK解")
    return False


try:
    #回归初位
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=10)

    # 1. 获取当前机械臂姿态
    orgin_position, orgin_rot = piper_right.get_pose()
    orgin_jnts_values = piper_right.get_joint_values()
    print("原始姿态:", orgin_position, "\n原始旋转矩阵:\n", orgin_rot)
    
    pose, gripper, control_button = joyconrobotics_right.get_control()
    print(f'{pose=}, {gripper=}, {control_button=}')
    while True:

        # 1. 获取当前机械臂姿态
        # orgin_position, orgin_rot = piper_right.get_pose()
        # orgin_jnts_values = piper_right.get_joint_values()
        # print("原始姿态:", orgin_position, "\n原始旋转矩阵:\n", orgin_rot)

        # 2. 从 JoyCon 获取相对控制输入
        result = joyconrobotics_right.get_control()
        print("JoyCon 返回:", result)

        if result is None:
            raise ValueError("JoyCon 返回 None，可能断开连接或读取失败")

        pose, gripper, control_button = result
        print(f'{pose=}, {gripper=}, {control_button=}')

        # # ---- 添加死区过滤 ----
        # pose = np.array(pose)
        # deadzone_pos = 0.002      # 平移死区 (m)
        # deadzone_rot = np.radians(10)  # 旋转死区 (约2度)

        # # 如果抖动太小，就直接跳过这次循环
        # if np.all(np.abs(pose[:3]) < deadzone_pos) and np.all(np.abs(pose[3:]) < deadzone_rot):
        #     time.sleep(0.05)
        #     continue

        # 3. 计算相对变化
   

        factor = 1  # 调节整体灵敏度的因子

        # === 映射到机械臂运动空间 ===
        delta_pos = np.array(pose[:3])  * factor
        delta_rot = rm.rotmat_from_euler(
            pose[3]  * factor,
            pose[4]  * factor,
            pose[5]  * factor
        )

        print("增量位置:\n", delta_pos)
        print("增量旋转矩阵:\n", delta_rot)
        # 4. 更新姿态
        target_pos = orgin_position + delta_pos
        target_rot = orgin_rot @ delta_rot  

        print("目标位置:", target_pos)
        print("目标旋转:\n", target_rot)
     
        # 6. 执行
        piper_right.move_l(target_pos, target_rot, speed=10)
        #safe_move_l(piper_right_sim, piper_right, target_pos, target_rot, speed=10)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("程序终止")
    #回归初位
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
except Exception as e:
    print("发生错误:", e)
    #回归初位
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
finally:
    #回归初位
    piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)

#断开连接
joyconrobotics_right.disconnnect()