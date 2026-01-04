'''
Author: wang yining
Date: 2025-11-05 19:26:27
LastEditTime: 2025-11-12 15:19:52
FilePath: /wrs_tiaozhanbei/my_project/joycon/piper4.py
Description: 控制机械臂并实时可视化轨迹（可选可视化版本）
e-mail: wangyining0408@outlook.com
'''
import sys, os
sys.path.append(os.path.expanduser("~/PycharmProjects/wrs_tiaozhanbei"))
sys.path.append(os.path.expanduser("~/joycon-robotics"))

import time
import threading
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from joyconrobotics import JoyconRobotics
from wrs.robot_con.piper.piper import PiperArmController
from wrs.robot_sim.manipulators.piper.piper import Piper
import wrs.basis.robot_math as rm

'''
操作指南(右手为例):

    ZR 切换控制状态(移动和不移动)allow_move
    R 开关夹爪(切换)
    X 上升
    B 下降
    摇杆:
        向前 末端执行器前方移动(piper的X轴正方向) 
        向后 反之
        向左 控制Joint1顺时针旋转
        向右 反之

    plus 启用 yaw 控制(第六关节)
    Y 开始记录轨迹
    A 结束记录轨迹
'''
class PiperControllerWithVisualization:
    def __init__(self):
        # 初始化机械臂控制器
        self.piper_right = PiperArmController(can_name='can0', has_gripper=True)
        self.piper_right_sim = Piper(enable_cc=True)

        self.glimit = [[-2.618, 0.0, -2.697, -1.832, -1.22, -2.094],
                       [2.618, 3.14, 0.0, 1.832, 1.22, 2.094]]

        # 初始化 JoyCon
        self.joyconrobotics_right = JoyconRobotics("right", close_y=True, glimit=self.glimit, lock_roll=False, horizontal_stick_mode="piper", pure_xz= True)

        # 初始姿态
        self.init_gpos = [0.210, -0.4, -0.047, -3.1, -1.45, -1.5]

        # 可视化参数
        self.visualize = False
        self.fig = None
        self.ax = None
        self.trajectory = []
        self.max_trajectory_points = 200

        # 可视化线程控制
        self.vis_thread = None
        self.vis_running = False

        # 主控制标志
        self.running = True
        self.lock = threading.Lock()

    # ========== 可视化部分 ==========
    def setup_visualization(self):
        """初始化可视化界面"""
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([0.1, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([-0.1, 0.4])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title("Piper Arm Real-time Trajectory")

    def visualization_thread(self):
        """独立线程实时刷新可视化"""
        self.setup_visualization()
        self.vis_running = True
        while self.vis_running:
            with self.lock:
                if len(self.trajectory) == 0:
                    time.sleep(0.05)
                    continue
                traj = np.array(self.trajectory)
                position = traj[-1]

            # 绘图
            self.ax.cla()
            self.ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
            self.ax.scatter(position[0], position[1], position[2], c='r', s=100, marker='o', label='Current Position')
            self.ax.set_xlim([0.1, 0.5])
            self.ax.set_ylim([-0.5, 0.5])
            self.ax.set_zlim([-0.1, 0.4])
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.legend()
            self.ax.set_title("Piper Arm Real-time Trajectory")
            plt.draw()
            plt.pause(0.05)

        plt.ioff()
        plt.close(self.fig)

    def start_visualization(self):
        """启动可视化线程"""
        if not self.visualize:
            print("ℹ️ 可视化关闭状态，未启动。")
            return
        if self.vis_thread is None or not self.vis_thread.is_alive():
            print("🟢 启动实时可视化线程...")
            self.vis_thread = threading.Thread(target=self.visualization_thread, daemon=True)
            self.vis_thread.start()

    def stop_visualization(self):
        """停止可视化线程"""
        if self.vis_running:
            print("🛑 停止可视化线程...")
            self.vis_running = False
            if self.vis_thread is not None:
                self.vis_thread.join(timeout=1)

    # ========== 机械臂控制部分 ==========
    def safe_move_l(self, target_pos, target_rot, speed=10):
        """安全移动函数，先检查IK解"""
        jnts = self.piper_right_sim.ik(target_pos, target_rot)
        if jnts is not None:
            self.piper_right.move_l(target_pos, target_rot, speed=speed)
            print(f"move_l输入位置: {target_pos}, 旋转矩阵: {target_rot}")
            return True
        #print("❌ 无可行IK解，跳过移动")
        return False

    def control_loop(self):
        """主控制循环"""
        try:
            #回归初位
            self.piper_right.move_j([0, 0, -0.1, 0, 0, 0], speed=10)
            self.piper_right.close_gripper()
            
            origin_position, origin_rot = self.piper_right.get_pose()
            print("原始姿态:", origin_position)

            pose, gripper, control_button = self.joyconrobotics_right.get_control()
            print(f'初始状态: pose={pose}, gripper={gripper}, control_button={control_button}')
            
            prev_allow_move = 0
            prev_r_button = 1  # 添加R按钮状态记录
            prev_allow_rot = 0
            base_pose = None
            base_pos = np.array(origin_position)
            base_rot = origin_rot.copy()
            base_joints = None

            rot_enable = False
            
            rel_pose = None
            pre_rel_pose = None

            # 添加移动状态标志
            move_enabled = False
            # 夹爪状态：True为打开，False为关闭
            gripper_open = True

            if self.visualize:
                # 启动可视化线程（如果开启）
                self.start_visualization()

            while self.running:
                result = self.joyconrobotics_right.get_control()
                if result is None:
                    raise ValueError("JoyCon 返回 None,可能断开连接或读取失败")

                pose, gripper, control_button = result
                print(f'此时总状态: pose={pose}, gripper={gripper}, control_button={control_button}')

                if self.joyconrobotics_right.joycon.get_button_home():
                    print("🏠 HOME 按钮按下，回到初始位置")
                    self.piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
                    time.sleep(1)
                    continue
                
                allow_move = int(self.joyconrobotics_right.joycon.get_button_zr())
                r_button = int(self.joyconrobotics_right.joycon.get_button_r())  # 获取R按钮状态
            
                # 检测 ZR 上升沿，实现单击切换
                if prev_allow_move == 0 and allow_move == 1:
                    move_enabled = not move_enabled  # 切换移动状态
                    if move_enabled:
                        base_pose = np.array(pose, dtype=float)
                        base_pos, base_rot = self.piper_right.get_pose()  # 机械臂当前位置
                        base_joints = self.piper_right.get_joint_values()
                        print("📍 ZR 单击：开启移动，记录当前基准")
                    else:
                        base_pose = None
                        print("❌ ZR 单击：关闭移动")

                # 检测 R 按钮上升沿，实现夹爪切换
                if prev_r_button == 1 and r_button == 0:
                    gripper_open = not gripper_open  # 切换夹爪状态
                    if gripper_open:
                        self.piper_right.open_gripper()
                        print("🟢 R 按钮：打开夹爪")
                    else:
                        self.piper_right.close_gripper()
                        print("🔴 R 按钮：关闭夹爪")

                if move_enabled:
                    joycon_stick_h = self.joyconrobotics_right.joycon.get_stick_right_horizontal()
                    if joycon_stick_h > 3300:
                        jnts_values_now = self.piper_right.get_joint_values()
                        jnts_values_now[0] -= 0.05  # Joint1 顺时针
                        self.piper_right.move_j(jnts_values_now, speed=5)
                        base_pos, base_rot = self.piper_right.get_pose() #机械臂当前位置
                    elif joycon_stick_h < 1200:
                        jnts_values_now = self.piper_right.get_joint_values()
                        jnts_values_now[0] += 0.05  # Joint1 逆时针
                        self.piper_right.move_j(jnts_values_now, speed=5)
                        base_pos, _ = self.piper_right.get_pose() #机械臂当前位置

                success = False
                
     
                allow_rot = prev_allow_rot
                move_l_enabled = False

                # 根据移动状态进行移动
                if move_enabled and base_pose is not None:
                    if rel_pose is None:
                        rel_pose = np.array(pose, dtype=float) - base_pose
                    if pre_rel_pose is None:
                        pre_rel_pose = rel_pose.copy()

                    base_rot = self.piper_right.get_pose()[1]  # 实时更新基准旋转矩阵
                    rel_pose = np.array(pose, dtype=float) - base_pose
                    if not np.allclose(pre_rel_pose[:3], rel_pose[:3], atol=1e-4):
                        move_l_enabled = True
                    else:
                        move_l_enabled = False
                    pre_rel_pose = rel_pose.copy()
                    rel_delta_pos_local = np.array([-rel_pose[2], 0.0, rel_pose[0]])
                    print(rel_delta_pos_local)
                    rel_delta_pos_world = base_rot @ rel_delta_pos_local
                    print(base_rot)
                    print(f'此时增量状态(末端坐标系XZ): {rel_delta_pos_local}')
                    print(rel_delta_pos_world)
                    target_pos = base_pos + rel_delta_pos_world
                    target_rot = base_rot
                    print(target_pos)

                    allow_rot = int(self.joyconrobotics_right.joycon.get_button_plus())

                    if prev_allow_rot == 0 and allow_rot == 1:
                        rot_enable = not rot_enable  # 切换移动状态
                        base_pose = np.array(pose, dtype=float)
                        base_pos, base_rot = self.piper_right.get_pose()  # 机械臂当前位置
                        base_joints = self.piper_right.get_joint_values()

                    if not rot_enable:
                        if move_l_enabled:
                            success = self.safe_move_l(target_pos, target_rot, speed=20)
                        else:
                            success = False
                    
                    if rot_enable:
                        target_joints = self.piper_right.get_joint_values()
                        factor = 0.5
                        target_joints[4] = np.clip(base_joints[4] + rel_pose[4] * factor, self.glimit[0][4], self.glimit[1][4])
                        target_joints[5] = np.clip(base_joints[5] + rel_pose[3] * factor, self.glimit[0][5], self.glimit[1][5])
                        self.piper_right.move_j(target_joints, speed=20)

                if success:
                    current_position, _ = self.piper_right.get_pose()
                    with self.lock:
                        self.trajectory.append(current_position.copy())
                        if len(self.trajectory) > self.max_trajectory_points:
                            self.trajectory.pop(0)

                prev_allow_move = allow_move
                prev_r_button = r_button  # 更新R按钮状态
                prev_allow_rot = allow_rot

                if control_button == 8:  # 退出
                    print("收到重置信号，停止程序")
                    break

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("程序被用户中断")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """资源清理"""
        self.running = False
        self.stop_visualization()
        try:
            self.piper_right.move_j([0, 0, 0, 0, 0, 0], speed=20)
        except:
            pass
        try:
            self.joyconrobotics_right.disconnnect()
        except:
            pass
        print("✅ 程序结束，资源已清理")


# ========== 主函数 ==========
def main():
    controller = PiperControllerWithVisualization()
    controller.visualize = False  # 可选：初始就启用可视化
    controller.control_loop()


if __name__ == "__main__":
    main()
