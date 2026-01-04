'''
Author: wang yining
Date: 2025-12-21 17:53:04
LastEditTime: 2025-12-21 18:06:58
FilePath: /wrs_tiaozhanbei/wrs/robot_con/piper/keyboard_control/easy_xyz_control.py
Description: 
e-mail: wangyining0408@outlook.com
'''
import time
import numpy as np
from pynput import keyboard

from wrs.robot_con.piper.piper import PiperArmController


# =========================
# 固定末端朝向（锁死）
# =========================
FIXED_ROT = np.array([
    [ 0.0, 0.0,  1.0],
    [ 0.0, 1.0,  0.0],
    [-1.0, 0.0,  0.0]
])

STEP_SIZE = 0.01   # 每次平移 1 cm
SPEED = 1


def main():
    print("Initializing Piper arm...")
    arm = PiperArmController(can_name="can0", has_gripper=False)
    time.sleep(0.5)

    # 读取当前位姿，作为起点
    tcp_pos, _ = arm.get_pose()
    tcp_pos = np.array(tcp_pos)

    print("Start keyboard control")
    print("W/S: +Z / -Z")
    print("A/D: +Y / -Y")
    print("Q/E: +X / -X")
    print("ESC: exit")

    def on_press(key):
        nonlocal tcp_pos

        delta = np.zeros(3)

        try:
            if key.char == 'w':
                delta[2] += STEP_SIZE
            elif key.char == 's':
                delta[2] -= STEP_SIZE
            elif key.char == 'a':
                delta[1] += STEP_SIZE
            elif key.char == 'd':
                delta[1] -= STEP_SIZE
            elif key.char == 'q':
                delta[0] += STEP_SIZE
            elif key.char == 'e':
                delta[0] -= STEP_SIZE
            else:
                return
        except AttributeError:
            return

        tcp_pos = tcp_pos + delta

        # 姿态始终使用固定 rotmat
        arm.move_p(
            tcp_pos,
            FIXED_ROT,
            speed=SPEED,
            block=False,
            is_euler=False
        )

    def on_release(key):
        if key == keyboard.Key.esc:
            print("Emergency stop & exit")
            return False

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
