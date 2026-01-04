'''
Author: wang yining
Date: 2025-10-27 16:44:13
LastEditTime: 2025-10-27 20:57:57
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/task_runner.py
Description: 一键运行 Piper 多任务控制器
e-mail: wangyining0408@outlook.com
'''

import threading
import time
import traceback
from pynput import keyboard

# 导入两个任务
from my_project.tiaozhanbei.empty_cup_place.task_cup import *
from my_project.tiaozhanbei.stack_bowls_three.task_bowl import *
from my_project.tiaozhanbei.place_shoe.task_shoe import *

class ResourceManager:
    def __init__(self):
        print("🧩 初始化共享资源...")
        self.left_arm = PiperArmController(can_name='can0', has_gripper=True)
        self.right_arm = PiperArmController(can_name='can1', has_gripper=True)
        self.cameras = {
            "middle": {"cam": init_camera(camera_id='middle'), "type": "fixed", "c2w": MIDDLE_CAM_C2W},
        }
        self.yolo_bowl = init_yolo(YOLO_MODEL_BOWLS_PATH)
        self.yolo_cup = init_yolo(YOLO_MODEL_CUPS_PATH)


class TaskRunner:
    def __init__(self):
        print("🚀 初始化 Piper 多任务控制器中...")

        resources = ResourceManager()
        self.cup_task = MultiCameraCupTask(resources)
        self.bowl_task = MultiCameraBowlTask(resources)

        # 初始化机械臂归零
        self.cup_task.left_arm.move_j([0] * 6, speed=20)
        self.cup_task.right_arm.move_j([0] * 6, speed=20)
        print("✅ 初始化完成，等待键盘指令...")

        # 控制状态
        self.is_running = False
        self._stop_flag = False

    # -----------------------------------
    # 执行具体任务（内部函数）
    # -----------------------------------
    def _run_task(self, task_name):
        if self.is_running:
            print("⚠️ 有任务正在执行，请稍候...")
            return

        self.is_running = True

        try:
            if task_name == "cup":
                print("\n🟢 开始执行【杯子抓取任务】...\n")
                success = self.cup_task.run(show_camera=False)
                print("✅ 杯子任务完成" if success else "❌ 杯子任务失败")

            elif task_name == "bowl":
                print("\n🟣 开始执行【碗叠放任务】...\n")
                success = self.bowl_task.run(show_camera=False)
                print("✅ 碗任务完成" if success else "❌ 碗任务失败")

        except Exception as e:
            traceback.print_exc()
            print("⚠️ 执行过程中出现异常！")
        finally:
            # 回零
            self.cup_task.left_arm.move_j([0] * 6, speed=20)
            self.cup_task.right_arm.move_j([0] * 6, speed=20)
            self.is_running = False
            print("\n🟡 任务结束，等待下一次按键...")

    # -----------------------------------
    # 键盘监听
    # -----------------------------------
    def on_press(self, key):
        try:
            if key.char == '1':
                threading.Thread(target=self._run_task, args=("cup",), daemon=True).start()
            elif key.char == '2':
                threading.Thread(target=self._run_task, args=("bowl",), daemon=True).start()
            elif key.char.lower() == 'q':
                print("🛑 收到退出指令，程序即将结束...")
                self._stop_flag = True
                return False
        except AttributeError:
            pass

    # -----------------------------------
    # 主循环
    # -----------------------------------
    def start(self):
        print("\n=== 控制指令 ===")
        print("  [1] 执行杯子抓取任务")
        print("  [2] 执行碗叠放任务")
        print("  [q] 退出程序")
        print("================\n")

        with keyboard.Listener(on_press=self.on_press) as listener:
            while not self._stop_flag:
                time.sleep(0.2)

        print("⚙️ 机械臂回到零位...")
        self.cup_task.left_arm.move_j([0] * 6, speed=20)
        self.cup_task.right_arm.move_j([0] * 6, speed=20)
        print("👋 程序已退出。")


def main():
    runner = TaskRunner()
    runner.start()


if __name__ == '__main__':
    main()
