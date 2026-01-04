'''
Author: wang yining
Date: 2025-10-27 16:44:13
LastEditTime: 2025-10-29 02:33:09
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/cup_task_runner.py
Description: 
e-mail: wangyining0408@outlook.com
'''


import threading
import time
import traceback
from pynput import keyboard
from my_project.tiaozhanbei.empty_cup_place.task_cup import MultiCameraCupTask


class CupTaskRunner:
    def __init__(self):
        print("🚀 初始化多摄像头杯子抓取任务中...")
        self.task = MultiCameraCupTask()
        # 初始化机械臂为零位
        self.task.left_arm.move_j([0] * 6, speed=20)
        self.task.right_arm.move_j([0] * 6, speed=20)
        self.task.left_arm.open_gripper(width=0.03)
        self.task.right_arm.open_gripper(width=0.03)
        print("✅ 初始化完成，等待指令...")

        # 任务状态
        self.is_running = False
        self.listener_thread = None
        self._stop_flag = False

    def _run_task_once(self):
        """执行一次抓取任务"""
        if self.is_running:
            print("⚠️ 任务仍在运行中，请稍候...")
            return

        self.is_running = True
        print("\n🟢 开始执行抓取杯子任务...\n")
        try:
            success = self.task.run(show_camera=False)
            print("✅ 抓取任务完成" if success else "❌ 抓取任务失败")
        except Exception as e:
            traceback.print_exc()
            print("⚠️ 执行过程中出现异常！")
        finally:
            self.is_running = False
            # 可在此回到初始位置
            self.task.left_arm.move_j([0] * 6, speed=20)
            self.task.right_arm.move_j([0] * 6, speed=20)
            print("\n🟡 等待下一次按键指令...")

    def on_press(self, key):
        """键盘监听回调"""
        try:
            if key.char == '1':
                # 执行抓取任务
                threading.Thread(target=self._run_task_once, daemon=True).start()
            elif key.char.lower() == 'q':
                print("🛑 收到退出指令，程序即将结束...")
                self._stop_flag = True
                return False  # 停止监听器
        except AttributeError:
            pass

    def start(self):
        """启动主循环监听"""
        print("\n=== 控制指令 ===")
        print("  [1] 执行抓取杯子任务")
        print("  [q] 退出程序")
        print("================\n")

        # 开启键盘监听
        with keyboard.Listener(on_press=self.on_press) as listener:
            while not self._stop_flag:
                time.sleep(0.2)

        print("⚙️ 机械臂回到零位...")
        self.task.left_arm.move_j([0] * 6, speed=20)
        self.task.right_arm.move_j([0] * 6, speed=20)
        print("👋 程序已退出。")


def main():
    runner = CupTaskRunner()
    runner.start()


if __name__ == '__main__':
    main()