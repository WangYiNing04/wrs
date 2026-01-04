'''
Author: wang yining
Date: 2025-10-27 16:44:13
LastEditTime: 2025-10-29 06:08:19
FilePath: /wrs_tiaozhanbei/my_project/tiaozhanbei/shoe_task_runner.py
Description: 
e-mail: wangyining0408@outlook.com
'''


import threading
import time
import traceback
from my_project.tiaozhanbei.place_shoe.task_shoe import MultiCameraShoeTask


class ShoeTaskRunner:
    def __init__(self):
        print("🚀 初始化多摄像头鞋子抓取任务中...")
        self.task = MultiCameraShoeTask()
        # 初始化机械臂为零位
        self.task.left_arm.move_j([0] * 6, speed=20)
        self.task.right_arm.move_j([0] * 6, speed=20)
        self.task.left_arm.open_gripper(width=0.04)
        self.task.right_arm.open_gripper(width=0.04)
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
        print("\n🟢 开始执行抓取鞋子任务...\n")
        try:
            start_time = time.time()
            success = self.task.run(show_camera=False)
            end_time = time.time()
            print(f"推理时间{start_time - end_time}")
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


    def start(self):
        """启动命令行控制循环"""
        print("\n=== 控制指令 ===")
        print("  [1] 执行摆放鞋子任务")
        print("  [q] 退出程序")
        print("================\n")

        while not self._stop_flag:
            try:
                cmd = input("请输入指令 [1=抓取, q=退出]: ").strip()
                if cmd == "1":
                    threading.Thread(target=self._run_task_once, daemon=True).start()
                elif cmd.lower() == "q":
                    print("🛑 收到退出指令，程序即将结束...")
                    self._stop_flag = True
                else:
                    print("⚠️ 无效指令，请输入 1 或 q")
            except (KeyboardInterrupt, EOFError):
                # 捕获 Ctrl+C / Ctrl+D
                print("\n🛑 程序被中断，正在退出...")
                self._stop_flag = True

        # 程序退出前回到零位
        print("⚙️ 机械臂回到零位...")
        self.task.left_arm.move_j([0] * 6, speed=20)
        self.task.right_arm.move_j([0] * 6, speed=20)
        print("👋 程序已退出。")



def main():
    runner = ShoeTaskRunner()
    runner.start()


if __name__ == '__main__':
    main()