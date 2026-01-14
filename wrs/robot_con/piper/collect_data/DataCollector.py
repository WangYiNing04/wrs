import os
import cv2
import time
import yaml
import threading
import queue
from fontTools.unicodedata import block
from pynput import keyboard
from pynput import mouse
from fontTools.unicodedata import block
from pynput.mouse import Listener as MouseListener
from pynput.mouse import Button

import numpy as np
from pathlib import Path
<<<<<<< HEAD
#from trac_ik import TracIK
=======
from trac_ik import TracIK
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
from wrs.robot_sim.manipulators.piper.piper import Piper
from wrs.robot_con.piper.piper import PiperArmController
import wrs.basis.robot_math as rm
from wrs.drivers.devices.realsense.realsense_d400s import *

'''
[1, 0, 0, -0.002;
 0, 0, 1, -0.008;
 0,-1, 0,    0;
 0, 0, 0,    1   ]
'''
def numpy_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    return data


class SmoothMotionController:
    """平滑运动控制器，用于优化机械臂移动流畅度"""
    
    def __init__(self, arm_controller, arm_sim=None):
        self.arm_controller = arm_controller
        self.arm_sim = arm_sim
<<<<<<< HEAD
        self.command_queue = queue.Queue(maxsize=2)  # 限制队列大小
=======
        self.command_queue = queue.Queue(maxsize=10)  # 限制队列大小
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
        self.is_moving = False
        self.current_target = None
        self.motion_thread = None
        self.stop_motion = False
        self.last_command_time = 0
        self.command_interval = 0.05  # 最小命令间隔50ms
<<<<<<< HEAD
        self.motion_complete = threading.Event()
        self.motion_complete.set()  # 初始状态为完成
=======
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
        
        # 启动运动控制线程
        self.start_motion_thread()
    
    def start_motion_thread(self):
        """启动运动控制线程"""
        if self.motion_thread is None or not self.motion_thread.is_alive():
            self.stop_motion = False
            self.motion_thread = threading.Thread(target=self._motion_worker, daemon=True)
            self.motion_thread.start()
    
    def _motion_worker(self):
        """运动控制工作线程"""
        while not self.stop_motion:
            try:
<<<<<<< HEAD

                # 等待运动完成
                self.motion_complete.wait()

                # 从队列获取命令，超时0.1秒
                command = self.command_queue.get(timeout=0.1)
                

                # 设置运动未完成标志
                self.motion_complete.clear()

=======
                # 从队列获取命令，超时0.1秒
                command = self.command_queue.get(timeout=0.1)
                
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
                if command['type'] == 'move_j':
                    self._execute_move_j(command)
                elif command['type'] == 'move_p':
                    self._execute_move_p(command)
                elif command['type'] == 'move_l':
                    self._execute_move_l(command)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Motion worker error: {e}")
                continue
    
    def _execute_move_j(self, command):
        """执行关节空间移动"""
        try:
            self.is_moving = True
            self.arm_controller.move_j(
                command['joint_angles'],
                speed=command.get('speed', 10),
                block=False,  # 非阻塞
                is_radians=command.get('is_radians', True)
            )
<<<<<<< HEAD

            self.current_target = command['joint_angles']
            self.last_joint_positions = None  # 重置上一次关节位置

            # 启动监控线程
            threading.Thread(target=self._monitor_joint_movement, args=(command['joint_angles'],), daemon=True).start()
        except Exception as e:
            print(f"Move J execution error: {e}")
            self.motion_complete.set()  # 出错时标记为完成
=======
            # 短暂等待确保命令发送
            time.sleep(0.01)
        except Exception as e:
            print(f"Move J execution error: {e}")
        finally:
            self.is_moving = False
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    
    def _execute_move_p(self, command):
        """执行位置控制移动"""
        try:
            self.is_moving = True
            self.arm_controller.move_p(
                command['pos'],
                command['rot'],
                speed=command.get('speed', 10),
                block=False,  # 非阻塞
                is_euler=command.get('is_euler', False)
            )
<<<<<<< HEAD
            self.current_target = (command['pos'], command['rot'])
            self.last_joint_positions = None  # 重置上一次关节位置
            # 启动监控线程
            threading.Thread(target=self._monitor_joint_movement, args=(command['joint_angles'],), daemon=True).start()
        except Exception as e:
            print(f"Move P execution error: {e}")
            self.motion_complete.set()  # 出错时标记为完成
=======
            time.sleep(0.01)
        except Exception as e:
            print(f"Move P execution error: {e}")
        finally:
            self.is_moving = False
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    
    def _execute_move_l(self, command):
        """执行线性移动"""
        try:
            self.is_moving = True
            self.arm_controller.move_l(
                command['pos'],
                command['rot'],
                speed=command.get('speed', 10),
                block=False,  # 非阻塞
                is_euler=command.get('is_euler', False)
            )
<<<<<<< HEAD
            self.current_target = (command['pos'], command['rot'])
            self.last_joint_positions = None  # 重置上一次关节位置
            # 启动监控线程
            threading.Thread(target=self._monitor_joint_movement, args=(command['joint_angles'],) ,daemon=True).start()
        except Exception as e:
            print(f"Move L execution error: {e}")
            self.motion_complete.set()  # 出错时标记为完成

    def _monitor_joint_movement(self, threshold=0.001):
        """监控关节运动是否完成"""
        while not self.stop_motion and not self.motion_complete.is_set():
            current_positions = self.arm_controller.get_joint_values()
            
            if self.last_joint_positions is not None:
                # 计算两次读取之间的最大变化量
                max_diff = max(abs(c - p) for c, p in zip(current_positions, self.last_joint_positions))
                
                # 如果变化量小于阈值，说明运动已停止
                if max_diff < threshold:

                    self.motion_complete.set()  # 标记运动完成
                    self.is_moving = False
                    break
            
            self.last_joint_positions = current_positions
            time.sleep(0.05)  # 短暂等待
    def _monitor_pose_movement(self, target_pos, target_rot, threshold=0.001):
        """监控位姿运动是否完成"""
        while not self.stop_motion and not self.motion_complete.is_set():
            current_pose = self.arm_controller.get_pose()
            current_pos = current_pose[:3]
            current_rot = current_pose[3:]
            
            if self.last_joint_positions is not None:
                # 计算位置变化（欧氏距离）
                pos_diff = sum((c - p)**2 for c, p in zip(current_pos, self.last_joint_positions[:3])) ** 0.5
                
                # 计算姿态变化（欧拉角最大差）
                ori_diff = max(abs(c - p) for c, p in zip(current_rot, self.last_joint_positions[3:]))
                
                # 如果位置和姿态变化都小于阈值，说明运动已停止
                if pos_diff < threshold and ori_diff < threshold:
                    # 检查是否到达目标位姿
                    pos_error = sum((c - t)**2 for c, t in zip(current_pos, target_pos)) ** 0.5
                    ori_error = max(abs(c - t) for c, t in zip(current_rot, target_rot))
                    if pos_error < threshold and ori_error < threshold:
                        print("运动完成：到达目标位姿")
                    else:
                        print("运动停止：但未到达目标位姿")
                    self.motion_complete.set()  # 标记运动完成
                    self.is_moving = False
                    break
            
            self.last_joint_positions = current_pose
            time.sleep(0.05)  # 短暂等待
=======
            time.sleep(0.01)
        except Exception as e:
            print(f"Move L execution error: {e}")
        finally:
            self.is_moving = False
    
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    def queue_move_j(self, joint_angles, speed=10, is_radians=True):
        """将关节移动命令加入队列"""
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return False  # 命令太频繁，忽略
        
        command = {
            'type': 'move_j',
            'joint_angles': joint_angles,
            'speed': speed,
            'is_radians': is_radians
        }
        
        try:
<<<<<<< HEAD
            # 如果队列已满，尝试替换最后一条命令
            if self.command_queue.full():
                # 获取队列中的所有项目
                items = []
                while not self.command_queue.empty():
                    try:
                        items.append(self.command_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # 保留第一条（正在运行的）和新的命令
                if items:
                    # 放回第一条命令（如果存在）
                    self.command_queue.put_nowait(items[0])
                
                # 添加新命令
                self.command_queue.put_nowait(command)
            else:
                self.command_queue.put_nowait(command)
            
=======
            # 清空队列中的旧命令，只保留最新的
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.command_queue.put_nowait(command)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            self.last_command_time = current_time
            return True
        except queue.Full:
            return False
    
    def queue_move_p(self, pos, rot, speed=10, is_euler=False):
        """将位置控制命令加入队列"""
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return False
        
        command = {
            'type': 'move_p',
            'pos': pos,
            'rot': rot,
            'speed': speed,
            'is_euler': is_euler
        }
        
        try:
<<<<<<< HEAD
            # 如果队列已满，尝试替换最后一条命令
            if self.command_queue.full():
                # 获取队列中的所有项目
                items = []
                while not self.command_queue.empty():
                    try:
                        items.append(self.command_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # 保留第一条（正在运行的）和新的命令
                if items:
                    # 放回第一条命令（如果存在）
                    self.command_queue.put_nowait(items[0])
                
                # 添加新命令
                self.command_queue.put_nowait(command)
            else:
                self.command_queue.put_nowait(command)
            
=======
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.command_queue.put_nowait(command)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            self.last_command_time = current_time
            return True
        except queue.Full:
            return False
    
    def queue_move_l(self, pos, rot, speed=10, is_euler=False):
        """将线性移动命令加入队列"""
        current_time = time.time()
        if current_time - self.last_command_time < self.command_interval:
            return False
        
        command = {
            'type': 'move_l',
            'pos': pos,
            'rot': rot,
            'speed': speed,
            'is_euler': is_euler
        }
        
        try:
<<<<<<< HEAD
            # 如果队列已满，尝试替换最后一条命令
            if self.command_queue.full():
                # 获取队列中的所有项目
                items = []
                while not self.command_queue.empty():
                    try:
                        items.append(self.command_queue.get_nowait())
                    except queue.Empty:
                        break
                
                # 保留第一条（正在运行的）和新的命令
                if items:
                    # 放回第一条命令（如果存在）
                    self.command_queue.put_nowait(items[0])
                
                # 添加新命令
                self.command_queue.put_nowait(command)
            else:
                self.command_queue.put_nowait(command)
            
=======
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.command_queue.put_nowait(command)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            self.last_command_time = current_time
            return True
        except queue.Full:
            return False
    
    def stop(self):
        """停止运动控制器"""
        self.stop_motion = True
        # 清空队列
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        
        if self.motion_thread and self.motion_thread.is_alive():
            self.motion_thread.join(timeout=1.0)
    
    def is_busy(self):
        """检查是否正在移动"""
<<<<<<< HEAD
        return self.is_moving or not self.command_queue.empty() or not self.motion_complete.is_set()
=======
        return self.is_moving or not self.command_queue.empty()
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6


def activate_camera():
    # 读取YAML配置文件
<<<<<<< HEAD
    with open('/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml', 'r') as file:
=======
    with open('./config/camera_correspondence.yaml', 'r') as file:
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
        camera_config = yaml.safe_load(file)

    # 从配置中提取相机ID
    camera_roles = {
<<<<<<< HEAD
        'head': camera_config['middle_camera']['ID'],
        'left_hand': camera_config['left_camera']['ID'],
        'right_hand': camera_config['right_camera']['ID']
=======
        'head': camera_config['head_camera']['ID'],
        'left_hand': camera_config['left_hand_camera']['ID'],
        'right_hand': camera_config['right_hand_camera']['ID']
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    }

    # 查找实际连接的设备
    available_serials, ctx = find_devices()
    print("检测到设备:", available_serials)

    # 初始化相机（用字典存储，键为角色名称）
    rs_pipelines = {}
    for role, cam_id in camera_roles.items():
        if cam_id in available_serials:
            print(f"正在初始化 {role} 相机 (ID: {cam_id})")
            pipeline = RealSenseD400(device=cam_id)
            pipeline.reset()
            time.sleep(5)
            pipeline = RealSenseD400(device=cam_id)  # 重新初始化
            rs_pipelines[role] = pipeline
        else:
            print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")

    # 实时显示画面（使用角色名称作为窗口标题）
    while True:
        for role, pipeline in rs_pipelines.items():
            try:
                pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                cv2.imshow(f"{role.capitalize()} Camera", color_img)
            except Exception as e:
                print(f"从 {role} 相机获取图像失败: {e}")

        if cv2.waitKey(1) == 27:  # ESC退出
            break

    # 释放资源
    for pipeline in rs_pipelines.values():
        pipeline.stop()
    cv2.destroyAllWindows()

class DataCollector:
    def __init__(self):
        self.left_arm_sim = Piper(enable_cc=True,rotmat=rm.rotmat_from_euler(0, 0, 0))
        self.right_arm_sim = Piper(enable_cc=True,rotmat=rm.rotmat_from_euler(0, 0, 0))
        self.left_arm_home_state = {
            "joint_positions": [0, 0, 0, 0, 0, 0],  # 关节角度（单位可能是弧度或度）
            "gripper_opening": 0.0,  # 夹爪开口宽度（米）
            "gripper_effort": -0.04,  # 夹爪扭矩（N·m）
        }
        self.right_arm_home_state = {
            "joint_positions": [0, 0, 0, 0, 0, 0],
            "gripper_opening": 0.0,
            "gripper_effort": -0.04,
        }

        self.left_gripper_effort = 0
        self.right_gripper_effort = 0
        self.ctl_left_arm = True
        self.step_size = 0.01 #单位步长
        self.angle_step = np.deg2rad(10)
        self.speed = 10
        self.move_mode = 0
        self.mouse_sensitivity = 0.1  # 旋转角度/像素 鼠标灵敏度
        #self.last_x, self.last_y = 0, 0  # 初始化鼠标位置
        self.gears = 1
        #self.can_names = ['can0', 'can1']
        print("Creating PiperArmController...")
<<<<<<< HEAD
        self.left_arm_con = PiperArmController(can_name='can0', has_gripper=True)
        time.sleep(0.1)
        self.right_arm_con = PiperArmController(can_name='can1' , has_gripper=True)
=======
        self.left_arm_con = PiperArmController(can_name='can_left', has_gripper=True)
        time.sleep(0.1)
        self.right_arm_con = PiperArmController(can_name='can_right' , has_gripper=True)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
        time.sleep(0.1)
        
        # 初始化平滑运动控制器
        print("Initializing smooth motion controllers...")
        self.left_motion_controller = SmoothMotionController(self.left_arm_con, self.left_arm_sim)
        self.right_motion_controller = SmoothMotionController(self.right_arm_con, self.right_arm_sim)
        
        # 添加键盘重复检测
        self.key_press_times = {}
        self.key_repeat_threshold = 0.1  # 100ms内重复按键视为重复


    #设置采集时的夹爪effort
    def set_gripper_effort(self,l_gripper_effort = 0,r_gripper_effort = 0):
        self.left_gripper_effort = l_gripper_effort
        self.right_gripper_effort = r_gripper_effort

    def set_home_state(self):
        '''
        设置初位，用于开始采集以及结束采集回到的位置。保存在./config/home_state.yaml中
        '''
        left_joint_status = self.left_arm_con.get_joint_values()
        left_gripper_status = self.left_arm_con.get_gripper_status()
        self.left_arm_home_state = {
            "joint_positions": numpy_to_list(left_joint_status),
            "gripper_opening": left_gripper_status[0],
            "gripper_effort": max(0, left_gripper_status[1]),

        }
        right_joint_status = self.right_arm_con.get_joint_values()
        right_gripper_status = self.right_arm_con.get_gripper_status()
        self.right_arm_home_state = {
            "joint_positions": numpy_to_list(right_joint_status),
            "gripper_opening": right_gripper_status[0],
            "gripper_effort": max(0, right_gripper_status[1]),

        }
        # 合并左右臂状态为一个字典
        home_state_data = {
            "left_arm": self.left_arm_home_state,
            "right_arm": self.right_arm_home_state
        }

        # 确保 config 目录存在
        config_dir = "config"
        os.makedirs(config_dir, exist_ok=True)

        # 保存到 YAML 文件
        yaml_path = os.path.join(config_dir, "home_state0.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(home_state_data, f, default_flow_style=False)

        print(f"Home state saved to {yaml_path}")

    def back_to_home_state(self,id=0) -> None:
        """Move the arm and gripper back to the saved home state.

        Reads joint positions and gripper status from `./config/home_state0.yaml`,
        then moves the arms and grippers to the recorded state.
        """

        #如果示教模式打开，提示先关闭示教模式
        left_arm_status = self.left_arm_con.get_status()
        right_arm_status = self.right_arm_con.get_status()

        #if teach_mode == teach:
        #print("示教模式已打开，将机械臂移动至安全位置后运行exit_teach_mode()以退出示教模式")
        #exit()

        # 1. 加载 YAML 文件
<<<<<<< HEAD
        config_path = Path(f'/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/home_state{id}.yaml')
=======
        config_path = Path(f'./config/home_state{id}.yaml')
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
        if not config_path.exists():
            raise FileNotFoundError(f"Home state file not found: {config_path}")

        with open(config_path, "r") as f:
            home_state = yaml.safe_load(f)

        # 2. 控制左臂回到 Home 状态
        if "left_arm" in home_state:
            left_state = home_state["left_arm"]
            # 移动关节到目标角度（从 YAML 中加载的可能是列表或 NumPy 数组）
            left_joint_positions = np.array(left_state["joint_positions"], dtype=float)
            self.left_arm_con.move_j(left_joint_positions, speed=10,is_radians=True,block=False)

            # 控制夹爪
            self.left_arm_con.gripper_control(
                angle=left_state["gripper_opening"],
                effort=left_state["gripper_effort"],
                enable=True
            )

        # 3. 控制右臂回到 Home 状态（逻辑同左臂）
        if "right_arm" in home_state:
            right_state = home_state["right_arm"]
            right_joint_positions = np.array(right_state["joint_positions"], dtype=float)
            print(right_joint_positions)
            self.right_arm_con.move_j(right_joint_positions,speed=10,is_radians=True,block=False)
            self.right_arm_con.gripper_control(
                angle=right_state["gripper_opening"],
                effort=right_state["gripper_effort"],
                enable=True
            )

        print("Successfully returned to home state.")
<<<<<<< HEAD

    def check_move_j_finish(self, target, threshold=0.001):
        """检查关节空间运动是否完成"""
        prev_positions = None
        max_attempts = 100  # 最大尝试次数防止无限循环
        attempts = 0
        
        while attempts < max_attempts:
            # 获取当前关节位置
            current_positions = self.left_arm_con.get_joint_values() if self.ctl_left_arm else self.right_arm_con.get_joint_values()
            
            # 检查是否第一次循环
            if prev_positions is not None:
                # 计算两次读取之间的最大变化量
                max_diff = max(abs(c - p) for c, p in zip(current_positions, prev_positions))
                
                # 如果变化量小于阈值，说明运动已停止
                if max_diff < threshold:
                    # 检查是否到达目标位置
                    error = max(abs(c - t) for c, t in zip(current_positions, target))
                    return error < threshold
            
            # 更新前一次位置并等待
            prev_positions = current_positions
            time.sleep(0.05)  # 短暂等待避免过度占用CPU
            attempts += 1
        
        # 超过最大尝试次数仍未满足条件
        return False

    def check_move_pose_finish(self, target, threshold=0.001):
        """检查笛卡尔空间运动是否完成"""
        prev_pose = None
        max_attempts = 100  # 最大尝试次数防止无限循环
        attempts = 0
        
        while attempts < max_attempts:
            # 获取当前位姿
            current_pose = self.left_arm_con.get_pose() if self.ctl_left_arm else self.right_arm_con.get_pose()
            
            # 检查是否第一次循环
            if prev_pose is not None:
                # 计算位置变化（欧氏距离）
                pos_diff = sum((c - p)**2 for c, p in zip(current_pose[:3], prev_pose[:3])) ** 0.5
                
                # 计算姿态变化（欧拉角最大差）
                ori_diff = max(abs(c - p) for c, p in zip(current_pose[3:], prev_pose[3:]))
                
                # 如果位置和姿态变化都小于阈值，说明运动已停止
                if pos_diff < threshold and ori_diff < threshold:
                    # 检查是否到达目标位姿
                    pos_error = sum((c - t)**2 for c, t in zip(current_pose[:3], target[:3])) ** 0.5
                    ori_error = max(abs(c - t) for c, t in zip(current_pose[3:], target[3:]))
                    return pos_error < threshold and ori_error < threshold
            
            # 更新前一位姿并等待
            prev_pose = current_pose
            time.sleep(0.05)  # 短暂等待避免过度占用CPU
            attempts += 1
        
        # 超过最大尝试次数仍未满足条件
        return False
    
=======
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    def exit_teach_mode(self):
        self.left_arm_con.exit_teach_mode()
        self.right_arm_con.exit_teach_mode()
    
    def cleanup(self):
        """清理资源，停止运动控制器"""
        print("Cleaning up motion controllers...")
        self.left_motion_controller.stop()
        self.right_motion_controller.stop()
        print("Motion controllers stopped.")

    def collect_data_window(self):
        """
        使用键盘控制机械臂末端执行器移动：
        - 方向键 ↑/↓ 控制前后（y 轴）
        - 方向键 ←/→ 控制左右（x 轴）
        - PgUp/PgDn 控制上下（z 轴）
        - 空格键切换夹爪
        - Tab 键切换控制左右臂
        - ESC 退出
        """

        # # 检查示教模式
        # left_teach_mode = self.left_arm_con.is_in_teach_mode()
        # right_teach_mode = self.right_arm_con.is_in_teach_mode()
        #
        # if (self.ctl_left_arm and left_teach_mode) or (not self.ctl_left_arm and right_teach_mode):
        #     print("示教模式已打开，请先关闭示教模式再操作！")
        #     print("将机械臂移动至安全位置后运行exit_teach_mode()以退出示教模式")
        #     return



        print("开始数据采集窗口...")
        print(f"控制{'左侧' if self.ctl_left_arm else '右侧'}机械臂夹爪")
        print("按空格键切换夹爪，ESC退出")
        gripper_open = False
        middle_mouse_pressed = False  # 跟踪鼠标中键状态
        gripper_effort = self.left_gripper_effort if self.ctl_left_arm else self.right_gripper_effort
        arm_con = self.left_arm_con if self.ctl_left_arm else self.right_arm_con
        #current_joint_pos = arm_con.get_joint_values()
        #target_tcp_pos, target_tcp_rotmat = self.left_arm_con.get_pose() if self.ctl_left_arm else self.right_arm_con.get_pose()
        # 初始化末端位姿
        target_tcp_pos, target_tcp_rotmat = arm_con.get_pose()
        last_x,last_y= 0,0
        #print("激活所有相机")
        #activate_camera()

        def on_click(x, y, button, pressed):
            """鼠标点击回调：检测中键按下/释放"""
            nonlocal middle_mouse_pressed,last_x,last_y
            if button == Button.middle:
                middle_mouse_pressed = pressed
                if pressed:
                    last_x, last_y = x, y  # 记录按下时的初始位置
                    print("鼠标中键按下，开始控制末端朝向")
                    print(middle_mouse_pressed)
                else:
                    print("鼠标中键释放，停止控制末端朝向")
            return True

        def on_move(x, y):
            """鼠标移动回调：计算位移差并更新机械臂朝向"""
            nonlocal middle_mouse_pressed, target_tcp_pos, target_tcp_rotmat,last_x,last_y

            # 如果正在执行move_j或者中键未按下，则忽略鼠标移动
            if not middle_mouse_pressed or getattr(arm_con, '_is_moving', False):
                return

            arm_sim = self.left_arm_sim if self.ctl_left_arm else self.right_arm_sim
            if self.move_mode == 0:
                target_tcp_pos, target_tcp_rotmat = self.left_arm_con.get_pose() if self.ctl_left_arm else self.right_arm_con.get_pose()
            else:
                    joint_value = arm_con.get_joint_values()
                    # wrs解末端位置
                    target_tcp_pos, target_tcp_rotmat = arm_sim.fk(joint_value)

            dx = x - last_x
            dy = y - last_y

            # 固定角度增量（单位：弧度）
            fixed_angle_step = np.deg2rad(5)  # 每次旋转 5°

            # 根据鼠标位移方向决定旋转轴
            delta_yaw = 0.0
            delta_pitch = 0.0
            if abs(dx) > abs(dy):
                delta_yaw = np.sign(dx) * fixed_angle_step  # 左右移动 -> Yaw
            else:
                delta_pitch = np.sign(dy) * fixed_angle_step  # 上下移动 -> Pitch

            # 更新欧拉角
            current_rpy = rm.rotmat_to_euler(target_tcp_rotmat)
            new_rpy = [
                current_rpy[0] + delta_yaw,
                current_rpy[1] + delta_pitch,
                current_rpy[2]  # 保持 Roll 不变
            ]

            # 更新旋转矩阵并移动机械臂
            new_rotmat = rm.rotmat_from_euler(*new_rpy)

            # 获取对应的运动控制器
            motion_controller = self.left_motion_controller if self.ctl_left_arm else self.right_motion_controller
            
            if self.move_mode == 0:
                # 使用平滑运动控制器进行位置控制移动
<<<<<<< HEAD
                motion_controller.queue_move_l(target_tcp_pos, new_rotmat, speed=10)
=======
                motion_controller.queue_move_p(target_tcp_pos, new_rotmat, speed=10)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            else:
                # 使用逆运动学计算关节角度
                jnt = arm_sim.ik(target_tcp_pos, new_rotmat)
                if jnt is not None:
                    # 使用平滑运动控制器进行关节移动
                    motion_controller.queue_move_j(jnt, speed=10, is_radians=True)

        def on_press(key):
            nonlocal gripper_open, gripper_effort, arm_con, target_tcp_pos, target_tcp_rotmat

            try:
                # 检查按键重复
                current_time = time.time()
                key_str = str(key)
                if key_str in self.key_press_times:
                    if current_time - self.key_press_times[key_str] < self.key_repeat_threshold:
                        return  # 忽略重复按键
                self.key_press_times[key_str] = current_time

                # 定义旋转矩阵生成函数（罗德里格斯公式）
                def rotmat_from_axangle(axis, angle):
                    axis = axis / np.linalg.norm(axis)
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                    return R

                # 旋转步长（5度）
                angle_step = np.deg2rad(10)
                # init
                gripper_effort = self.left_gripper_effort if self.ctl_left_arm else self.right_gripper_effort
                arm_con = self.left_arm_con if self.ctl_left_arm else self.right_arm_con
                arm_sim = self.left_arm_sim if self.ctl_left_arm else self.right_arm_sim
                motion_controller = self.left_motion_controller if self.ctl_left_arm else self.right_motion_controller

                # 获取当前末端位姿
                if self.move_mode == 0:
                    current_tcp_pos, current_tcp_rotmat = arm_con.get_pose()
                    target_tcp_pos, target_tcp_rotmat = current_tcp_pos, current_tcp_rotmat
                else:
                    joint_value = arm_con.get_joint_values()
                    target_tcp_pos, target_tcp_rotmat = arm_sim.fk(joint_value)

                need_move = False  # 标记是否需要移动机械臂

                # 步长控制
                if hasattr(key, 'char') and key.char:
                    char_key = key.char
                    if char_key == '1':
                        self.step_size = 0.01
                        # 旋转步长（5度）
                        angle_step = np.deg2rad(10)
                        print(f"步长设置为: {self.step_size:.3f}米")
                    elif char_key == '2':
                        self.step_size = 0.02
                        # 旋转步长（5度）
                        angle_step = np.deg2rad(10)
                        print(f"步长设置为: {self.step_size:.3f}米")
                    elif char_key == '3':
                        self.step_size = 0.03
                        # 旋转步长（5度）
                        angle_step = np.deg2rad(10)
                        print(f"步长设置为: {self.step_size:.3f}米")
                    elif char_key == '4':
                        self.step_size = 0.04
                        # 旋转步长（5度）
                        angle_step = np.deg2rad(10)
                        print(f"步长设置为: {self.step_size:.3f}米")


                    #home state

                    elif char_key == '9':
                        self.back_to_home_state(id=0)

                    elif char_key == '0':
                        self.back_to_home_state(id=1)

                if key == keyboard.Key.space:
                    if gripper_open:
                        arm_con.gripper_control(angle=0.0, effort=gripper_effort, enable=True)
                        print("夹爪关闭")
                    else:
                        arm_con.gripper_control(angle=0.08, effort=gripper_effort, enable=True)
                        print("夹爪打开")
                    gripper_open = not gripper_open

                elif key == keyboard.Key.tab:
                    # 切换机械臂控制
                    self.ctl_left_arm = not self.ctl_left_arm
                    print(f"切换到控制{'左侧' if self.ctl_left_arm else '右侧'}机械臂")

                # 旋转控制 - 修复：将旋转控制移到字符检测之外
                rot_axis = None
                rot_angle = 0.0

                if key == keyboard.KeyCode.from_char('r'):  # 绕局部X轴正旋转
                    rot_axis = np.array([1, 0, 0])
                    rot_angle = angle_step
                    print(1)
                elif key == keyboard.KeyCode.from_char('f'):  # 绕局部X轴负旋转
                    rot_axis = np.array([1, 0, 0])
                    rot_angle = -angle_step
                    print(1)
                elif key == keyboard.KeyCode.from_char('t'):  # 绕局部Y轴正旋转
                    rot_axis = np.array([0, 1, 0])
                    rot_angle = angle_step
                    print(1)
                elif key == keyboard.KeyCode.from_char('g'):  # 绕局部Y轴负旋转
                    rot_axis = np.array([0, 1, 0])
                    rot_angle = -angle_step
                    print(1)
                elif key == keyboard.KeyCode.from_char('y'):  # 绕局部Z轴正旋转
                    rot_axis = np.array([0, 0, 1])
                    rot_angle = angle_step
                    print(1)
                elif key == keyboard.KeyCode.from_char('h'):  # 绕局部Z轴负旋转
                    rot_axis = np.array([0, 0, 1])
                    rot_angle = -angle_step
                    print(1)

                # 平移控制
                delta = np.zeros(3)
                if hasattr(key, 'char') and key.char:
                    char_key = key.char.lower()
                    if char_key == 's':  # 末端坐标系 +Y 方向（向前）
                        delta = np.array([0, 0, -self.step_size])
                        print(1)
                    elif char_key == 'w':  # 末端坐标系 -Y 方向（向后）
                        delta = np.array([0, 0, self.step_size])
                        print(1)
                    elif char_key == 'd':  # 末端坐标系 -X 方向（向左）
                        delta = np.array([0, -self.step_size, 0])
                        print(1)
                    elif char_key == 'a':  # 末端坐标系 +X 方向（向右）
                        delta = np.array([0, self.step_size, 0])
                        print(1)
                    elif char_key == 'q':  # 末端坐标系 +Z 方向（向上）
                        delta = np.array([-self.step_size, 0, 0])
                        print(1)
                    elif char_key == 'e':  # 末端坐标系 -Z 方向（向下）
                        delta = np.array([self.step_size, 0, 0])
                        print(1)

                # 处理旋转
                if rot_axis is not None:
                    # 将局部旋转轴转换到世界坐标系
                    world_rot_axis = target_tcp_rotmat @ rot_axis
                    rotmat_delta = rotmat_from_axangle(world_rot_axis, rot_angle)
                    target_tcp_rotmat = rotmat_delta @ target_tcp_rotmat
                    need_move = True

                # 处理平移
                if np.any(delta):
                    world_delta = target_tcp_rotmat @ delta
                    target_tcp_pos += world_delta
                    need_move = True

                if need_move:
                    if self.move_mode == 0:
                        # 使用平滑运动控制器进行线性移动
<<<<<<< HEAD
                        motion_controller.queue_move_l(target_tcp_pos, target_tcp_rotmat, speed=2)
=======
                        motion_controller.queue_move_p(target_tcp_pos, target_tcp_rotmat, speed=2)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
                    else:
                        # 使用逆运动学计算关节角度
                        jnt = arm_sim.ik(target_tcp_pos, target_tcp_rotmat)
                        if jnt is not None:
                            # 使用平滑运动控制器进行关节移动
                            motion_controller.queue_move_j(jnt, speed=5, is_radians=True)

            except AttributeError:
                pass


        def on_release(key):
            nonlocal arm_con
            if key == keyboard.Key.esc:
                arm_con = self.left_arm_con if self.ctl_left_arm else self.right_arm_con
                arm_con.emergency_stop()
                
                # 清理运动控制器
                self.cleanup()

                print("退出数据采集窗口...")
                return False  # 停止监听器
            return None

        #start
        try:
            # with MouseListener(on_move=on_move, on_click=on_click) as mouse_listener, \
            #      keyboard.Listener(on_press=on_press, on_release=on_release) as key_listener:
            #     mouse_listener.join()
            #     key_listener.join()
            with keyboard.Listener(on_press=on_press, on_release=on_release) as key_listener:
                key_listener.join()
                while key_listener.running:
                    # 固定频率处理
                    time.sleep(0.1)  # 20Hz
                    # 可以在这里添加其他需要定期执行的任务
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            # 清理运动控制器
            self.cleanup()
            # 回到初始位置
            self.back_to_home_state(id=1)

    def start_gui(self):
        """启动PyQt图形界面"""
        app = QApplication(sys.argv)
        gui = DataCollectorGUI(self)
        gui.show()
        sys.exit(app.exec())


<<<<<<< HEAD
# import sys
# from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
#                              QLabel, QPushButton, QMessageBox)
# from PyQt6.QtCore import Qt, QTimer
# from PyQt6.QtGui import QKeyEvent


# class DataCollectorGUI(QMainWindow):
#     def __init__(self, data_collector):
#         super().__init__()
#         self.data_collector = data_collector
#         self.gripper_open = False
#         self.ctl_left_arm = True
#         self.init_ui()

#     def init_ui(self):
#         # 主窗口设置
#         self.setWindowTitle("机械臂数据采集工具")
#         self.setGeometry(100, 100, 400, 300)

#         # 中央部件
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)

#         # 布局
#         layout = QVBoxLayout()

#         # 状态标签
#         self.status_label = QLabel("准备就绪", self)
#         self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         layout.addWidget(self.status_label)

#         # 夹爪状态标签
#         self.gripper_label = QLabel("夹爪状态: 关闭", self)
#         self.gripper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         layout.addWidget(self.gripper_label)

#         # 控制按钮
#         self.toggle_gripper_btn = QPushButton("切换夹爪状态 (空格键)", self)
#         self.toggle_gripper_btn.clicked.connect(self.toggle_gripper)
#         layout.addWidget(self.toggle_gripper_btn)

#         # 退出按钮
#         exit_btn = QPushButton("退出 (ESC)", self)
#         exit_btn.clicked.connect(self.close)
#         layout.addWidget(exit_btn)

#         # 机械臂选择按钮
#         arm_select_btn = QPushButton("切换控制机械臂 (当前: 左侧)", self)
#         arm_select_btn.clicked.connect(self.toggle_arm_selection)
#         layout.addWidget(arm_select_btn)

#         central_widget.setLayout(layout)

#         # 初始化状态
#         self.update_status()

#     def toggle_gripper(self):
#         """切换夹爪状态"""
#         gripper_effort = (self.data_collector.left_gripper_effort if self.ctl_left_arm
#                           else self.data_collector.right_gripper_effort)
#         arm_con = (self.data_collector.left_arm_con if self.ctl_left_arm
#                    else self.data_collector.right_arm_con)

#         if self.gripper_open:
#             arm_con.gripper_control(angle=0.0, effort=gripper_effort, enable=True)
#             self.gripper_open = False
#             self.gripper_label.setText("夹爪状态: 关闭")
#         else:
#             arm_con.gripper_control(angle=0.02, effort=gripper_effort, enable=True)
#             self.gripper_open = True
#             self.gripper_label.setText("夹爪状态: 打开")

#         self.update_status()

#     def toggle_arm_selection(self):
#         """切换控制的机械臂"""
#         self.ctl_left_arm = not self.ctl_left_arm
#         self.gripper_open = False
#         self.update_status()

#     def update_status(self):
#         """更新状态显示"""
#         arm_text = "左侧" if self.ctl_left_arm else "右侧"
#         self.status_label.setText(f"控制中: {arm_text}机械臂 | 夹爪状态: {'打开' if self.gripper_open else '关闭'}")

#         # 更新按钮文本
#         for btn in self.findChildren(QPushButton):
#             if btn.text().startswith("切换控制机械臂"):
#                 btn.setText(f"切换控制机械臂 (当前: {arm_text})")

#     def keyPressEvent(self, event: QKeyEvent):
#         """键盘事件处理"""
#         if event.key() == Qt.Key.Key_Space:
#             self.toggle_gripper()
#         elif event.key() == Qt.Key.Key_Escape:
#             self.close()
#         else:
#             super().keyPressEvent(event)

#     def closeEvent(self, event):
#         """窗口关闭事件"""
#         reply = QMessageBox.question(
#             self, '确认退出',
#             '确定要退出数据采集工具吗?',
#             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
#             QMessageBox.StandardButton.No
#         )

#         if reply == QMessageBox.StandardButton.Yes:
#             # 清理运动控制器
#             self.data_collector.cleanup()
#             # 回到初始位置
#             self.data_collector.back_to_home_state()
#             event.accept()
#         else:
#             event.ignore()


if __name__ == "__main__":
    # import threading

    # # 启动相机线程
    # camera_thread = threading.Thread(target=activate_camera, daemon=True)
    # camera_thread.start()

    datacollector = DataCollector()

    #print(datacollector.left_arm_con.get_joint_values())
    #print(datacollector.left_arm_con.get_pose())
    #datacollector.back_to_home_state(id=0)
    # # time.sleep(3)
    # # datacollector.back_to_home_state(id=0)
    #datacollector.collect_data_window()
=======
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QPushButton, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent


class DataCollectorGUI(QMainWindow):
    def __init__(self, data_collector):
        super().__init__()
        self.data_collector = data_collector
        self.gripper_open = False
        self.ctl_left_arm = True
        self.init_ui()

    def init_ui(self):
        # 主窗口设置
        self.setWindowTitle("机械臂数据采集工具")
        self.setGeometry(100, 100, 400, 300)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 布局
        layout = QVBoxLayout()

        # 状态标签
        self.status_label = QLabel("准备就绪", self)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # 夹爪状态标签
        self.gripper_label = QLabel("夹爪状态: 关闭", self)
        self.gripper_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gripper_label)

        # 控制按钮
        self.toggle_gripper_btn = QPushButton("切换夹爪状态 (空格键)", self)
        self.toggle_gripper_btn.clicked.connect(self.toggle_gripper)
        layout.addWidget(self.toggle_gripper_btn)

        # 退出按钮
        exit_btn = QPushButton("退出 (ESC)", self)
        exit_btn.clicked.connect(self.close)
        layout.addWidget(exit_btn)

        # 机械臂选择按钮
        arm_select_btn = QPushButton("切换控制机械臂 (当前: 左侧)", self)
        arm_select_btn.clicked.connect(self.toggle_arm_selection)
        layout.addWidget(arm_select_btn)

        central_widget.setLayout(layout)

        # 初始化状态
        self.update_status()

    def toggle_gripper(self):
        """切换夹爪状态"""
        gripper_effort = (self.data_collector.left_gripper_effort if self.ctl_left_arm
                          else self.data_collector.right_gripper_effort)
        arm_con = (self.data_collector.left_arm_con if self.ctl_left_arm
                   else self.data_collector.right_arm_con)

        if self.gripper_open:
            arm_con.gripper_control(angle=0.0, effort=gripper_effort, enable=True)
            self.gripper_open = False
            self.gripper_label.setText("夹爪状态: 关闭")
        else:
            arm_con.gripper_control(angle=0.02, effort=gripper_effort, enable=True)
            self.gripper_open = True
            self.gripper_label.setText("夹爪状态: 打开")

        self.update_status()

    def toggle_arm_selection(self):
        """切换控制的机械臂"""
        self.ctl_left_arm = not self.ctl_left_arm
        self.gripper_open = False
        self.update_status()

    def update_status(self):
        """更新状态显示"""
        arm_text = "左侧" if self.ctl_left_arm else "右侧"
        self.status_label.setText(f"控制中: {arm_text}机械臂 | 夹爪状态: {'打开' if self.gripper_open else '关闭'}")

        # 更新按钮文本
        for btn in self.findChildren(QPushButton):
            if btn.text().startswith("切换控制机械臂"):
                btn.setText(f"切换控制机械臂 (当前: {arm_text})")

    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件处理"""
        if event.key() == Qt.Key.Key_Space:
            self.toggle_gripper()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(
            self, '确认退出',
            '确定要退出数据采集工具吗?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 清理运动控制器
            self.data_collector.cleanup()
            # 回到初始位置
            self.data_collector.back_to_home_state()
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    import threading

    # 启动相机线程
    camera_thread = threading.Thread(target=activate_camera, daemon=True)
    camera_thread.start()

    datacollector = DataCollector()
    datacollector.back_to_home_state(id=1)
    time.sleep(3)
    datacollector.back_to_home_state(id=0)
    datacollector.collect_data_window()
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

    #可以查看示教是否打开
    #print(datacollector.left_arm_con.get_status())
    # print("末端执行器get_pose():")
    #print(datacollector.left_arm_con.get_pose())
    #datacollector.set_home_state()

    #datacollector.start_gui()
    # print(datacollector.left_arm_con.get_joint_values_raw())

    # print("关闭示教")
    #datacollector.exit_teach_mode()
    #print(datacollector.left_arm_con.get_status())
<<<<<<< HEAD
    # import threading

    # # 启动相机线程
    # camera_thread = threading.Thread(target=activate_camera, daemon=True)
    # camera_thread.start()
=======

>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    try:
        time.sleep(100)
    except KeyboardInterrupt:
        print("程序被中断")
    finally:
        # 确保清理资源
        if 'datacollector' in locals():
            datacollector.cleanup()
 
