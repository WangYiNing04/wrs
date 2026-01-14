'''
Author: wang yining
Date: 2025-12-31 00:43:04
LastEditTime: 2026-01-14 19:49:53
FilePath: /wrs_tiaozhanbei/wrs/robot_con/openarm/openarm.py
Description: 
e-mail: wangyining0408@outlook.com
'''

import time
import numpy as np
import sys
from openarm_can import OpenArm as HWOpenArm
from openarm_can import MITParam, PosVelParam

import wrs.basis.robot_math as rm
import os
import subprocess
import glob
from contextlib import contextmanager
import io

#import ik_solver
from wrs.robot_sim.manipulators.openarm.openarm import OpenArm

#run setup_vcan.sh firstly
#use canfd slcan

#计算动力学矩阵
try:
    # 优先从 cmeel.prefix 导入正确的 pinocchio（如果存在）
    import sys
    import os
    cmeel_path = None
    for path in sys.path:
        if 'site-packages' in path:
            cmeel_pinocchio = os.path.join(path, 'cmeel.prefix', 'lib', 'python3.10', 'site-packages')
            if os.path.exists(cmeel_pinocchio):
                cmeel_path = cmeel_pinocchio
                break
    
    if cmeel_path:
        sys.path.insert(0, cmeel_path)
    
    import pinocchio as pin
    
    # 尝试不同的导入方式以兼容不同版本的 pinocchio
    try:
        # 方法1: 直接使用 pin.buildModelFromUrdf (pinocchio 2.x+)
        if hasattr(pin, 'buildModelFromUrdf'):
            build_model = pin.buildModelFromUrdf
        # 方法2: 从 urdf 子模块导入
        elif hasattr(pin, 'urdf') and hasattr(pin.urdf, 'buildModelFromUrdf'):
            build_model = pin.urdf.buildModelFromUrdf
        # 方法3: 尝试从 pinocchio.urdf 导入
        else:
            from pinocchio.urdf import buildModelFromUrdf as build_model
        
        # 加载模型
        urdf_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_sim/robots/openarm/openarm.urdf"
        model = build_model(urdf_path)
        data = model.createData()

        def get_gravity_torque(q):
            # q 是当前的关节弧度
            return pin.computeGeneralizedGravity(model, data, q)
        
    except AttributeError as e:
        # 如果所有方法都失败，尝试使用 pinocchio 的其他 API
        try:
            # 尝试使用 pinocchio 的旧 API
            import pinocchio.urdf as urdf
            urdf_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_sim/robots/openarm/openarm.urdf"
            model = urdf.buildModelFromUrdf(urdf_path)
            data = model.createData()
            
            def get_gravity_torque(q):
                return pin.computeGeneralizedGravity(model, data, q)
        except Exception as e2:
            print(f"无法加载 Pinocchio 模型，错误：{e2}")
            print("提示：请检查 pinocchio 版本和 URDF 文件路径")
            raise
    
except ImportError:
    print("Pinocchio 未安装，重力补偿功能不可用。")
    def get_gravity_torque(q):
        return np.zeros(len(q))
except Exception as e:
    print(f"加载 Pinocchio 模型时出错：{e}")
    print("提示：请确保安装了正确版本的 pinocchio (机器人学库)")
    def get_gravity_torque(q):
        return np.zeros(len(q))

try:
    import wrs.motion.trajectory.piecewisepoly_toppra as pwp

    TOPPRA_EXIST = True
except:
    TOPPRA_EXIST = False

########################################################################################
from openarm_can import MotorType

ARM_MOTOR_TYPES = [
    MotorType.DM8009,
    MotorType.DM8009,
    MotorType.DM4340,
    MotorType.DM4340,
    MotorType.DM4310,
    MotorType.DM4310,
    MotorType.DM4310,
]

ARM_SEND_IDS =  list(range(0x01, 0x08))
ARM_RECV_IDS =  list(range(0x11, 0x18))


GRIPPER_MOTOR_TYPES = [
    MotorType.DM4310, # gripper
]

GRIPPER_SEND_IDS = [0x08]
GRIPPER_RECV_IDS = [0x18]

########################################################################################
'''
force_vcan_setup: bool
    如果设置为 True,则强制运行 setup_vcan.sh 脚本以设置 vcan0 接口。
'''
class OpenArmController:  
    def __init__(self,
                 can_name="vcan0",
                 motor_types=None,
                 send_ids=None,
                 recv_ids=None,
                 has_gripper=True,
                 auto_enable=True,
                 force_vcan_setup=False,
                 load_meshes=False):

        if can_name == "vcan0":
            self._ensure_vcan(force_setup=force_vcan_setup)

        self._openarm = HWOpenArm(can_name, False)

        # ---- 使用默认值 ----
        self._motor_types = motor_types or ARM_MOTOR_TYPES
        self._send_ids = send_ids or ARM_SEND_IDS
        self._recv_ids = recv_ids or ARM_RECV_IDS

        assert len(self._motor_types) == len(self._send_ids) == len(self._recv_ids)

        # ---- 关键：初始化 arm motors ----
        self._openarm.init_arm_motors(
            self._motor_types,
            self._send_ids,
            self._recv_ids
        )

        # ---- gripper ----
        self._has_gripper = has_gripper
        if has_gripper:
            self._openarm.init_gripper_motor(
                MotorType.DM4310,
                0x08,
                0x18
            )

        self._arm = self._openarm.get_arm()
        self._gripper = self._openarm.get_gripper()
        
        self.dof = len(self._arm.get_motors())

        # ---- joint limits ----
        self._joint_limits = np.array([
            [-1.396263,  3.490659],   # J1
            [-1.745329,  1.745329],   # J2
            [-1.570796,  1.570796],   # J3
            [ 0.0,       2.443461],   # J4
            [-1.570796,  1.570796],   # J5
            [-0.785398,  0.785398],   # J6
            [-1.570796,  1.570796],   # J7
        ])

        self._gripper_limits = np.array([-0.41523613, 0.58766308])

        if auto_enable:
            self.enable()

        #ik_solver
        self.arm_sim = OpenArm(load_meshes=False)
        
        # 示教模式相关参数
        self.teaching_mode = False
        self._prev_q = None
        self._prev_time = None
        
        # 补偿参数（可通过校准方法设置）
        self.friction_params = np.array([0.0] * self.dof)  # 库伦摩擦系数
        self.damping_params = np.array([0.0] * self.dof)  # 阻尼系数
        
        # 常量文件夹路径（绝对路径）
        self._constants_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constant")
        # 确保常量文件夹存在
        os.makedirs(self._constants_dir, exist_ok=True)
        
        # 默认参数文件路径
        self._default_friction_file = os.path.join(self._constants_dir, "friction_params.npy")
        self._default_damping_file = os.path.join(self._constants_dir, "damping_params.npy")
        
        # 加载已保存的参数
        self.load_compensation_params(friction_file=self._default_friction_file, 
                                     damping_file=self._default_damping_file)

    def _ensure_vcan(self, force_setup=False):
        """
        Checks if vcan0 exists. If not (or if forced), checks for /dev/ttyACM* and runs setup_vcan.sh.
        """
        vcan_exists = False
        # Check if vcan0 is in network interfaces
        try:
            with open('/proc/net/dev', 'r') as f:
                if 'vcan0' in f.read():
                    vcan_exists = True
        except FileNotFoundError:
            pass

        if not vcan_exists or force_setup:
            print(f"vcan0 setup required (exists: {vcan_exists}, force: {force_setup})...")
            
            # Check for /dev/ttyACM* devices
            acm_devices = glob.glob("/dev/ttyACM*")
            if not acm_devices:
                raise RuntimeError("Error: No /dev/ttyACM* devices found. Cannot setup vcan.")
            
            print(f"Found devices: {acm_devices}. Running setup_vcan.sh...")
            
             # ⭐ 关键：定位 setup_vcan.sh 的绝对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            setup_script = os.path.join(script_dir, "setup_vcan.sh")

            if not os.path.isfile(setup_script):
                raise FileNotFoundError(f"setup_vcan.sh not found at: {setup_script}")

            print(f"Running: {setup_script}")

            try:
                subprocess.run(
                    ["sudo", "-E", "bash", setup_script], #设置visudo或者以root权限运行
                    check=True
                )
                time.sleep(1)
                print("vcan0 setup completed successfully.")

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to execute setup_vcan.sh: {e}")

    # ---- Power ----
    @property
    def is_enabled(self) -> bool:
        # 确保处于状态模式（callback_mode=0）才能读取实时状态
        return False

    def enable(self):
        self._openarm.enable_all()
        time.sleep(0.2)
        print("OpenArm motors enabled.")


    def disable(self):
        self._openarm.disable_all()
        print("OpenArm motors disabled.")


    def switch_to_mit_mode(self):
        # 检查当前模式是否为 MIT 模式
        current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
        if not all(mode == 1 for mode in current_modes if mode is not None):  # 1 表示 MIT 模式
            print("切换到 MIT 模式...")
            self.set_ctrl_mode_all(control_mode=1)
            time.sleep(0.1)  # 等待模式切换完成

    # ---- Joint space ----
    def move_j(self,
           joint_angles,
           speed : float= 0.3,
           *,
           is_radians=True,
           block=False,
           tolerance=0.01,
           debug: bool= False):

        angles = np.asarray(joint_angles, dtype=float)
        if angles.size != self.dof:
            raise ValueError("Invalid joint dimension")

        if not is_radians:
            angles = np.radians(angles)

        params = [
            PosVelParam(q, speed)
            for q in angles
        ]

        self._arm.posvel_control_all(params)

        if block:
            self._wait_until_joint_reached(angles, tolerance,debug=debug)


    def move_m(self,
            target_tau : list):

        self.switch_to_mit_mode()

        for i in range(self.dof):
            # 使用 MIT 模式下发，Kp, Kd 设为 0 即为纯力矩模式
            param = MITParam(q=0, dq=0, kp=0, kd=0, tau=target_tau[i])
            self._arm.mit_control_one(i, param)


    #TODO
    def move_jntspace_highfreq(self, joint_angles, *, vel=0.0):
        angles = np.asarray(joint_angles, dtype=float)
        if angles.size != self.dof:
            raise ValueError("Invalid joint dimension")

        params = [
            PosVelParam(q, vel)
            for q in angles
        ]

        self._arm.posvel_control_all(params)
    
    
    def move_jntspace_path(self,
                        path,
                        max_jntvel: list = None,
                        max_jntacc: list = None,
                        start_frame_id=1,
                        speed=0.1,
                        control_frequency=0.01, ):
        """
        :param path: [jnt_values0, jnt_values1, ...], results of motion planning
        :param max_jntvel: 1x6 list to describe the maximum joint speed for the arm
        :param max_jntacc: 1x6 list to describe the maximum joint acceleration for the arm
        :param start_frame_id:
        :return:
        author: weiwei
        """
        if TOPPRA_EXIST:
            if path is None:
                raise ValueError("The given is incorrect!")
            path = np.array(path)
            # # Refer to https://www.ufactory.cc/_files/ugd/896670_9ce29284b6474a97b0fc20c221615017.pdf
            # # the robotic arm can accept joint position commands sent at a fixed high frequency like 100Hz
            control_frequency = control_frequency
            tpply = pwp.PiecewisePolyTOPPRA() 
            interpolated_path = tpply.interpolate_by_max_spdacc(path=path,
                                                                ctrl_freq=control_frequency,
                                                                max_vels=max_jntvel,
                                                                max_accs=max_jntacc,
                                                                toggle_debug=False)
            interpolated_path = interpolated_path[start_frame_id:]
            for jnt_values in interpolated_path:
                self.move_j(
                    joint_angles=jnt_values,
                    is_radians  = True,
                    speed=speed,
                    block=False, )
                time.sleep(.1)
            return
        else:
            raise NotImplementedError

    #TODO
    # ---- Cartesian ----
    def move_p(self,
            pos,
            rotmat,
            speed: float= 0.3,
            block: bool=False,
            control_frequency: float=0.1):
        current_joints = self.get_joint_values()
        joints_values = self.arm_sim.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=current_joints)
        print(f"求解出的关节角度:{joints_values}")

        if joints_values is None:
            print("无法求解该位置的逆解，动作被忽略。")
            return
        
        #self.move_j(joints_values, block=block, speed=speed)

        joints_values_path = [current_joints, joints_values]
        self.move_jntspace_path(path= joints_values_path,speed= speed,control_frequency=control_frequency)
        

    #move_l(...)

    # ---- Feedback ----
    def get_joint_values(self) -> np.ndarray:
        # 确保处于状态模式（callback_mode=0）才能读取实时状态
        self._openarm.set_callback_mode_all(0)  # 0为state模式,1为param模式
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_position() for m in self._arm.get_motors()])

    def get_pose(self):
        joints_values = self.get_joint_values()
        pos, rot = self.arm_sim.fk(joints_values)
        return pos, rot


    def get_joint_torques(self):
        # 确保处于状态模式（callback_mode=0）才能读取实时状态
        self._openarm.set_callback_mode_all(0)  # 0为state模式,1为param模式
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_torque() for m in self._arm.get_motors()])
    
    #多余
    # def get_joint_velocities(self, use_history=False):
    #     """
    #     获取关节速度（通过位置差分计算）
        
    #     Args:
    #         use_history: 如果为True，使用历史位置队列来计算更平滑的速度（适合慢速拖动）
        
    #     Returns:
    #         np.ndarray: 关节速度数组 (rad/s)
    #     """
    #     # 初始化历史队列（如果不存在）
    #     if not hasattr(self, '_q_history'):
    #         self._q_history = []
    #         self._t_history = []
        
    #     # 获取当前位置和时间（确保每次都刷新数据）
    #     current_q = self.get_joint_values()
    #     current_time = time.time()
        
    #     # 对于慢速拖动，使用历史数据平滑速度计算
    #     if use_history:
    #         # 将当前位置添加到历史队列
    #         self._q_history.append(current_q.copy())
    #         self._t_history.append(current_time)
            
    #         # 保持队列长度不超过10（增加队列长度以获得更稳定的速度估计）
    #         if len(self._q_history) > 10:
    #             self._q_history.pop(0)
    #             self._t_history.pop(0)
            
    #         # 如果历史数据足够，使用线性拟合计算速度
    #         if len(self._q_history) >= 3:
    #             # 使用最近3个点进行线性拟合
    #             q_array = np.array(self._q_history[-3:])
    #             t_array = np.array(self._t_history[-3:])
    #             dt_total = t_array[-1] - t_array[0]
                
    #             if dt_total > 1e-6:
    #                 # 对每个关节分别计算速度
    #                 dq = np.zeros(self.dof)
    #                 for j in range(self.dof):
    #                     # 线性拟合：q = a + b*t，速度就是 b
    #                     # 使用相对时间（从第一个点开始）
    #                     t_rel = t_array - t_array[0]
    #                     if np.std(t_rel) > 1e-6:  # 确保时间有变化
    #                         try:
    #                             coeffs = np.polyfit(t_rel, q_array[:, j], 1)
    #                             dq[j] = coeffs[0]  # 斜率就是速度
    #                         except:
    #                             # 如果拟合失败，使用简单差分
    #                             dq[j] = (q_array[-1, j] - q_array[0, j]) / dt_total
    #                     else:
    #                         dq[j] = 0.0
    #                 return dq
    #             else:
    #                 # 时间间隔太小，返回零速度
    #                 return np.zeros(self.dof)
    #         else:
    #             # 历史数据不足，返回零速度
    #             return np.zeros(self.dof)
        
    #     # 标准速度计算（单点差分）- 不使用历史数据时
    #     # 检查是否需要初始化
    #     if not hasattr(self, '_prev_q') or self._prev_q is None or \
    #        not hasattr(self, '_prev_time') or self._prev_time is None:
    #         # 首次调用，初始化
    #         self._prev_q = current_q.copy()
    #         self._prev_time = current_time
    #         return np.zeros(self.dof)
        
    #     # 计算时间差
    #     dt = current_time - self._prev_time
    #     if dt < 1e-6:  # 避免除零
    #         return np.zeros(self.dof)
        
    #     # 计算速度（位置差分）
    #     dq = (current_q - self._prev_q) / dt
        
    #     # 更新历史值
    #     self._prev_q = current_q.copy()
    #     self._prev_time = current_time
        
    #     return dq
    

    def get_joint_velocities(self):
        # 确保处于状态模式（callback_mode=0）才能读取实时状态
        self._openarm.set_callback_mode_all(0)  # 0为state模式,1为param模式
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_velocity() for m in self._arm.get_motors()])
        

    def get_gravity_torque(self, q=None):
        """
        计算指定关节角度下的重力补偿力矩。
        
        Args:
            q: 关节角度数组（弧度），如果为None则使用当前关节角度
        
        Returns:
            np.ndarray: 重力补偿力矩数组 (N·m)
        """
        if q is None:
            q = self.get_joint_values()
        else:
            q = np.asarray(q)
            if q.size != self.dof:
                raise ValueError(f"关节角度维度不匹配：期望 {self.dof}，得到 {q.size}")
        
        return get_gravity_torque(q)

    def get_gripper_status(self):
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_position() for m in self._gripper.get_motors()])
    
    # ---- Gripper ----

    def open_gripper(self, vel: float = 0.2):
        """
        打开夹爪（到最大开口）
        """
        if not self._has_gripper:
            return

        open_pos = self._gripper_limits[0] 
        param = PosVelParam(open_pos, vel)
        self._gripper.posvel_control_one(0, param)


    def close_gripper(self, vel: float = 0.2):
        """
        闭合夹爪（到最大闭合）
        """
        if not self._has_gripper:
            return

        close_pos = self._gripper_limits[1] 
        param = PosVelParam(close_pos, vel)
        self._gripper.posvel_control_one(0, param)

    
    def map_gripper_width_to_motor_position(self, gripper_width):
        """
        将夹爪宽度（米）映射到电机位置（弧度）
        
        参数:
            gripper_width: 夹爪宽度，单位米，范围 [0, 0.085]
        
        返回:
            motor_position: 电机位置，单位弧度，范围 [-0.41523613, 0.58766308]
        
        映射关系:
            [0, 0.085] 米 --> [-0.41523613, 0.58766308] 弧度
            0.0 米 (打开) --> -0.41523613 弧度
            0.085 米 (闭合) --> 0.58766308 弧度
        """
        # 夹爪宽度范围
        width_min = 0.0      # 米，完全打开
        width_max = 0.085   # 米，完全闭合
        
        # 电机位置范围（从 openarm.py 中的 _gripper_limits）
        motor_open = self._gripper_limits[0]    # 弧度，打开位置
        motor_close = self._gripper_limits[1]   # 弧度，闭合位置
        
        # 限制输入范围
        gripper_width = np.clip(gripper_width, width_min, width_max)
        
        # 线性映射
        if width_max == width_min:
            # 避免除零
            return motor_open
        
        # 归一化到 [0, 1]
        normalized = (gripper_width - width_min) / (width_max - width_min)
        
        # 映射到电机位置范围
        motor_position = motor_close - normalized * (motor_close - motor_open)
        
        return motor_position

    def map_motor_position_to_gripper_width(self, motor_position):
        """
        将电机位置（弧度）映射到夹爪宽度（米）
        这是 map_gripper_width_to_motor_position 的反向映射
        
        参数:
            motor_position: 电机位置，单位弧度，范围 [-0.41523613, 0.58766308]
        
        返回:
            gripper_width: 夹爪宽度，单位米，范围 [0, 0.085]
        
        映射关系:
            [-0.41523613, 0.58766308] 弧度 --> [0, 0.085] 米
            -0.41523613 弧度 (闭合) --> 0.085 米
            0.58766308 弧度 (打开) --> 0.0 米
        """
        # 电机位置范围
        motor_open = self._gripper_limits[0]    # 弧度，打开位置
        motor_close = self._gripper_limits[1]   # 弧度，闭合位置
        
        # 夹爪宽度范围
        width_min = 0.0      # 米，完全打开
        width_max = 0.085   # 米，完全闭合
        
        # 限制输入范围
        motor_position = np.clip(motor_position, motor_open, motor_close)
        
        # 线性映射
        if motor_close == motor_open:
            # 避免除零
            return width_min
        
        # 从电机位置反推归一化值
        # 原公式: motor_position = motor_close - normalized * (motor_close - motor_open)
        # 反推: normalized = (motor_close - motor_position) / (motor_close - motor_open)
        normalized = (motor_close - motor_position) / (motor_close - motor_open)
        
        # 映射到夹爪宽度范围
        gripper_width = width_min + normalized * (width_max - width_min)
        
        return float(gripper_width)

    def gripper_control(self, pos: float, vel: float = 0.2):
        """
        夹爪位置控制（归一化输入）
        
        输入夹爪的真实宽度
        """
        if not self._has_gripper:
            return

        if pos > 0.085 or pos < 0.0:
            print("夹爪位置超出范围，动作被忽略。")
            return

        gripper_position = self.map_gripper_width_to_motor_position(pos)

        print(f"gripper_position: {gripper_position}")

        param = PosVelParam(gripper_position, vel)
        self._gripper.posvel_control_one(0, param)

    def _wait_until_joint_reached(self, target, tol,debug: bool= False):
        while True:
            q = self.get_joint_values()
            if np.max(np.abs(q - target)) < tol:
                break
            if debug:
                print(f'[DEBUG] joints_values: {q}')
            time.sleep(0.01)


    # ---- Low-level / utils ----
    # emergency_stop()
    # close_connection()

    '''
    以下为debug而写的调试函数
    '''
    # ---- Debug / Single motor ----
    def get_motor(self, index: int):
        motors = self._arm.get_motors()
        if index < 0 or index >= len(motors):
            raise IndexError(f"Motor index {index} out of range")
        return motors[index]
    
    def enable_motor(self, index: int):
        m = self.get_motor(index)
        m.enable()
        time.sleep(0.05)
        print(f"Motor {index} enabled (send_id={hex(m.send_id)})")

    def disable_motor(self, index: int):
        m = self.get_motor(index)
        m.disable()
        print(f"Motor {index} disabled")


    # ---- Debug: single motor control ----
    '''
    index = 0表示 第一个电机
    
    序号 0 ~ 7 ,电机 1 ~ 8
    '''
    def move_single_motor(self,
                        index: int,
                        position: float,
                        *,
                        vel: float= 0.3):

        if index < 0 or index >= self.dof:
            raise IndexError(f"Motor index {index} out of range")

        param = PosVelParam(position, vel)
        self._arm.posvel_control_one(index, param)

    def move_gripper_motor(self,
                        position: float,
                        *,
                        index: int= 0,
                        vel: float= 0.3):

        param = PosVelParam(position, vel)
        self._gripper.posvel_control_one(0, param)


    #TODO
    def move_single_motor_mit(self,
                          index: int,
                          target_rad: float,
                          *,
                          kp=20.0,
                          kd=1.0,
                          freq=500,
                          duration=2.0):
        """
        使用 MIT 模式将单个电机转到指定弧度
        """

        motor_count = self.dof
        if index < 0 or index >= motor_count:
            raise IndexError(f"Motor index {index} out of range")

        dt = 1.0 / freq
        param = MITParam(target_rad, -0.5, kp, kd, 0.0)

        steps = int(duration * freq)

        for _ in range(steps):
            self._arm.mit_control_one(index, param)
            time.sleep(dt)

    #查询电机参数
    def get_all_motor_params(self, rid, timeout_us=2000, suppress_warnings=True):
        """
        function:
            修改rid查询不同参数
        
        查询所有臂部电机的特定参数（非实时状态）。
        
        :param rid: MotorVariable 枚举值 (例如 MotorVariable.PMAX, MotorVariable.KP_ASR)
        :param timeout_us: 等待 CAN 帧回传的超时时间（微秒）
        :param suppress_warnings: 是否抑制警告信息（默认True）
        :return: 包含所有电机该参数值的 list
        1. 安全限制与阈值 (Limits & Thresholds)
            UV_Value (Under Voltage): 欠压保护阈值。电压低于此值电机停止工作。
            OV_Value (Over Voltage): 过压保护阈值。防止回馈制动导致电压过高烧毁驱动器。
            OT_Value (Over Temperature): 过温保护阈值。监测 MOS 管或线圈温度。
            OC_Value (Over Current): 过流保护阈值。
            PMAX / VMAX / TMAX: 电机允许运行的最大 位置 (Position)、速度 (Velocity) 和 力矩 (Torque)。
        2. 运动特性与规划 (Motion Profiling)
            ACC / DEC: 梯形波控制模式下的加速度与减速度。
            MAX_SPD: 最大限制转速。
            Damp (Damping): 阻尼系数。在力矩模式或阻抗控制中用于增加系统稳定性。
            Inertia: 负载惯量补偿参数。
        3. PID 与控制环路 (Control Loops)
            KP_ASR / KI_ASR: 速度环（Speed Regulator）的比例和积分增益。
            KP_APR / KI_APR: 位置环（Position Regulator）的比例和积分增益。
            I_BW (Current Bandwidth): 电流环带宽。
            V_BW (Velocity Bandwidth): 速度环带宽。
            GREF: 指令平滑增益。
            Shutterstock
        4. 电机物理特性 (Motor Physics)
            NPP (Number of Pole Pairs): 电机的极对数（极其重要，设错会导致速度计算错误）
            Rs / LS: 定子电阻 (Resistance) 和 电感 (Inductance)。
            Flux: 磁链常数。
            KT_Value: 力矩常数 (N⋅m/A)。
            Gr (Gear Ratio): 减速比。如果是直驱则为 1。
        5. 身份与版本 (ID & Version)
            MST_ID (Master ID): 主机 CAN ID。
            ESC_ID: 电机本身的 CAN ID。
            SN (Serial Number): 电机唯一序列号。
            hw_ver / sw_ver / sub_ver: 硬件、软件及子版本号。
        6. 校准与偏移 (Calibration & Offsets)
            u_off / v_off: U 相和 V 相电流传感器的零点偏移校准。
            m_off (Mechanical Offset): 机械零点偏移量。
            dir (Direction): 电机正反转方向设置。
            p_m: 当前位置的机械映射值。
        7. 内部通信与状态 (System & Misc)
            TIMEOUT: CAN 通信超时保护时间。如果在此时间内没收到心跳，电机可能进入保护模式。
            CTRL_MODE: 当前控制模式（如位置模式、速度模式、力矩模式、MIT 模式等）。
            can_br (Baud Rate): CAN 总线波特率设置。
            xout: 调试输出变量。
            COUNT: 枚举总数，通常用于循环遍历所有参数。

        UV_Value = 0,
        KT_Value = 1,
        OT_Value = 2,
        OC_Value = 3,
        ACC = 4,
        DEC = 5,
        MAX_SPD = 6,
        MST_ID = 7,
        ESC_ID = 8,
        TIMEOUT = 9,
        CTRL_MODE = 10,
        Damp = 11,
        Inertia = 12,
        hw_ver = 13,
        sw_ver = 14,
        SN = 15,
        NPP = 16,
        Rs = 17,
        LS = 18,
        Flux = 19,
        Gr = 20,
        PMAX = 21,
        VMAX = 22,
        TMAX = 23,
        I_BW = 24,
        KP_ASR = 25,
        KI_ASR = 26,
        KP_APR = 27,
        KI_APR = 28,
        OV_Value = 29,
        GREF = 30,
        Deta = 31,
        V_BW = 32,
        IQ_c1 = 33,
        VL_c1 = 34,
        can_br = 35,
        sub_ver = 36,
        u_off = 50,
        v_off = 51,
        k1 = 52,
        k2 = 53,
        m_off = 54,
        dir = 55,
        p_m = 80,
        xout = 81,
        COUNT = 82

        """
        from openarm_can import MotorVariable
        
        # 抑制警告信息的上下文管理器
        @contextmanager
        def suppress_stderr():
            if suppress_warnings:
                # 保存原始的 stderr
                original_stderr = sys.stderr
                # 创建一个空的 StringIO 对象来捕获输出
                sys.stderr = io.StringIO()
                try:
                    yield
                finally:
                    # 恢复原始的 stderr
                    sys.stderr = original_stderr
            else:
                yield

        with suppress_stderr():
            self._openarm.set_callback_mode_all(1) # 0为state模式,1为param模式
            # 1. 发送查询指令给所有电机
            # rid 对应 C++ 层的 RID 枚举，在 Python 中绑定为 MotorVariable
            self._openarm.query_param_all(rid)
            
            # 2. 等待并接收 CAN 回包
            # 查询参数通常比状态回传慢，建议设置稍大的 timeout
            self._openarm.refresh_all()
            self._openarm.recv_all()

        # 3. 从每个电机对象中提取获取到的参数
        params = []
        for m in self._arm.get_motors():
            # get_param 返回的是 ParamResult 结构体
            res = m.get_param(rid)
            try:
                params.append(res)
            except Exception as e:
                if not suppress_warnings:
                    print(f"Error getting parameter {rid}: {e}")
        
        # 查询完成后，切换回状态模式以便后续读取实时状态
        self._openarm.set_callback_mode_all(0)
        
        return params
    
    #TODO
    def get_full_motor_config(self, index: int):
        """
        查询单个电机的一组核心配置参数（调试用）
        """
        from openarm_can import MotorVariable
        target_rids = [
            10
        ]
        
        results = {}
        motors = self._arm.get_motors()
        if index >= len(motors): return results

        for rid in target_rids:
            self._openarm.query_param_all(rid)
            self._openarm.recv_all(1000)
            res = motors[index].get_param(rid)
            results[str(rid)] = res.value if res.valid else "Timeout"
            
        return results
    
    def set_ctrl_mode_all(self,control_mode: int= 1):
        """
        设置所有电机的控制模式为位置模式
        control_mode: MIT = 1, POS_VEL = 2, VEL = 3, TORQUE_POS = 4
        """
        from openarm_can import MotorVariable

        if control_mode not in [1,2,3,4]:
            raise ValueError("Invalid control mode. Use 1 (MIT), 2 (POS_VEL), 3 (VEL), or 4 (TORQUE_POS).")
        
        self._openarm.set_callback_mode_all(1)  # 参数模式用于设置控制模式
        self._openarm.set_ctrl_mode_all(control_mode)
        # 设置完成后切换回状态模式，以便读取实时状态
        self._openarm.set_callback_mode_all(0)

    def start_lead_through(self, 
                           enable_damping=True,
                           enable_friction=True,
                           damping_coeffs=None,
                           friction_coeffs=None):
        """
        开启示教模式（力拖动），切换到 MIT 模式并实时下发力矩指令。
        
        Args:
            enable_damping: 是否启用阻尼补偿
            enable_friction: 是否启用摩擦力补偿
            damping_coeffs: 阻尼系数数组，如果为None则使用self.damping_params
            friction_coeffs: 摩擦力系数数组，如果为None则使用self.friction_params
        """
        from openarm_can import MotorVariable

        # 检查当前模式是否为 MIT 模式
        current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
        if not all(mode == 1 for mode in current_modes if mode is not None):  # 1 表示 MIT 模式
            print("切换到 MIT 模式...")
            self.set_ctrl_mode_all(control_mode=1)
            time.sleep(0.1)  # 等待模式切换完成

        print("进入示教模式（力拖动）...")
        print("提示：按 Ctrl+C 退出示教模式")
        self.teaching_mode = True
        
        # 初始化速度计算的历史值
        self._prev_q = None
        self._prev_time = None
        
        # 使用提供的参数或默认参数
        damping = damping_coeffs if damping_coeffs is not None else self.damping_params
        friction = friction_coeffs if friction_coeffs is not None else self.friction_params

        try:
            while self.teaching_mode:
                # 1. 获取当前状态
                q = self.get_joint_values()
                dq = self.get_joint_velocities()

                # 2. 计算重力补偿力矩
                tau_g = get_gravity_torque(q)

                # 3. 计算摩擦力补偿（库伦摩擦模型）
                tau_f = np.zeros(self.dof)
                if enable_friction:
                    # 库伦摩擦：tau_f = -friction_coeff * sign(dq)
                    # 当速度很小时，使用线性过渡避免抖动
                    velocity_threshold = 0.01  # rad/s
                    for i in range(self.dof):
                        if abs(dq[i]) > velocity_threshold:
                            tau_f[i] = -friction[i] * np.sign(dq[i])
                        else:
                            # 低速时线性过渡
                            tau_f[i] = -friction[i] * dq[i] / velocity_threshold

                # 4. 计算阻尼补偿（粘性摩擦）
                tau_d = np.zeros(self.dof)
                if enable_damping:
                    # 粘性阻尼：tau_d = -damping_coeff * dq
                    tau_d = -damping * dq

                # 5. 计算总补偿力矩
                # 目标力矩 = 重力矩 + 摩擦力矩 + 阻尼力矩
                target_tau = tau_g + tau_f + tau_d

                # 6. 限制力矩范围（安全保护）
                max_torque = 5.0  # N·m，根据实际电机限制调整
                target_tau = np.clip(target_tau, -max_torque, max_torque)

                print(target_tau)
                #7. 下发力矩指令
                for i in range(self.dof):
                    # 使用 MIT 模式下发，Kp, Kd 设为 0 即为纯力矩模式
                    param = MITParam(q=0, dq=0, kp=0, kd=0, tau=target_tau[i]*0.001)
                    self._arm.mit_control_one(i, param)

                time.sleep(0.005)  # 200Hz 控制频率
                
        except KeyboardInterrupt:
            print("\n退出示教模式...")
            self.stop_lead_through()
        except Exception as e:
            print(f"\n示教模式发生错误: {e}")
            self.stop_lead_through()
            raise
    
    def stop_lead_through(self):
        """
        停止示教模式，切换到位置控制模式并保持当前位置。
        """
        if not self.teaching_mode:
            return
        
        self.teaching_mode = False
        
        # 切换到位置控制模式
        print("切换到位置控制模式...")
        self.set_ctrl_mode_all(control_mode=2)  # POS_VEL 模式
        time.sleep(0.1)

        # # 获取当前位置
        # current_q = self.get_joint_values()
        # print(current_q)

        # # 保持当前位置
        # self.move_j(current_q, speed=0.1, block=False)
        
        # 清理速度计算的历史值
        self._prev_q = None
        self._prev_time = None
        
        print("示教模式已停止，机械臂保持在当前位置。")
    
    def calibrate_friction_params(self, 
                                  duration=30.0,
                                  joint_index=None,
                                  save_to_file=None,
                                  use_default_path=True):
        """
        测定摩擦力补偿参数。
        
        方法：让操作者缓慢拖动指定关节（或所有关节），记录不同速度下的力矩，
        通过线性回归拟合库伦摩擦系数。
        
        Args:
            duration: 测定持续时间（秒）
            joint_index: 要测定的关节索引，None表示测定所有关节
            save_to_file: 保存参数的文件路径，None表示使用默认路径
            use_default_path: 如果为True且save_to_file为None，使用默认路径保存
        
        Returns:
            np.ndarray: 测定的摩擦力系数数组
        """
        print("=" * 60)
        print("开始摩擦力参数测定")
        print("=" * 60)
        print(f"请缓慢拖动机械臂，持续 {duration} 秒")
        if joint_index is not None:
            print(f"重点测定关节 {joint_index}")
        print("提示：尽量覆盖正反两个方向，速度变化要缓慢均匀")
        print("按 Enter 开始，或 Ctrl+C 取消...")
        
        try:
            input()
        except KeyboardInterrupt:
            print("取消测定")
            return None
        
        # 切换到MIT模式
        current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
        if not all(mode == 1 for mode in current_modes if mode is not None):
            print("切换到 MIT 模式...")
            self.set_ctrl_mode_all(control_mode=1)
            time.sleep(0.1)
        
        # 初始化数据记录
        velocities = []
        torques = []
        positions = []
        
        # 初始化速度计算
        self._prev_q = None
        self._prev_time = None
        
        start_time = time.time()
        print("开始记录数据...")
        
        try:
            while time.time() - start_time < duration:
                q = self.get_joint_values()
                dq = self.get_joint_velocities()
                tau = self.get_joint_torques()
                
                # 只记录重力补偿后的剩余力矩（近似为摩擦力）
                tau_g = get_gravity_torque(q)
                tau_residual = tau - tau_g
                
                velocities.append(dq.copy())
                torques.append(tau_residual.copy())
                positions.append(q.copy())
                
                # 只进行重力补偿，不下发其他力矩
                for i in range(self.dof):
                    param = MITParam(q=0, dq=0, kp=0, kd=0, tau=tau_g[i])
                    self._arm.mit_control_one(i, param)
                
                time.sleep(0.01)  # 100Hz采样
                
                # 显示进度
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    print(f"进度: {elapsed:.1f}/{duration:.1f} 秒")
        
        except KeyboardInterrupt:
            print("\n测定被中断")
        
        # 转换为numpy数组
        velocities = np.array(velocities)
        torques = np.array(torques)
        
        print(f"\n共采集 {len(velocities)} 个数据点")
        
        # 拟合摩擦力参数
        friction_params = np.zeros(self.dof)
        joints_to_calibrate = [joint_index] if joint_index is not None else range(self.dof)
        
        for i in joints_to_calibrate:
            # 提取该关节的数据
            v = velocities[:, i]
            t = torques[:, i]
            
            # 过滤掉速度过小的点（噪声）
            mask = np.abs(v) > 0.01
            if np.sum(mask) < 10:
                print(f"关节 {i}: 有效数据点不足，使用默认值")
                continue
            
            v_filtered = v[mask]
            t_filtered = t[mask]
            
            # 使用符号函数拟合库伦摩擦
            # tau_f = -friction_coeff * sign(v)
            # 对于正速度，tau_f 应该为负；对于负速度，tau_f 应该为正
            positive_mask = v_filtered > 0
            negative_mask = v_filtered < 0
            
            if np.sum(positive_mask) > 5 and np.sum(negative_mask) > 5:
                # 分别计算正负方向的摩擦系数
                friction_pos = -np.mean(t_filtered[positive_mask])
                friction_neg = np.mean(t_filtered[negative_mask])
                friction_params[i] = (friction_pos + friction_neg) / 2.0
            else:
                # 数据不足，使用绝对值平均
                friction_params[i] = np.mean(np.abs(t_filtered))
            
            print(f"关节 {i}: 摩擦力系数 = {friction_params[i]:.4f} N·m")
        
        # 更新参数
        if joint_index is not None:
            self.friction_params[joint_index] = friction_params[joint_index]
        else:
            self.friction_params = friction_params
        
        # 保存到文件
        if save_to_file or use_default_path:
            if save_to_file is None and use_default_path:
                save_to_file = self._default_friction_file
            
            # 如果是相对路径，转换为绝对路径（相对于常量文件夹）
            if not os.path.isabs(save_to_file):
                save_to_file = os.path.join(self._constants_dir, save_to_file)
            
            np.save(save_to_file, friction_params)
            print(f"\n参数已保存到: {save_to_file}")
        
        print("=" * 60)
        print("摩擦力参数测定完成")
        print("=" * 60)
        
        return friction_params
    
    def calibrate_damping_params(self,
                                 duration=20.0,
                                 joint_index=None,
                                 save_to_file=None,
                                 use_default_path=True):
        """
        测定阻尼补偿参数。
        
        方法：让操作者快速拖动指定关节，记录速度-力矩关系，
        通过线性回归拟合粘性阻尼系数。
        
        Args:
            duration: 测定持续时间（秒）
            joint_index: 要测定的关节索引，None表示测定所有关节
            save_to_file: 保存参数的文件路径，None表示使用默认路径
            use_default_path: 如果为True且save_to_file为None，使用默认路径保存
        
        Returns:
            np.ndarray: 测定的阻尼系数数组
        """
        print("=" * 60)
        print("开始阻尼参数测定")
        print("=" * 60)
        print(f"请快速拖动机械臂，持续 {duration} 秒")
        print("提示：需要快速运动以产生明显的阻尼效果")
        print("按 Enter 开始，或 Ctrl+C 取消...")
        
        try:
            input()
        except KeyboardInterrupt:
            print("取消测定")
            return None
        
        # 切换到MIT模式
        current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
        if not all(mode == 1 for mode in current_modes if mode is not None):
            print("切换到 MIT 模式...")
            self.set_ctrl_mode_all(control_mode=1)
            time.sleep(0.1)
        
        # 初始化数据记录
        velocities = []
        torques = []
        positions = []
        
        # 初始化速度计算（重要：先获取一个初始位置）
        initial_q = self.get_joint_values()
        self._prev_q = initial_q.copy()
        self._prev_time = time.time()
        # 初始化历史队列
        self._q_history = [initial_q.copy()]
        self._t_history = [time.time()]
        
        start_time = time.time()
        print("开始记录数据...")
        print(f"初始关节位置: {initial_q}")
        
        try:
            # 使用更长的采样间隔来获得更准确的速度（对于慢速拖动）
            sample_interval = 0.02  # 50Hz采样，间隔更长有利于慢速运动的速度计算
            sample_count = 0
            while time.time() - start_time < duration:
                q = self.get_joint_values()
                # 对于慢速拖动，使用历史数据平滑速度计算
                dq = self.get_joint_velocities(use_history=True)
                tau = self.get_joint_torques()
                
                # 调试：每50个样本打印一次位置和速度（更频繁的调试信息）
                if sample_count % 50 == 0:
                    idx = joint_index if joint_index is not None else 0
                    print(f"样本 {sample_count}: 位置={q[idx]:.6f}, "
                          f"速度={dq[idx]:.6f}, "
                          f"位置变化={q[idx] - initial_q[idx]:.6f}")
                sample_count += 1
                
                # 计算重力补偿和摩擦力补偿后的剩余力矩（近似为阻尼）
                tau_g = get_gravity_torque(q)
                tau_f = -self.friction_params * np.sign(dq)
                tau_residual = tau - tau_g - tau_f
                
                velocities.append(dq.copy())
                torques.append(tau_residual.copy())
                positions.append(q.copy())
                
                # 只进行重力补偿和摩擦力补偿
                target_tau = tau_g + tau_f
                for i in range(self.dof):
                    param = MITParam(q=0, dq=0, kp=0, kd=0, tau=target_tau[i])
                    self._arm.mit_control_one(i, param)
                
                time.sleep(sample_interval)  # 50Hz采样，更适合慢速拖动
                
                # 显示进度
                elapsed = time.time() - start_time
                if int(elapsed) % 5 == 0 and elapsed > 0:
                    print(f"进度: {elapsed:.1f}/{duration:.1f} 秒")
        
        except KeyboardInterrupt:
            print("\n测定被中断")
        
        # 转换为numpy数组
        velocities = np.array(velocities)
        torques = np.array(torques)
        
        print(f"\n共采集 {len(velocities)} 个数据点")
        
        # 拟合阻尼参数
        damping_params = np.zeros(self.dof)
        joints_to_calibrate = [joint_index] if joint_index is not None else range(self.dof)
        
        for i in joints_to_calibrate:
            # 提取该关节的数据
            v = velocities[:, i]
            t = torques[:, i]
            
            # 统计速度分布信息
            v_abs = np.abs(v)
            v_max = np.max(v_abs)
            v_mean = np.mean(v_abs)
            v_std = np.std(v_abs)
            v_nonzero_count = np.sum(v_abs > 1e-6)  # 非零速度点数量
            
            # 统计位置变化（用于诊断）
            q_data = np.array(positions)[:, i]
            q_range = np.max(q_data) - np.min(q_data)
            q_std = np.std(q_data)
            
            print(f"\n关节 {i} 数据统计:")
            print(f"  位置范围: {q_range:.6f} rad (最小: {np.min(q_data):.6f}, 最大: {np.max(q_data):.6f})")
            print(f"  位置标准差: {q_std:.6f} rad")
            print(f"  最大速度: {v_max:.6f} rad/s")
            print(f"  平均速度: {v_mean:.6f} rad/s")
            print(f"  速度标准差: {v_std:.6f} rad/s")
            print(f"  非零速度点数: {v_nonzero_count}/{len(v)}")
            
            # 对于阻尼大的电机，使用更低的阈值
            # 尝试不同的速度阈值，从很低到高（适合慢速拖动）
            velocity_thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
            mask = None
            selected_threshold = None
            
            for threshold in velocity_thresholds:
                mask_candidate = np.abs(v) > threshold
                valid_count = np.sum(mask_candidate)
                if valid_count >= 10:
                    mask = mask_candidate
                    selected_threshold = threshold
                    break
            
            # 如果位置有变化但速度阈值都不满足，使用位置变化来检测运动
            if (mask is None or np.sum(mask) < 10) and q_range > 0.001:
                # 使用位置变化来检测运动点
                q_diff = np.abs(np.diff(q_data))
                q_motion_mask = np.concatenate([[False], q_diff > 1e-5])  # 位置变化大于阈值
                if np.sum(q_motion_mask) >= 10:
                    # 使用有位置变化的点，即使速度很小
                    mask = q_motion_mask
                    selected_threshold = 0.0  # 标记为使用位置变化检测
                    print(f"  检测到位置变化，使用位置变化检测运动（变化阈值: 1e-5 rad）")
                else:
                    # 如果位置变化也很小，降低阈值
                    q_motion_mask = np.concatenate([[False], q_diff > 1e-6])
                    if np.sum(q_motion_mask) >= 10:
                        mask = q_motion_mask
                        selected_threshold = 0.0
                        print(f"  使用更低的位置变化阈值: 1e-6 rad")
            
            if mask is None or np.sum(mask) < 10:
                print(f"关节 {i}: 有效数据点不足")
                print(f"  位置变化范围: {q_range:.6f} rad")
                print(f"  最大速度: {v_max:.6f} rad/s")
                if q_range < 0.001:
                    print(f"  建议：关节 {i} 位置几乎没有变化，请确保在测定过程中拖动关节")
                else:
                    print(f"  建议：虽然位置有变化，但速度太小，可能需要更慢的拖动或更长的采样间隔")
                damping_params[i] = 2.0  # 使用默认值
                continue
            
            v_filtered = v[mask]
            t_filtered = t[mask]
            
            print(f"  使用速度阈值: {selected_threshold:.3f} rad/s")
            print(f"  有效数据点数: {len(v_filtered)}")
            print(f"  有效速度范围: [{np.min(np.abs(v_filtered)):.4f}, {np.max(np.abs(v_filtered)):.4f}] rad/s")
            
            # 线性回归：tau_d = -damping * v
            # 使用最小二乘法拟合
            if len(v_filtered) > 1:
                # tau = -damping * v，所以 damping = -tau / v
                # 使用最小二乘：damping = -mean(tau * v) / mean(v^2)
                damping_params[i] = -np.mean(t_filtered * v_filtered) / np.mean(v_filtered ** 2)
                # 确保阻尼系数为正
                damping_params[i] = max(0.0, damping_params[i])
                
                # 计算拟合质量（相关系数）
                correlation = np.corrcoef(v_filtered, t_filtered)[0, 1]
                print(f"  速度-力矩相关系数: {correlation:.4f}")
            else:
                damping_params[i] = 2.0  # 默认值
            
            print(f"关节 {i}: 阻尼系数 = {damping_params[i]:.4f} N·m·s/rad")
        
        # 更新参数
        if joint_index is not None:
            self.damping_params[joint_index] = damping_params[joint_index]
        else:
            self.damping_params = damping_params
        
        # 保存到文件
        if save_to_file or use_default_path:
            if save_to_file is None and use_default_path:
                save_to_file = self._default_damping_file
            
            # 如果是相对路径，转换为绝对路径（相对于常量文件夹）
            if not os.path.isabs(save_to_file):
                save_to_file = os.path.join(self._constants_dir, save_to_file)
            
            np.save(save_to_file, damping_params)
            print(f"\n参数已保存到: {save_to_file}")
        
        print("=" * 60)
        print("阻尼参数测定完成")
        print("=" * 60)
        
        return damping_params
    
    def verify_gravity_compensation(self, 
                                    test_positions=None,
                                    tolerance=0.5):
        """
        验证重力补偿的准确性。
        
        方法：在多个测试位置保持静止，测量实际力矩与理论重力矩的差异。
        
        Args:
            test_positions: 测试位置列表，每个位置是一个关节角度数组。
                           如果为None，使用几个典型位置
            tolerance: 力矩误差容忍度（N·m）
        
        Returns:
            dict: 包含验证结果的字典
        """
        print("=" * 60)
        print("开始重力补偿验证")
        print("=" * 60)
        
        if test_positions is None:
            # 使用几个典型位置
            test_positions = [
                np.zeros(self.dof),  # 零位
                np.array([0.5, 0.3, -0.5, 1.0, 0.0, 0.0, 0.0]),  # 伸展位置
                np.array([-0.5, 0.5, -1.0, 1.5, -0.5, 0.2, -0.3]),  # 弯曲位置
            ]
        
        results = {
            'positions': [],
            'theoretical_torques': [],
            'measured_torques': [],
            'errors': [],
            'max_error': 0.0,
            'mean_error': 0.0,
            'passed': True
        }
        
        for idx, target_q in enumerate(test_positions):
            print(f"\n测试位置 {idx + 1}/{len(test_positions)}")
            print(f"目标关节角度: {target_q}")
            
            # 移动到目标位置
            self.move_j(target_q, speed=0.1, block=True, tolerance=0.05)
            time.sleep(0.5)  # 等待稳定
            
            # 切换到MIT模式，只进行重力补偿
            current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
            if not all(mode == 1 for mode in current_modes if mode is not None):
                self.set_ctrl_mode_all(control_mode=1)
                time.sleep(0.1)
            
            # 获取实际位置和力矩
            actual_q = self.get_joint_values()
            measured_tau = self.get_joint_torques()
            
            # 计算理论重力矩
            theoretical_tau = get_gravity_torque(actual_q)
            
            # 计算误差
            error = measured_tau - theoretical_tau
            
            # 记录结果
            results['positions'].append(actual_q.copy())
            results['theoretical_torques'].append(theoretical_tau.copy())
            results['measured_torques'].append(measured_tau.copy())
            results['errors'].append(error.copy())
            
            # 显示结果
            print(f"理论重力矩: {theoretical_tau}")
            print(f"实际测量力矩: {measured_tau}")
            print(f"误差: {error}")
            print(f"最大误差: {np.max(np.abs(error)):.4f} N·m")
            
            # 应用重力补偿
            for i in range(self.dof):
                param = MITParam(q=0, dq=0, kp=0, kd=0, tau=theoretical_tau[i])
                self._arm.mit_control_one(i, param)
            
            time.sleep(2.0)  # 观察效果
        
        # 计算统计结果
        all_errors = np.concatenate(results['errors'])
        results['max_error'] = np.max(np.abs(all_errors))
        results['mean_error'] = np.mean(np.abs(all_errors))
        results['passed'] = results['max_error'] < tolerance
        
        print("\n" + "=" * 60)
        print("重力补偿验证结果")
        print("=" * 60)
        print(f"最大误差: {results['max_error']:.4f} N·m")
        print(f"平均误差: {results['mean_error']:.4f} N·m")
        print(f"容忍度: {tolerance} N·m")
        print(f"验证结果: {'通过' if results['passed'] else '未通过'}")
        print("=" * 60)
        
        return results
    
    def load_compensation_params(self, friction_file=None, damping_file=None, use_default_path=True):
        """
        从文件加载补偿参数。
        
        Args:
            friction_file: 摩擦力参数文件路径，None表示使用默认路径
            damping_file: 阻尼参数文件路径，None表示使用默认路径
            use_default_path: 如果为True且文件路径为None，使用默认路径加载
        """
        # 处理摩擦力参数文件
        if friction_file is None and use_default_path:
            friction_file = self._default_friction_file
        
        if friction_file:
            # 如果是相对路径，转换为绝对路径（相对于常量文件夹）
            if not os.path.isabs(friction_file):
                friction_file = os.path.join(self._constants_dir, friction_file)
            
            if os.path.exists(friction_file):
                self.friction_params = np.load(friction_file)
                print(f"已加载摩擦力参数: {friction_file}")
            elif use_default_path:
                print(f"摩擦力参数文件不存在: {friction_file}，使用默认值")
        
        # 处理阻尼参数文件
        if damping_file is None and use_default_path:
            damping_file = self._default_damping_file
        
        if damping_file:
            # 如果是相对路径，转换为绝对路径（相对于常量文件夹）
            if not os.path.isabs(damping_file):
                damping_file = os.path.join(self._constants_dir, damping_file)
            
            if os.path.exists(damping_file):
                self.damping_params = np.load(damping_file)
                print(f"已加载阻尼参数: {damping_file}")
            elif use_default_path:
                print(f"阻尼参数文件不存在: {damping_file}，使用默认值")

    def close_connection(self):
        """
        关闭与机械臂的连接，释放资源。
        """
        try:
            # 禁用所有电机
            self.disable()
            print("All motors have been disabled.")

            # 关闭底层硬件连接
            self._openarm.close()
            print("Connection to OpenArm has been closed.")
        except Exception as e:
            print(f"Error occurred while closing connection: {e}")

if __name__ == "__main__":
    try:
        arm = OpenArmController(
            can_name="vcan0",
            auto_enable=False
        )

        # 必须：整臂上电（Python 只能这样）
        arm.enable()

        print("DOF:", arm.dof)

        #测试单个电机
        # for i in range(3000):

        #     arm.move_single_motor(
        #         index=0,
        #         position=2.80212863,
        #         vel= 0.2
    
        #     )
        #     time.sleep(0.01)

        #     print(arm.get_joint_values())

        #测试多个电机    
        #joints_values = [0,0,0,0,0,0,0] #初位
        #arm.move_j(joint_angles=joints_values,speed=0.1,block=True,debug=False)        
        #[ 1.46353094, 1.16369116, -1.56462196, 2.14141299, 0.07228962, -0.00820172, 1.51884489]
        # joints_values = [ 1.46353094, 1.16369116, -1.56462196, 2.14141299, 0.07228962, -0.00820172, 1.51884489]
        # arm.move_j(joint_angles=joints_values,speed=0.1,block=True,debug=False)  
        
        #测试move_jntspace_path
        # current_joints = arm.get_joint_values()
        # joints_values = [1.44792246,  0.5188,     -1.57076153,  1.04821431,  0.10617751, -0.06196755, -0.52610891]
        # joints_values_path = [current_joints, joints_values]
        # arm.move_jntspace_path(path= joints_values_path,speed= 0.1)

        #测试夹爪
        #arm.move_gripper_motor(position=0.1)
        #arm.open_gripper()
        #arm.close_gripper()
        print(arm.is_enabled)
        # print(arm.get_gripper_status())
        #-0.10242618  1.78130007 -1.59094377  2.45727474  0.01926452 -0.02765698 1.1381323
        #print(arm.get_joint_values())
        # print(arm.get_joint_torques())

        #获取电机参数
        # param = arm.get_all_motor_params(rid=10, timeout_us=2000)
        # print(param)

        #设置电机控制模式
        #arm.set_ctrl_mode_all(control_mode=2)

        #设置零点
        # arm._openarm.set_zero_all()

        #测试get_pose() 获取位姿
        # pos, rot = arm.get_pose()
        # print("End-Effector Position:", pos)
        # print("End-Effector Rotation Matrix:\n", rot)

        #测试move_p()
        # target_pos = pos + np.array([0.05, 0.0, 0.0])  # 在 x 方向移动 5 cm
        # target_rot = rot  # 保持当前姿态不变
        # arm.move_p(pos=target_pos, rotmat=target_rot, block=True, speed=0.1)

        #get_gravity_torque() 获取重力力矩   
        #    
        # 使用默认路径保存（会自动保存到 constant 文件夹）
        #arm.calibrate_damping_params()       
        #print(arm.get_gravity_torque(q=[0,0,0,0,0,0,0]))
        #arm.disable()
        # while True:
        #     time.sleep(1)

        # print(arm.get_joint_torques())
        # tau = arm.get_joint_torques()
        #[-0.03956044 -0.01318681 -0.03418803  0.1025641   0.002442    0.002442 -0.002442  ]
        # arm.move_m(target_tau=tau)
        #print(arm.get_joint_values())
        # print(arm.friction_params)
        
        # print(arm.damping_params)

        # arm.start_lead_through(enable_damping=True, enable_friction=True)


    except KeyboardInterrupt:
        arm.disable()

        print(arm.get_joint_values())






    
  