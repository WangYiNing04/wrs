'''
Author: wang yining
Date: 2025-12-31 00:43:04
LastEditTime: 2026-01-04 15:56:13
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

#import ik_solver
from wrs.robot_sim.manipulators.openarm.openarm import OpenArm

#run setup_vcan.sh firstly
#use canfd slcan

#计算动力学矩阵
try:
    import pinocchio as pin

    # 加载模型
    model = pin.buildModelFromUrdf("/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_sim/robots/openarm/openarm.urdf")
    data = model.createData()

    def get_gravity_torque(q):
        # q 是当前的关节弧度
        return pin.computeGeneralizedGravityForces(model, data, q)
    
except ImportError:
    print("Pinocchio 未安装，重力补偿功能不可用。")
    def get_gravity_torque(q):
        return np.zeros(len(q))
except Exception as e:
    print(f"加载 Pinocchio 模型时出错：{e}")
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
                 force_vcan_setup=False):

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

        self._gripper_limits = np.array([-0.95426108, 0.00019074])

        if auto_enable:
            self.enable()

        #ik_solver
        self.arm_sim = OpenArm()

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
        return all(m.is_enabled() for m in self._arm.get_motors())


    def enable(self):
        self._openarm.enable_all()
        time.sleep(0.2)
        print("OpenArm motors enabled.")


    def disable(self):
        self._openarm.disable_all()
        print("OpenArm motors disabled.")


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

    #TODO
    def move_m(self,
            joint_angles,
            *,
            vel_ref=0.0,
            block=False,
            tolerance=0.01):

        angles = np.asarray(joint_angles, dtype=float)
        if angles.size != self.dof:
            raise ValueError("Invalid joint dimension")

        params = [
            PosVelParam(q, vel_ref)
            for q in angles
        ]

        self._arm.posvel_control_all(params)


        if block:
            self._wait_until_joint_reached(angles, tolerance)

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
    
    #TODO
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
            speed: float= 0.1,
            block=False):
        current_joints = self.get_joint_values()
        joints_values = self.arm_sim.ik(tgt_pos=pos, tgt_rotmat=rotmat, seed_jnt_values=current_joints)
        print(f"求解出的关节角度:{joints_values}")

        if joints_values is None:
            print("无法求解该位置的逆解，动作被忽略。")
            return
        
        self.move_j(joints_values, block=block, speed=speed)

    #move_l(...)

    # ---- Feedback ----
    def get_joint_values(self) -> np.ndarray:
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_position() for m in self._arm.get_motors()])

    def get_pose(self):
        joints_values = self.get_joint_values()
        pos, rot = self.arm_sim.fk(joints_values)
        return pos, rot


    def get_joint_torques(self):
        self._openarm.refresh_all()
        self._openarm.recv_all()
        return np.array([m.get_torque() for m in self._arm.get_motors()])
    

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

    #TODO
    def gripper_control(self, pos: float, vel: float = 0.2):
        """
        夹爪位置控制（归一化输入）
        
        pos ∈ [0, 1]
        0.0 -> open
        1.0 -> close
        """
        if not self._has_gripper:
            return

        pos = float(np.clip(pos, 0.0, 1.0))

        open_pos, close_pos = self._gripper_limits

        # 线性插值
        target = open_pos + pos * (close_pos - open_pos)

        param = PosVelParam(target, vel)
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
    def get_all_motor_params(self, rid, timeout_us=2000):
        """
        function:
            修改rid查询不同参数
        
        查询所有臂部电机的特定参数（非实时状态）。
        
        :param rid: MotorVariable 枚举值 (例如 MotorVariable.PMAX, MotorVariable.KP_ASR)
        :param timeout_us: 等待 CAN 帧回传的超时时间（微秒）
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
                if res.valid:
                    params.append(res.value)
                else:
                    params.append(None) 
            except Exception as e:
                print("返回浮点数")
                params.append(res)
                
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
        
        self._openarm.set_callback_mode_all(1)
        self._openarm.set_ctrl_mode_all(control_mode)

    #TODO
    def start_lead_through(self):
        """
        开启示教模式，切换到 MIT 模式或力矩模式，并实时下发力矩指令。
        """
        from openarm_can import MotorVariable

        # 检查当前模式是否为 MIT 模式
        current_modes = self.get_all_motor_params(rid=10, timeout_us=2000)
        if not all(mode == 1 for mode in current_modes):  # 1 表示 MIT 模式
            print("切换到 MIT 模式...")
            self.set_ctrl_mode_all(control_mode=1)
            
            return

        print("进入示教模式...")
        self.teaching_mode = True

        try:
            while self.teaching_mode:
                # 1. 获取当前状态
                q = self.get_joint_values()
                dq = self.get_joint_velocities()  # 需要获取速度

                # 2. 计算补偿力矩
                tau_g = get_gravity_torque(q)

                # 3. 摩擦力补偿 (简单示例)
                tau_f = 0.5 * np.sign(dq)  # 补偿常数需调试

                # 4. 下发指令
                # 目标力矩 = 重力矩 + 摩擦力矩
                target_tau = tau_g + tau_f

                for i in range(self.dof):
                    # 使用 MIT 模式下发，Kp, Kd 设为 0 即为纯力矩模式
                    param = MITParam(q=0, dq=0, kp=0, kd=0, tau=target_tau[i])
                    self._arm.mit_control_one(i, param)

                time.sleep(0.005)  # 200Hz
        except KeyboardInterrupt:
            print("退出示教模式...")
            self.teaching_mode = False

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

        joints_values = [-0.13561456,  0.50946059, -1.53067063,  2.45727474, -1.57263294,  0.1489662 ,-1.45437552]
        arm.move_j(joint_angles=joints_values,speed=0.1,block=True,debug=False)  
        
        #测试move_jntspace_path
        # current_joints = arm.get_joint_values()
        # joints_values = [1.44792246,  0.5188,     -1.57076153,  1.04821431,  0.10617751, -0.06196755, -0.52610891]
        # joints_values_path = [current_joints, joints_values]
        # arm.move_jntspace_path(path= joints_values_path,speed= 0.1)

        #测试夹爪
        #arm.move_gripper_motor(position=0.1)
        #arm.open_gripper()
        #arm.close_gripper()
        # print(arm.get_gripper_status())
        #-0.10242618  1.78130007 -1.59094377  2.45727474  0.01926452 -0.02765698 1.1381323
        print(arm.get_joint_values())
        # print(arm.get_joint_torques())

        #获取电机参数
        # param = arm.get_all_motor_params(rid=10, timeout_us=2000)
        # print(param)

        #设置电机控制模式
        #arm.set_ctrl_mode_all(control_mode=1)

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
                                     
        #arm.disable()
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        arm.disable()

        print(arm.get_joint_values())






    
  