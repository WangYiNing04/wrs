import logging
import math
import struct
import threading
import time
import os

from joyconrobotics import JoyconRobotics
import numpy as np
from wrs.robot_sim.manipulators.piper.piper import Piper
from wrs.robot_con.piper.piper import PiperArmController
import wrs.basis.robot_math as rm

np.set_printoptions(linewidth=200)

JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]

class JoyConController:
    def __init__(
        self,
        name,
        initial_position=None,
        timeout = 60,
        *args,
        **kwargs,
    ):
        self.name = name
        
       
        self.robot_sim = Piper
        self.glimit = [[-2.618, 0.0, -2.697, -1.832, -1.22, -2.094], 
                       [2.618,  3.14,  0.0,  1.832,  1.22,  2.094]]
        
        self.init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.target_qpos = self.init_qpos.copy() 
        
        self.target_gpos = self.init_gpos.copy() 
        
       
        self.max_retries = timeout
        self.retries = 0
        self.offset_position_m = self.init_gpos[0:3]

        while self.retries < self.max_retries:
            try:
                self.joyconrobotics = JoyconRobotics(
                    device=name, 
                    horizontal_stick_mode='yaw_diff', 
                    close_y=True, 
                    limit_dof=True, 
                    offset_position_m=self.offset_position_m, 
                    glimit=self.glimit,
                    dof_speed=[1.0, 1.0, 1.0, 1.0, 1.0, 0.5], 
                    common_rad=False,
                    lerobot=True,
                    pitch_down_double=True
                )
                break  # 连接成功，跳出循环
            except Exception as e:
                self.retries += 1
                print(f"Failed to connect to {name} Joycon: {e}")
                print(f"Retrying ({self.retries}/{self.max_retries}) in 1 second...")
                time.sleep(1)  # 连接失败后等待 1 秒重试
        else:
            print("Failed to connect after several attempts.")
        
        self.target_gpos_last = self.init_gpos.copy() 
        self.joint_angles_last = self.init_qpos.copy() 
        
    def get_command(self, present_pose):
        
        target_pose, gripper_state, button_control = self.joyconrobotics.get_control()
        # print("target_pose:", [f"{x:.3f}" for x in target_pose])
        
        for i in range(6):
            if target_pose[i] < self.glimit[0][i]:
                target_pose[i] = self.glimit[0][i]  
            elif target_pose[i] > self.glimit[1][i]:
                target_pose[i] = self.glimit[1][i]  
            else:
                 target_pose[i]
                 
        x = target_pose[0] # init_gpos[0] + 
        z = target_pose[2] # init_gpos[2] + 
        _, _, _, roll, pitch, yaw = target_pose
        y = 0.01
        pitch = -pitch 
        roll = roll - math.pi/2 # lerobo末端旋转90度
        
        # 双臂朝中间偏
        # if self.name == 'left':
        #     yaw = yaw - 0.4
        # elif self.name == 'right':
        #     yaw = yaw + 0.4
        
        target_gpos = np.array([x, y, z, roll, pitch, 0.0])
        try:
            jnts_values = self.robot_sim.ik(tgt_pos=[x, y, z],tgt_rotmat=rm.rotmat_from_euler([roll, pitch, 0]))
        except Exception as e:
            print("IK计算出错:", e)
            jnts_values = None

        if jnts_values is None:
            IK_success = False
            print("⚠️ IK 无解，保持上次位置")
        else:
            IK_success = True

        if IK_success:
            self.target_qpos = np.concatenate(([yaw,], jnts_values[:5], [gripper_state,])) 

            
            self.target_gpos_last = target_gpos.copy() 
            joint_angles = self.target_qpos
            
            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[1] = -joint_angles[1]
            joint_angles[0] = -joint_angles[0]
            joint_angles[4] = -joint_angles[4]
            self.joint_angles_last = joint_angles.copy() 

        else:
            self.target_gpos = self.target_gpos_last.copy()
            self.joyconrobotics.set_position(self.target_gpos[0:3])
            joint_angles = self.joint_angles_last
        
        # if button_control != 0:
        #     self.joyconrobotics.reset_joycon()
            
        return joint_angles, button_control