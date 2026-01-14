#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立仿真窗口进程 - 实时预览OpenArm机械臂"""
import sys
import os
import numpy as np
import json
import time

# 添加项目根目录到路径
project_root = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.manipulators.openarm.openarm as sim_openarm
    import wrs.modeling.geometric_model as mgm
    
    # 预设文件路径（用于读取最新关节状态）
    presets_file = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/openarm/constant/joint_presets.json"
    joint_state_file = os.path.join(os.path.dirname(presets_file), "current_joints.json")
    
    # 创建仿真世界
    base = wd.World(
        cam_pos=np.array([1.5, -1.5, 1.0]),
        lookat_pos=np.array([0, 0, 0.3]),
        auto_rotate=False,
        w=800,
        h=600
    )
    
    # 添加坐标系
    mgm.gen_frame(ax_length=0.2).attach_to(base)
    
    # 创建仿真机械臂
    sim_arm = sim_openarm.OpenArm(load_meshes=True)
    current_joints = np.zeros(7)
    sim_arm.goto_given_conf(current_joints)
    sim_arm_meshmodel = sim_arm.gen_meshmodel(toggle_tcp_frame=True)
    sim_arm_meshmodel.attach_to(base)
    
    # 更新函数 - 从文件读取最新关节角度
    def update_simulation(task):
        try:
            # 尝试从文件读取最新关节角度
            if os.path.exists(joint_state_file):
                with open(joint_state_file, 'r') as f:
                    data = json.load(f)
                    if 'joints' in data:
                        new_joints = np.array(data['joints'])
                        if not np.array_equal(new_joints, current_joints):
                            # 更新机械臂姿态
                            nonlocal current_joints, sim_arm_meshmodel
                            current_joints = new_joints
                            sim_arm_meshmodel.detach()
                            sim_arm.goto_given_conf(current_joints)
                            sim_arm_meshmodel = sim_arm.gen_meshmodel(toggle_tcp_frame=True)
                            sim_arm_meshmodel.attach_to(base)
        except Exception as e:
            pass  # 忽略读取错误
        return task.cont
    
    base.taskMgr.add(update_simulation, "update_simulation", appendTask=True)
    
    print("仿真窗口已启动")
    print("提示：GUI会通过文件同步关节角度")
    base.run()
except Exception as e:
    print(f"仿真窗口启动失败: {e}")
    import traceback
    traceback.print_exc()
    input("按Enter键退出...")
