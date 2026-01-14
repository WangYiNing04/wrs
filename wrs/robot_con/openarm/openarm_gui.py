#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenArm 商业级图形界面控制器
Author: AI Assistant
Date: 2025
Description: 提供完整的机械臂控制界面，包括关节控制、笛卡尔控制、夹爪控制等功能
"""

import sys
import time
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict
import traceback
import logging
import json

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                                 QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit,
                                 QTabWidget, QGroupBox, QMessageBox, QStatusBar,
                                 QProgressBar, QCheckBox, QSlider, QFrame,
                                 QListWidget, QListWidgetItem, QInputDialog)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QMutex
    from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False
    print("警告: PyQt5 未安装，GUI 功能不可用。请运行: pip install PyQt5")

# 导入 OpenArmController
import os
import sys
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

try:
    from openarm import OpenArmController
except ImportError:
    try:
        from .openarm import OpenArmController
    except ImportError:
        raise ImportError("无法导入 OpenArmController，请确保 openarm.py 在同一目录下")

# 导入仿真相关模块
try:
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.manipulators.openarm.openarm as sim_openarm
    import wrs.modeling.geometric_model as mgm
    SIMULATION_AVAILABLE = True
except ImportError as e:
    SIMULATION_AVAILABLE = False
    print(f"警告: 仿真功能不可用: {e}")


class StatusMonitorThread(QThread):
    """状态监控线程，用于实时更新机械臂状态"""
    status_update = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    sim_update = pyqtSignal(list)  # 发送关节角度到仿真（使用list类型避免numpy数组类型注册问题）
    
    def __init__(self, controller: OpenArmController, update_rate: float = 0.1):
        super().__init__()
        self.controller = controller
        self.update_rate = update_rate
        self.running = False
        self.mutex = QMutex()
        self.enable_sim_update = False  # 是否启用仿真更新
        
    def run(self):
        self.running = True
        while self.running:
            try:
                self.mutex.lock()
                if self.controller is None:
                    break
                    
                # 获取关节角度
                joint_values = self.controller.get_joint_values()
                
                # 获取关节力矩
                joint_torques = self.controller.get_joint_torques()
                
                # 获取关节速度
                try:
                    joint_velocities = self.controller.get_joint_velocities()
                except:
                    joint_velocities = None
                
                # 获取末端位姿
                try:
                    pos, rot = self.controller.get_pose()
                except:
                    pos, rot = None, None
                
                # 获取夹爪状态
                try:
                    gripper_status = self.controller.get_gripper_status()
                except:
                    gripper_status = None
                
                # 获取使能状态
                is_enabled = self.controller.is_enabled
                
                status = {
                    'joint_values': joint_values,
                    'joint_torques': joint_torques,
                    'joint_velocities': joint_velocities,
                    'position': pos,
                    'rotation': rot,
                    'gripper_status': gripper_status,
                    'is_enabled': is_enabled,
                    'timestamp': datetime.now()
                }
                
                self.status_update.emit(status)
                
                # 如果启用仿真更新，发送关节角度（转换为list避免类型问题）
                if self.enable_sim_update:
                    self.sim_update.emit(joint_values.tolist())
                
                self.mutex.unlock()
                
            except Exception as e:
                self.mutex.unlock()
                self.error_occurred.emit(f"状态更新错误: {str(e)}")
            
            time.sleep(self.update_rate)
    
    def stop(self):
        self.running = False
        self.wait()


class OpenArmGUI(QMainWindow):
    """OpenArm 商业级图形界面控制器"""
    
    def __init__(self):
        super().__init__()
        if not PYQT5_AVAILABLE:
            QMessageBox.critical(None, "错误", "PyQt5 未安装，无法启动 GUI。\n请运行: pip install PyQt5")
            sys.exit(1)
            
        self.controller: Optional[OpenArmController] = None
        self.monitor_thread: Optional[StatusMonitorThread] = None
        self.is_connected = False
        
        # 仿真相关
        self.sim_world: Optional[wd.World] = None
        self.sim_arm: Optional[sim_openarm.OpenArm] = None
        self.sim_enabled = False
        self.sim_arm_meshmodel = None  # 用于实时更新的mesh模型
        self.sim_update_joints = None  # 用于传递关节角度的变量
        self.sim_process = None  # 仿真窗口进程
        self.sim_running = False  # 仿真窗口运行标志
        self.sim_lock = QMutex()  # 仿真更新锁
        
        # 预设关节组相关
        self._constants_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constant")
        os.makedirs(self._constants_dir, exist_ok=True)
        self._presets_file = os.path.join(self._constants_dir, "joint_presets.json")
        self.joint_presets: Dict[str, List[float]] = {}
        
        # 设置日志
        self.setup_logging()
        
        # 初始化 UI
        self.init_ui()
        
        # 设置样式
        self.setup_styles()
        
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'openarm_gui_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("OpenArm 机械臂控制器 - FAFU-Robot")
        self.setGeometry(100, 100, 1600, 1000)  # 增大窗口以容纳仿真
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右侧状态面板
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪 - 请连接机械臂")
        
        # 创建菜单栏
        self.create_menu_bar()
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        connect_action = file_menu.addAction('连接机械臂(&C)')
        connect_action.setShortcut('Ctrl+C')
        connect_action.triggered.connect(self.connect_robot)
        
        disconnect_action = file_menu.addAction('断开连接(&D)')
        disconnect_action.setShortcut('Ctrl+D')
        disconnect_action.triggered.connect(self.disconnect_robot)
        
        file_menu.addSeparator()
        
        exit_action = file_menu.addAction('退出(&X)')
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        
        # 控制菜单
        control_menu = menubar.addMenu('控制(&T)')
        
        enable_action = control_menu.addAction('使能所有电机(&E)')
        enable_action.setShortcut('Ctrl+E')
        enable_action.triggered.connect(self.enable_robot)
        
        disable_action = control_menu.addAction('禁用所有电机(&D)')
        disable_action.setShortcut('Ctrl+Shift+D')
        disable_action.triggered.connect(self.disable_robot)
        
        emergency_action = control_menu.addAction('紧急停止(&S)')
        emergency_action.setShortcut('Escape')
        emergency_action.triggered.connect(self.emergency_stop)
        
        control_menu.addSeparator()
        
        sim_action = control_menu.addAction('打开仿真窗口(&S)')
        sim_action.setShortcut('Ctrl+S')
        sim_action.triggered.connect(self.toggle_simulation)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        about_action = help_menu.addAction('关于(&A)')
        about_action.triggered.connect(self.show_about)
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 连接控制组
        connection_group = QGroupBox("连接控制")
        connection_layout = QVBoxLayout()
        
        self.can_name_input = QLineEdit("vcan0")
        self.can_name_input.setPlaceholderText("CAN 接口名称 (如: vcan0, can0)")
        connection_layout.addWidget(QLabel("CAN 接口:"))
        connection_layout.addWidget(self.can_name_input)
        
        btn_layout = QHBoxLayout()
        self.connect_btn = QPushButton("连接机械臂")
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.connect_btn.clicked.connect(self.connect_robot)
        btn_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("断开连接")
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.disconnect_btn.clicked.connect(self.disconnect_robot)
        btn_layout.addWidget(self.disconnect_btn)
        
        connection_layout.addLayout(btn_layout)
        connection_group.setLayout(connection_layout)
        layout.addWidget(connection_group)
        
        # 安全控制组
        safety_group = QGroupBox("安全控制")
        safety_layout = QVBoxLayout()
        
        btn_layout2 = QHBoxLayout()
        self.enable_btn = QPushButton("使能所有电机")
        self.enable_btn.setEnabled(False)
        self.enable_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.enable_btn.clicked.connect(self.enable_robot)
        btn_layout2.addWidget(self.enable_btn)
        
        self.disable_btn = QPushButton("禁用所有电机")
        self.disable_btn.setEnabled(False)
        self.disable_btn.setStyleSheet("background-color: #FF9800; color: white;")
        self.disable_btn.clicked.connect(self.disable_robot)
        btn_layout2.addWidget(self.disable_btn)
        
        safety_layout.addLayout(btn_layout2)
        
        self.emergency_btn = QPushButton("紧急停止 (ESC)")
        self.emergency_btn.setEnabled(False)
        self.emergency_btn.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; font-weight: bold; padding: 10px;")
        self.emergency_btn.clicked.connect(self.emergency_stop)
        safety_layout.addWidget(self.emergency_btn)
        
        safety_group.setLayout(safety_layout)
        layout.addWidget(safety_group)
        
        # 创建标签页
        tabs = QTabWidget()
        
        # 关节控制标签页
        joint_tab = self.create_joint_control_tab()
        tabs.addTab(joint_tab, "关节控制")
        
        # 笛卡尔控制标签页
        cartesian_tab = self.create_cartesian_control_tab()
        tabs.addTab(cartesian_tab, "笛卡尔控制")
        
        # 夹爪控制标签页
        gripper_tab = self.create_gripper_control_tab()
        tabs.addTab(gripper_tab, "夹爪控制")
        
        # 单电机控制标签页（调试用）
        single_motor_tab = self.create_single_motor_tab()
        tabs.addTab(single_motor_tab, "单电机控制")
        
        # 预设关节组标签页
        presets_tab = self.create_presets_tab()
        tabs.addTab(presets_tab, "预设关节组")
        
        layout.addWidget(tabs)
        
        # 加载预设关节组
        self.load_joint_presets()
        
        layout.addStretch()
        return panel
    
    def create_joint_control_tab(self):
        """创建关节控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 速度设置
        speed_group = QGroupBox("运动参数")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.joint_speed_spin = QDoubleSpinBox()
        self.joint_speed_spin.setRange(0.01, 1.0)
        self.joint_speed_spin.setValue(0.3)
        self.joint_speed_spin.setSingleStep(0.05)
        self.joint_speed_spin.setDecimals(2)
        speed_layout.addWidget(self.joint_speed_spin)
        
        self.block_checkbox = QCheckBox("阻塞执行")
        self.block_checkbox.setChecked(False)
        speed_layout.addWidget(self.block_checkbox)
        
        self.sim_preview_checkbox = QCheckBox("仿真预览")
        self.sim_preview_checkbox.setChecked(False)
        self.sim_preview_checkbox.setEnabled(SIMULATION_AVAILABLE)
        if not SIMULATION_AVAILABLE:
            self.sim_preview_checkbox.setToolTip("仿真功能不可用")
        speed_layout.addWidget(self.sim_preview_checkbox)
        
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # 关节角度输入
        joint_group = QGroupBox("关节角度 (弧度)")
        joint_layout = QGridLayout()
        
        self.joint_spins = []
        joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        joint_limits = [
            [-1.396263, 3.490659],
            [-1.745329, 1.745329],
            [-1.570796, 1.570796],
            [0.0, 2.443461],
            [-1.570796, 1.570796],
            [-0.785398, 0.785398],
            [-1.570796, 1.570796]
        ]
        
        for i, (label, limits) in enumerate(zip(joint_labels, joint_limits)):
            joint_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(limits[0], limits[1])
            spin.setValue(0.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.1)
            self.joint_spins.append(spin)
            joint_layout.addWidget(spin, i, 1)
        
        joint_group.setLayout(joint_layout)
        layout.addWidget(joint_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.read_all_joints_btn = QPushButton("读取所有关节")
        self.read_all_joints_btn.setEnabled(False)
        self.read_all_joints_btn.clicked.connect(self.read_all_joint_values)
        btn_layout.addWidget(self.read_all_joints_btn)
        
        self.move_joint_btn = QPushButton("移动到目标位置")
        self.move_joint_btn.setEnabled(False)
        self.move_joint_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.move_joint_btn.clicked.connect(self.move_joint)
        btn_layout.addWidget(self.move_joint_btn)
        
        self.home_btn = QPushButton("回零位")
        self.home_btn.setEnabled(False)
        self.home_btn.clicked.connect(self.move_to_home)
        btn_layout.addWidget(self.home_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        return widget
    
    def create_cartesian_control_tab(self):
        """创建笛卡尔控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 速度设置
        speed_group = QGroupBox("运动参数")
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("速度:"))
        self.cartesian_speed_spin = QDoubleSpinBox()
        self.cartesian_speed_spin.setRange(0.01, 1.0)
        self.cartesian_speed_spin.setValue(0.1)
        self.cartesian_speed_spin.setSingleStep(0.05)
        self.cartesian_speed_spin.setDecimals(2)
        speed_layout.addWidget(self.cartesian_speed_spin)
        
        self.cartesian_sim_preview_checkbox = QCheckBox("仿真预览")
        self.cartesian_sim_preview_checkbox.setChecked(False)
        self.cartesian_sim_preview_checkbox.setEnabled(SIMULATION_AVAILABLE)
        if not SIMULATION_AVAILABLE:
            self.cartesian_sim_preview_checkbox.setToolTip("仿真功能不可用")
        speed_layout.addWidget(self.cartesian_sim_preview_checkbox)
        
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        # 位置输入
        pos_group = QGroupBox("末端位置 (米)")
        pos_layout = QGridLayout()
        
        self.pos_spins = []
        pos_labels = ['X', 'Y', 'Z']
        for i, label in enumerate(pos_labels):
            pos_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-2.0, 2.0)
            spin.setValue(0.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            self.pos_spins.append(spin)
            pos_layout.addWidget(spin, i, 1)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # 姿态输入（欧拉角）
        rot_group = QGroupBox("末端姿态 (欧拉角 - 度)")
        rot_layout = QGridLayout()
        
        self.rot_spins = []
        rot_labels = ['Roll', 'Pitch', 'Yaw']
        for i, label in enumerate(rot_labels):
            rot_layout.addWidget(QLabel(label), i, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-180.0, 180.0)
            spin.setValue(0.0)
            spin.setDecimals(2)
            spin.setSingleStep(5.0)
            self.rot_spins.append(spin)
            rot_layout.addWidget(spin, i, 1)
        
        rot_group.setLayout(rot_layout)
        layout.addWidget(rot_group)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        
        self.move_cartesian_btn = QPushButton("移动到目标位姿")
        self.move_cartesian_btn.setEnabled(False)
        self.move_cartesian_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.move_cartesian_btn.clicked.connect(self.move_cartesian)
        btn_layout.addWidget(self.move_cartesian_btn)
        
        self.read_pose_btn = QPushButton("读取当前位姿")
        self.read_pose_btn.setEnabled(False)
        self.read_pose_btn.clicked.connect(self.read_current_pose)
        btn_layout.addWidget(self.read_pose_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        return widget
    
    def create_gripper_control_tab(self):
        """创建夹爪控制标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 预设动作
        preset_group = QGroupBox("预设动作")
        preset_layout = QVBoxLayout()
        
        btn_layout1 = QHBoxLayout()
        self.open_gripper_btn = QPushButton("打开夹爪")
        self.open_gripper_btn.setEnabled(False)
        self.open_gripper_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.open_gripper_btn.clicked.connect(self.open_gripper)
        btn_layout1.addWidget(self.open_gripper_btn)
        
        self.close_gripper_btn = QPushButton("闭合夹爪")
        self.close_gripper_btn.setEnabled(False)
        self.close_gripper_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.close_gripper_btn.clicked.connect(self.close_gripper)
        btn_layout1.addWidget(self.close_gripper_btn)
        
        preset_layout.addLayout(btn_layout1)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # 位置控制 - 真实值（米）
        pos_group = QGroupBox("位置控制 - 真实宽度 (米)")
        pos_layout = QVBoxLayout()
        
        pos_layout.addWidget(QLabel("夹爪宽度 (0.0 = 打开, 0.085 = 闭合):"))
        
        slider_layout = QHBoxLayout()
        self.gripper_slider = QSlider(Qt.Horizontal)
        self.gripper_slider.setRange(0, 85)  # 0-85mm对应0-0.085m
        self.gripper_slider.setValue(0)
        self.gripper_slider.valueChanged.connect(self.on_gripper_slider_changed)
        slider_layout.addWidget(self.gripper_slider)
        
        self.gripper_width_spin = QDoubleSpinBox()  # 真实宽度（米）
        self.gripper_width_spin.setRange(0.0, 0.085)
        self.gripper_width_spin.setValue(0.0)
        self.gripper_width_spin.setDecimals(4)
        self.gripper_width_spin.setSingleStep(0.001)
        self.gripper_width_spin.valueChanged.connect(self.on_gripper_width_spin_changed)
        slider_layout.addWidget(self.gripper_width_spin)
        
        pos_layout.addLayout(slider_layout)
        
        # 原始值显示（电机位置，弧度）
        raw_group = QGroupBox("电机原始值 (弧度)")
        raw_layout = QHBoxLayout()
        raw_layout.addWidget(QLabel("电机位置:"))
        self.gripper_raw_label = QLabel("0.0000")
        self.gripper_raw_label.setStyleSheet("background-color: #f5f5f5; padding: 5px; font-family: monospace;")
        raw_layout.addWidget(self.gripper_raw_label)
        raw_layout.addStretch()
        raw_group.setLayout(raw_layout)
        pos_layout.addWidget(raw_group)
        
        pos_layout.addWidget(QLabel("速度:"))
        self.gripper_vel_spin = QDoubleSpinBox()
        self.gripper_vel_spin.setRange(0.01, 1.0)
        self.gripper_vel_spin.setValue(0.2)
        self.gripper_vel_spin.setDecimals(2)
        pos_layout.addWidget(self.gripper_vel_spin)
        
        btn_layout2 = QHBoxLayout()
        self.read_gripper_btn = QPushButton("读取当前值")
        self.read_gripper_btn.setEnabled(False)
        self.read_gripper_btn.clicked.connect(self.read_gripper_status)
        btn_layout2.addWidget(self.read_gripper_btn)
        
        self.move_gripper_btn = QPushButton("移动到目标位置")
        self.move_gripper_btn.setEnabled(False)
        self.move_gripper_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.move_gripper_btn.clicked.connect(self.move_gripper)
        btn_layout2.addWidget(self.move_gripper_btn)
        
        pos_layout.addLayout(btn_layout2)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        layout.addStretch()
        return widget
    
    def create_single_motor_tab(self):
        """创建单电机控制标签页（调试用）"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        info_label = QLabel("⚠️ 警告: 单电机控制模式仅供调试使用，请谨慎操作！")
        info_label.setStyleSheet("color: #f44336; font-weight: bold; padding: 10px;")
        layout.addWidget(info_label)
        
        motor_group = QGroupBox("电机选择")
        motor_layout = QHBoxLayout()
        motor_layout.addWidget(QLabel("电机索引 (0-6):"))
        self.motor_index_spin = QSpinBox()
        self.motor_index_spin.setRange(0, 6)
        self.motor_index_spin.setValue(0)
        motor_layout.addWidget(self.motor_index_spin)
        motor_group.setLayout(motor_layout)
        layout.addWidget(motor_group)
        
        control_group = QGroupBox("控制参数")
        control_layout = QGridLayout()
        
        control_layout.addWidget(QLabel("目标位置 (弧度):"), 0, 0)
        self.single_motor_pos_spin = QDoubleSpinBox()
        self.single_motor_pos_spin.setRange(-3.5, 3.5)
        self.single_motor_pos_spin.setValue(0.0)
        self.single_motor_pos_spin.setDecimals(4)
        control_layout.addWidget(self.single_motor_pos_spin, 0, 1)
        
        control_layout.addWidget(QLabel("速度:"), 1, 0)
        self.single_motor_vel_spin = QDoubleSpinBox()
        self.single_motor_vel_spin.setRange(0.01, 1.0)
        self.single_motor_vel_spin.setValue(0.3)
        self.single_motor_vel_spin.setDecimals(2)
        control_layout.addWidget(self.single_motor_vel_spin, 1, 1)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        self.move_single_motor_btn = QPushButton("移动单电机")
        self.move_single_motor_btn.setEnabled(False)
        self.move_single_motor_btn.setStyleSheet("background-color: #FF9800; color: white;")
        self.move_single_motor_btn.clicked.connect(self.move_single_motor)
        layout.addWidget(self.move_single_motor_btn)
        
        layout.addStretch()
        return widget
    
    def create_presets_tab(self):
        """创建预设关节组管理标签页"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # 预设列表
        list_group = QGroupBox("预设关节组列表")
        list_layout = QVBoxLayout()
        
        self.presets_list = QListWidget()
        self.presets_list.itemDoubleClicked.connect(self.on_preset_double_clicked)
        list_layout.addWidget(self.presets_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # 操作按钮
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout()
        
        # 第一行按钮
        btn_row1 = QHBoxLayout()
        self.save_preset_btn = QPushButton("保存当前关节")
        self.save_preset_btn.setEnabled(False)
        self.save_preset_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.save_preset_btn.clicked.connect(self.save_current_joints_as_preset)
        btn_row1.addWidget(self.save_preset_btn)
        
        self.load_preset_btn = QPushButton("加载到输入框")
        self.load_preset_btn.setEnabled(False)
        self.load_preset_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.load_preset_btn.clicked.connect(self.load_preset_to_inputs)
        btn_row1.addWidget(self.load_preset_btn)
        
        btn_layout.addLayout(btn_row1)
        
        # 第二行按钮
        btn_row2 = QHBoxLayout()
        self.execute_preset_btn = QPushButton("执行预设")
        self.execute_preset_btn.setEnabled(False)
        self.execute_preset_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.execute_preset_btn.clicked.connect(self.execute_selected_preset)
        btn_row2.addWidget(self.execute_preset_btn)
        
        self.delete_preset_btn = QPushButton("删除预设")
        self.delete_preset_btn.setEnabled(False)
        self.delete_preset_btn.setStyleSheet("background-color: #f44336; color: white;")
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        btn_row2.addWidget(self.delete_preset_btn)
        
        btn_layout.addLayout(btn_row2)
        
        btn_group.setLayout(btn_layout)
        layout.addWidget(btn_group)
        
        # 预设信息显示
        info_group = QGroupBox("预设信息")
        info_layout = QVBoxLayout()
        self.preset_info_label = QLabel("选择一个预设以查看详细信息")
        self.preset_info_label.setStyleSheet("background-color: #f5f5f5; padding: 10px; font-family: monospace;")
        self.preset_info_label.setWordWrap(True)
        info_layout.addWidget(self.preset_info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # 连接列表选择事件
        self.presets_list.itemSelectionChanged.connect(self.on_preset_selection_changed)
        
        layout.addStretch()
        return widget
    
    def create_right_panel(self):
        """创建右侧状态面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 状态显示组
        status_group = QGroupBox("实时状态")
        status_layout = QVBoxLayout()
        
        # 连接状态
        self.connection_status_label = QLabel("状态: 未连接")
        self.connection_status_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        status_layout.addWidget(self.connection_status_label)
        
        # 使能状态
        self.enable_status_label = QLabel("使能状态: 未知")
        status_layout.addWidget(self.enable_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # 关节状态显示（角度、速度、力矩）
        joint_display_group = QGroupBox("关节状态")
        joint_display_layout = QGridLayout()
        
        # 表头
        joint_display_layout.addWidget(QLabel("关节"), 0, 0)
        joint_display_layout.addWidget(QLabel("角度 (rad)"), 0, 1)
        joint_display_layout.addWidget(QLabel("速度 (rad/s)"), 0, 2)
        joint_display_layout.addWidget(QLabel("力矩 (N·m)"), 0, 3)
        
        # 设置表头样式
        for col in range(4):
            header_label = joint_display_layout.itemAtPosition(0, col).widget()
            if header_label:
                header_label.setStyleSheet("font-weight: bold; padding: 3px;")
        
        self.joint_display_labels = []  # 角度标签
        self.joint_velocity_labels = []  # 速度标签
        self.joint_torque_labels = []  # 力矩标签
        
        joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        for i, label in enumerate(joint_labels):
            row = i + 1  # 从第1行开始（第0行是表头）
            
            # 关节名称
            joint_display_layout.addWidget(QLabel(label), row, 0)
            
            # 角度值
            angle_label = QLabel("0.0000")
            angle_label.setStyleSheet("background-color: #f5f5f5; padding: 3px; font-family: monospace;")
            self.joint_display_labels.append(angle_label)
            joint_display_layout.addWidget(angle_label, row, 1)
            
            # 速度值
            velocity_label = QLabel("0.0000")
            velocity_label.setStyleSheet("background-color: #e3f2fd; padding: 3px; font-family: monospace;")
            self.joint_velocity_labels.append(velocity_label)
            joint_display_layout.addWidget(velocity_label, row, 2)
            
            # 力矩值
            torque_label = QLabel("0.0000")
            torque_label.setStyleSheet("background-color: #fff3e0; padding: 3px; font-family: monospace;")
            self.joint_torque_labels.append(torque_label)
            joint_display_layout.addWidget(torque_label, row, 3)
        
        joint_display_group.setLayout(joint_display_layout)
        layout.addWidget(joint_display_group)
        
        # 末端位姿显示
        pose_display_group = QGroupBox("末端位姿")
        pose_display_layout = QVBoxLayout()
        
        pos_layout = QGridLayout()
        pos_layout.addWidget(QLabel("位置 (m):"), 0, 0)
        self.pos_display_labels = []
        for i, label in enumerate(['X', 'Y', 'Z']):
            pos_layout.addWidget(QLabel(label), 0, i+1)
            value_label = QLabel("0.0000")
            value_label.setStyleSheet("background-color: #f5f5f5; padding: 3px;")
            self.pos_display_labels.append(value_label)
            pos_layout.addWidget(value_label, 1, i+1)
        pose_display_layout.addLayout(pos_layout)
        
        pose_display_group.setLayout(pose_display_layout)
        layout.addWidget(pose_display_group)
        
        # 夹爪状态显示
        gripper_display_group = QGroupBox("夹爪状态")
        gripper_display_layout = QVBoxLayout()
        self.gripper_status_label = QLabel("位置: 未知")
        self.gripper_status_label.setStyleSheet("background-color: #f5f5f5; padding: 5px;")
        gripper_display_layout.addWidget(self.gripper_status_label)
        gripper_display_group.setLayout(gripper_display_layout)
        layout.addWidget(gripper_display_group)
        
        # 日志显示
        log_group = QGroupBox("操作日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 10px;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        return panel
    
    def setup_styles(self):
        """设置界面样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 5px;
                border-radius: 3px;
                min-height: 25px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
    
    def log_message(self, message: str, level: str = "INFO"):
        """记录日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        # 如果log_text已创建，则更新GUI显示
        if hasattr(self, 'log_text') and self.log_text is not None:
            self.log_text.append(log_entry)
            
            # 保持日志在合理长度
            if self.log_text.document().blockCount() > 100:
                cursor = self.log_text.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.movePosition(cursor.Down, cursor.MoveAnchor, 50)
                cursor.movePosition(cursor.Start, cursor.KeepAnchor)
                cursor.removeSelectedText()
        
        # 同时写入文件日志
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def connect_robot(self):
        """连接机械臂"""
        if self.is_connected:
            QMessageBox.warning(self, "警告", "机械臂已连接！")
            return
        
        can_name = self.can_name_input.text().strip()
        if not can_name:
            QMessageBox.warning(self, "错误", "请输入 CAN 接口名称！")
            return
        
        try:
            self.log_message(f"正在连接机械臂 (CAN: {can_name})...")
            self.status_bar.showMessage("正在连接...")
            
            self.controller = OpenArmController(
                can_name=can_name,
                auto_enable=False,
                force_vcan_setup=False
            )
            
            self.is_connected = True
            
            # 更新 UI
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.enable_btn.setEnabled(True)
            self.disable_btn.setEnabled(True)
            self.emergency_btn.setEnabled(True)
            self.read_all_joints_btn.setEnabled(True)
            self.move_joint_btn.setEnabled(True)
            self.move_cartesian_btn.setEnabled(True)
            self.read_pose_btn.setEnabled(True)
            self.open_gripper_btn.setEnabled(True)
            self.close_gripper_btn.setEnabled(True)
            self.read_gripper_btn.setEnabled(True)
            self.move_gripper_btn.setEnabled(True)
            self.move_single_motor_btn.setEnabled(True)
            
            # 启用预设相关按钮
            if hasattr(self, 'save_preset_btn'):
                self.save_preset_btn.setEnabled(True)
                self.log_message("已启用'保存当前关节'按钮", "INFO")
            else:
                self.log_message("警告: save_preset_btn 属性不存在", "WARNING")
            
            self.connection_status_label.setText("状态: 已连接")
            self.connection_status_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; color: #4CAF50;")
            
            # 更新预设执行按钮状态
            if hasattr(self, 'execute_preset_btn'):
                selected_items = self.presets_list.selectedItems()
                self.execute_preset_btn.setEnabled(len(selected_items) > 0)
            
            # 启动状态监控线程
            self.start_monitoring()
            
            # 如果仿真窗口正在运行，启用仿真更新
            if self.sim_running and self.monitor_thread:
                self.monitor_thread.enable_sim_update = True
            
            self.log_message("机械臂连接成功！")
            self.status_bar.showMessage(f"已连接 - CAN: {can_name}")
            
        except Exception as e:
            error_msg = f"连接失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "连接错误", error_msg)
            self.controller = None
            self.is_connected = False
    
    def disconnect_robot(self):
        """断开机械臂连接"""
        if not self.is_connected:
            return
        
        try:
            self.log_message("正在断开连接...")
            
            # 停止监控
            self.stop_monitoring()
            
            # 禁用电机
            if self.controller:
                try:
                    self.controller.disable()
                except:
                    pass
                
                try:
                    self.controller.close_connection()
                except:
                    pass
            
            self.controller = None
            self.is_connected = False
            
            # 更新 UI
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.enable_btn.setEnabled(False)
            self.disable_btn.setEnabled(False)
            self.emergency_btn.setEnabled(False)
            self.read_all_joints_btn.setEnabled(False)
            self.move_joint_btn.setEnabled(False)
            self.move_cartesian_btn.setEnabled(False)
            self.read_pose_btn.setEnabled(False)
            self.open_gripper_btn.setEnabled(False)
            self.close_gripper_btn.setEnabled(False)
            self.read_gripper_btn.setEnabled(False)
            self.move_gripper_btn.setEnabled(False)
            self.move_single_motor_btn.setEnabled(False)
            self.save_preset_btn.setEnabled(False)
            # 更新预设执行按钮状态
            if hasattr(self, 'execute_preset_btn'):
                selected_items = self.presets_list.selectedItems()
                self.execute_preset_btn.setEnabled(False)
            
            self.connection_status_label.setText("状态: 未连接")
            self.connection_status_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px; color: #f44336;")
            self.enable_status_label.setText("使能状态: 未知")
            
            # 清空显示
            for label in self.joint_display_labels:
                label.setText("0.0000")
            if hasattr(self, 'joint_velocity_labels'):
                for label in self.joint_velocity_labels:
                    label.setText("0.0000")
            if hasattr(self, 'joint_torque_labels'):
                for label in self.joint_torque_labels:
                    label.setText("0.0000")
            for label in self.pos_display_labels:
                label.setText("0.0000")
            self.gripper_status_label.setText("位置: 未知")
            
            self.log_message("已断开连接")
            self.status_bar.showMessage("已断开连接")
            
        except Exception as e:
            error_msg = f"断开连接时出错: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "错误", error_msg)
    
    def enable_robot(self):
        """使能所有电机"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            self.log_message("使能所有电机...")
            self.controller.enable()
            self.log_message("所有电机已使能")
            self.status_bar.showMessage("所有电机已使能")
        except Exception as e:
            error_msg = f"使能失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def disable_robot(self):
        """禁用所有电机"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            self.log_message("禁用所有电机...")
            self.controller.disable()
            self.log_message("所有电机已禁用")
            self.status_bar.showMessage("所有电机已禁用")
        except Exception as e:
            error_msg = f"禁用失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def emergency_stop(self):
        """紧急停止"""
        if not self.is_connected or not self.controller:
            return
        
        reply = QMessageBox.question(
            self, 
            "紧急停止", 
            "确定要执行紧急停止吗？\n这将立即禁用所有电机！",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                self.log_message("执行紧急停止！", "WARNING")
                self.controller.disable()
                self.log_message("紧急停止完成")
                self.status_bar.showMessage("紧急停止已执行", 5000)
                QMessageBox.warning(self, "紧急停止", "所有电机已禁用！")
            except Exception as e:
                error_msg = f"紧急停止失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.critical(self, "错误", error_msg)
    
    def start_monitoring(self):
        """启动状态监控"""
        if self.monitor_thread is not None:
            return
        
        if self.controller:
            self.monitor_thread = StatusMonitorThread(self.controller, update_rate=0.1)
            self.monitor_thread.status_update.connect(self.update_status_display)
            self.monitor_thread.error_occurred.connect(lambda msg: self.log_message(msg, "ERROR"))
            self.monitor_thread.sim_update.connect(self.update_simulation_joints)
            # 如果仿真窗口正在运行，启用仿真更新
            if self.sim_running:
                self.monitor_thread.enable_sim_update = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止状态监控"""
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread = None
    
    def update_status_display(self, status: dict):
        """更新状态显示"""
        try:
            # 更新关节角度
            if status.get('joint_values') is not None:
                joint_values = status['joint_values']
                for i, label in enumerate(self.joint_display_labels):
                    if i < len(joint_values):
                        label.setText(f"{joint_values[i]:.4f}")
            
            # 更新关节速度
            if status.get('joint_velocities') is not None:
                joint_velocities = status['joint_velocities']
                for i, label in enumerate(self.joint_velocity_labels):
                    if i < len(joint_velocities):
                        label.setText(f"{joint_velocities[i]:.4f}")
            elif hasattr(self, 'joint_velocity_labels'):
                # 如果没有速度数据，显示 "--"
                for label in self.joint_velocity_labels:
                    label.setText("--")
            
            # 更新关节力矩
            if status.get('joint_torques') is not None:
                joint_torques = status['joint_torques']
                for i, label in enumerate(self.joint_torque_labels):
                    if i < len(joint_torques):
                        label.setText(f"{joint_torques[i]:.4f}")
            elif hasattr(self, 'joint_torque_labels'):
                # 如果没有力矩数据，显示 "--"
                for label in self.joint_torque_labels:
                    label.setText("--")
            
            # 更新末端位姿
            if status.get('position') is not None:
                pos = status['position']
                for i, label in enumerate(self.pos_display_labels):
                    if i < len(pos):
                        label.setText(f"{pos[i]:.4f}")
            
            # 更新夹爪状态
            if status.get('gripper_status') is not None:
                gripper_pos = status['gripper_status']
                if len(gripper_pos) > 0:
                    raw_pos = gripper_pos[0]
                    try:
                        # 转换为真实宽度
                        if self.controller:
                            width_m = self.controller.map_motor_position_to_gripper_width(raw_pos)
                            self.gripper_status_label.setText(f"原始值: {raw_pos:.4f} rad, 宽度: {width_m:.4f} m")
                        else:
                            self.gripper_status_label.setText(f"原始值: {raw_pos:.4f} rad")
                    except:
                        self.gripper_status_label.setText(f"原始值: {raw_pos:.4f} rad")
            
            # 更新使能状态
            is_enabled = status.get('is_enabled', False)
            if is_enabled:
                self.enable_status_label.setText("使能状态: ✓ 已使能")
                self.enable_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                self.enable_status_label.setText("使能状态: ✗ 未使能")
                self.enable_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
                
        except Exception as e:
            self.log_message(f"状态更新错误: {str(e)}", "ERROR")
    
    def read_all_joint_values(self):
        """读取所有关节的当前值"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            joint_values = self.controller.get_joint_values()
            for i, spin in enumerate(self.joint_spins):
                if i < len(joint_values):
                    spin.setValue(joint_values[i])
            self.log_message(f"读取所有关节角度: {joint_values}")
        except Exception as e:
            error_msg = f"读取关节值失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "错误", error_msg)
    
    def move_joint(self):
        """执行关节运动"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            # 获取目标角度
            target_angles = np.array([spin.value() for spin in self.joint_spins])
            speed = self.joint_speed_spin.value()
            block = self.block_checkbox.isChecked()
            
            # 仿真预览
            if self.sim_preview_checkbox.isChecked() and SIMULATION_AVAILABLE:
                if not self.simulate_joint_motion(target_angles):
                    return  # 仿真失败，不执行实际运动
            
            self.log_message(f"执行关节运动: {target_angles}, 速度: {speed}, 阻塞: {block}")
            self.status_bar.showMessage("正在执行关节运动...")
            
            self.controller.move_j(
                joint_angles=target_angles,
                speed=speed,
                is_radians=True,
                block=block
            )
            
            self.log_message("关节运动指令已发送")
            if not block:
                self.status_bar.showMessage("关节运动指令已发送（非阻塞）")
            else:
                self.status_bar.showMessage("关节运动完成")
                
        except Exception as e:
            error_msg = f"关节运动失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
            self.status_bar.showMessage("关节运动失败")
    
    def move_to_home(self):
        """移动到零位"""
        if not self.is_connected or not self.controller:
            return
        
        reply = QMessageBox.question(
            self,
            "确认",
            "确定要移动到零位吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                home_angles = [0.0] * 7
                speed = self.joint_speed_spin.value()
                block = self.block_checkbox.isChecked()
                
                self.log_message("移动到零位...")
                self.controller.move_j(
                    joint_angles=home_angles,
                    speed=speed,
                    is_radians=True,
                    block=block
                )
                self.log_message("已移动到零位")
            except Exception as e:
                error_msg = f"移动到零位失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.critical(self, "错误", error_msg)
    
    def read_current_pose(self):
        """读取当前位姿"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            pos, rot = self.controller.get_pose()
            if pos is not None:
                for i, spin in enumerate(self.pos_spins):
                    if i < len(pos):
                        spin.setValue(pos[i])
            
            # 将旋转矩阵转换为欧拉角
            if rot is not None:
                try:
                    import wrs.basis.robot_math as rm
                    # 使用 robot_math 中的 rotmat_to_euler 函数
                    # 默认使用 'sxyz' 顺序（固定轴XYZ顺序）
                    euler_angles = rm.rotmat_to_euler(rot, order='sxyz')
                    # 转换为度并更新UI
                    for i, spin in enumerate(self.rot_spins):
                        if i < len(euler_angles):
                            spin.setValue(np.degrees(euler_angles[i]))
                    self.log_message(f"当前位置: {pos}, 姿态: {np.degrees(euler_angles)}")
                except ImportError:
                    self.log_message("警告: 无法导入 robot_math，无法转换旋转矩阵", "WARNING")
                    self.log_message(f"当前位置: {pos}, 旋转矩阵已读取但未转换")
                except Exception as e:
                    self.log_message(f"旋转矩阵转换失败: {str(e)}", "WARNING")
                    self.log_message(f"当前位置: {pos}, 旋转矩阵已读取但转换失败")
            
            self.log_message("当前位姿已读取")
        except Exception as e:
            error_msg = f"读取位姿失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "错误", error_msg)
    
    def move_cartesian(self):
        """执行笛卡尔空间运动"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            # 获取目标位置
            target_pos = np.array([spin.value() for spin in self.pos_spins])
            
            # 获取目标姿态（欧拉角转旋转矩阵）
            try:
                import wrs.basis.robot_math as rm
                euler_angles = [np.radians(spin.value()) for spin in self.rot_spins]
                target_rot = rm.rotmat_from_euler(*euler_angles)
            except ImportError:
                # 如果无法导入 robot_math，使用简化的旋转矩阵计算
                self.log_message("警告: 无法导入 robot_math，使用单位旋转矩阵", "WARNING")
                target_rot = np.eye(3)
            
            speed = self.cartesian_speed_spin.value()
            
            # 仿真预览（包括IK求解验证）
            if self.cartesian_sim_preview_checkbox.isChecked() and SIMULATION_AVAILABLE:
                if not self.simulate_cartesian_motion(target_pos, target_rot):
                    return  # 仿真失败（可能是IK求解失败），不执行实际运动
            
            self.log_message(f"执行笛卡尔运动: 位置={target_pos}, 速度={speed}")
            self.status_bar.showMessage("正在执行笛卡尔运动...")
            
            self.controller.move_p(
                pos=target_pos,
                rotmat=target_rot,
                speed=speed,
                block=False
            )
            
            self.log_message("笛卡尔运动指令已发送")
            self.status_bar.showMessage("笛卡尔运动指令已发送")
            
        except Exception as e:
            error_msg = f"笛卡尔运动失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
            self.status_bar.showMessage("笛卡尔运动失败")
    
    def open_gripper(self):
        """打开夹爪"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            vel = self.gripper_vel_spin.value()
            self.log_message("打开夹爪...")
            self.controller.open_gripper(vel=vel)
            self.log_message("夹爪打开指令已发送")
        except Exception as e:
            error_msg = f"打开夹爪失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def close_gripper(self):
        """闭合夹爪"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            vel = self.gripper_vel_spin.value()
            self.log_message("闭合夹爪...")
            self.controller.close_gripper(vel=vel)
            self.log_message("夹爪闭合指令已发送")
        except Exception as e:
            error_msg = f"闭合夹爪失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def on_gripper_slider_changed(self, value: int):
        """夹爪滑块值改变（单位：毫米，范围0-85）"""
        width_m = value / 1000.0  # 转换为米
        self.gripper_width_spin.blockSignals(True)
        self.gripper_width_spin.setValue(width_m)
        self.gripper_width_spin.blockSignals(False)
    
    def on_gripper_width_spin_changed(self, value: float):
        """夹爪宽度输入值改变（单位：米）"""
        slider_value = int(value * 1000)  # 转换为毫米
        slider_value = max(0, min(85, slider_value))  # 限制在0-85范围内
        self.gripper_slider.blockSignals(True)
        self.gripper_slider.setValue(slider_value)
        self.gripper_slider.blockSignals(False)
    
    def read_gripper_status(self):
        """读取夹爪当前状态"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            gripper_status = self.controller.get_gripper_status()
            if gripper_status is not None and len(gripper_status) > 0:
                # 获取电机原始位置（弧度）
                raw_position = gripper_status[0]
                self.gripper_raw_label.setText(f"{raw_position:.4f}")
                
                # 转换为真实宽度（米）
                try:
                    width_m = self.controller.map_motor_position_to_gripper_width(raw_position)
                    self.gripper_width_spin.blockSignals(True)
                    self.gripper_width_spin.setValue(width_m)
                    self.gripper_width_spin.blockSignals(False)
                    
                    # 更新滑块
                    slider_value = int(width_m * 1000)
                    self.gripper_slider.blockSignals(True)
                    self.gripper_slider.setValue(slider_value)
                    self.gripper_slider.blockSignals(False)
                    
                    self.log_message(f"读取夹爪状态: 原始值={raw_position:.4f} rad, 宽度={width_m:.4f} m")
                except Exception as e:
                    self.log_message(f"转换夹爪位置失败: {str(e)}", "WARNING")
                    self.log_message(f"读取夹爪原始值: {raw_position:.4f} rad")
        except Exception as e:
            error_msg = f"读取夹爪状态失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "错误", error_msg)
    
    def move_gripper(self):
        """移动夹爪到目标位置（使用真实宽度，单位：米）"""
        if not self.is_connected or not self.controller:
            return
        
        try:
            width_m = self.gripper_width_spin.value()
            vel = self.gripper_vel_spin.value()
            
            # 检查范围
            if width_m < 0.0 or width_m > 0.085:
                QMessageBox.warning(self, "错误", f"夹爪宽度超出范围 [0.0, 0.085] 米")
                return
            
            self.log_message(f"移动夹爪到宽度: {width_m:.4f} m, 速度: {vel}")
            self.controller.gripper_control(pos=width_m, vel=vel)
            self.log_message("夹爪控制指令已发送")
        except Exception as e:
            error_msg = f"夹爪控制失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def move_single_motor(self):
        """单电机控制"""
        if not self.is_connected or not self.controller:
            return
        
        reply = QMessageBox.warning(
            self,
            "警告",
            "单电机控制模式仅供调试使用，可能导致机械臂处于不安全状态！\n确定要继续吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                index = self.motor_index_spin.value()
                position = self.single_motor_pos_spin.value()
                vel = self.single_motor_vel_spin.value()
                
                self.log_message(f"单电机控制: 电机{index}, 位置={position:.4f}, 速度={vel}", "WARNING")
                self.controller.move_single_motor(
                    index=index,
                    position=position,
                    vel=vel
                )
                self.log_message("单电机控制指令已发送")
            except Exception as e:
                error_msg = f"单电机控制失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.critical(self, "错误", error_msg)
    
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 OpenArm 控制器",
            "OpenArm 机械臂控制器 v1.0\n\n"
            "商业级图形界面控制器\n"
            "提供完整的机械臂控制功能\n\n"
            "功能包括:\n"
            "• 关节空间控制\n"
            "• 笛卡尔空间控制\n"
            "• 夹爪控制\n"
            "• 实时状态监控\n"
            "• 安全控制\n\n"
            "© 2025"
        )
    
    def toggle_simulation(self):
        """打开/关闭实时仿真预览窗口"""
        if not SIMULATION_AVAILABLE:
            QMessageBox.warning(self, "警告", "仿真功能不可用，请检查依赖库是否正确安装。")
            return
        
        if not self.sim_running:
            self.start_simulation_window()
        else:
            self.stop_simulation_window()
    
    def init_simulation(self):
        """初始化仿真环境（用于验证，不创建可视化窗口）"""
        try:
            if self.sim_arm is None:
                # 创建仿真机械臂（不加载mesh以提高速度）
                self.sim_arm = sim_openarm.OpenArm(load_meshes=False)
            
            # 如果已连接，同步当前关节角度
            if self.is_connected and self.controller:
                try:
                    current_joints = self.controller.get_joint_values()
                    self.sim_arm.goto_given_conf(current_joints)
                except:
                    pass
            
            self.sim_enabled = True
            
        except Exception as e:
            error_msg = f"初始化仿真环境失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.sim_arm = None
            self.sim_enabled = False
    
    def start_simulation_window(self):
        """启动实时仿真预览窗口"""
        if not SIMULATION_AVAILABLE:
            return
        
        if self.sim_running:
            self.log_message("仿真窗口已在运行")
            return
        
        try:
            self.log_message("正在启动实时仿真预览窗口...")
            
            # 初始化仿真机械臂（如果还没有）
            if self.sim_arm is None:
                self.init_simulation()
            
            if not self.sim_enabled:
                QMessageBox.warning(self, "错误", "仿真环境初始化失败")
                return
            
            # 使用subprocess创建独立进程运行仿真窗口（Panda3D必须在主线程运行）
            import subprocess
            
            # 使用subprocess创建独立进程运行仿真窗口
            # Panda3D必须在主线程运行，所以使用独立进程
            import subprocess
            
            # 创建临时脚本文件来运行仿真窗口
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sim_script_path = os.path.join(script_dir, "sim_window.py")
            
            # 获取项目根目录
            current_file = os.path.abspath(__file__)
            # wrs/robot_con/openarm/openarm_gui.py -> 项目根目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
            
            # 创建仿真窗口脚本
            sim_script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立仿真窗口进程 - 实时预览OpenArm机械臂"""
import sys
import os
import numpy as np
import json
import time

# 添加项目根目录到路径
project_root = r"{project_root}"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.manipulators.openarm.openarm as sim_openarm
    import wrs.modeling.geometric_model as mgm
    
    # 预设文件路径（用于读取最新关节状态）
    presets_file = r"{self._presets_file}"
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
    print(f"仿真窗口启动失败: {{e}}")
    import traceback
    traceback.print_exc()
    input("按Enter键退出...")
'''
            
            # 写入脚本文件
            with open(sim_script_path, 'w', encoding='utf-8') as f:
                f.write(sim_script_content)
            
            # 设置执行权限
            os.chmod(sim_script_path, 0o755)
            
            # 启动独立进程
            self.sim_process = subprocess.Popen(
                [sys.executable, sim_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.sim_running = True
            
            # 启用状态监控线程的仿真更新（通过文件同步）
            if self.monitor_thread:
                self.monitor_thread.enable_sim_update = True
            
            self.log_message("实时仿真预览窗口已启动（独立进程，通过文件同步）")
            QMessageBox.information(
                self,
                "仿真预览",
                "实时仿真预览窗口已启动（独立进程）。\n\n"
                "窗口将实时显示机械臂的当前状态。\n"
                "关节角度通过文件同步更新。"
            )
            
        except Exception as e:
            error_msg = f"启动仿真窗口失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
            self.sim_running = False
    
    
    def stop_simulation_window(self):
        """停止实时仿真预览窗口"""
        if not self.sim_running:
            return
        
        try:
            self.sim_running = False
            
            # 禁用状态监控线程的仿真更新
            if self.monitor_thread:
                self.monitor_thread.enable_sim_update = False
            
            # 终止仿真进程
            if self.sim_process:
                try:
                    self.sim_process.terminate()
                    self.sim_process.wait(timeout=2)
                except:
                    try:
                        self.sim_process.kill()
                    except:
                        pass
                self.sim_process = None
            
            self.log_message("实时仿真预览窗口已关闭")
            
        except Exception as e:
            self.log_message(f"关闭仿真窗口错误: {str(e)}", "ERROR")
    
    def update_simulation_joints(self, joint_values: list):
        """更新仿真关节角度（由状态监控线程调用，通过文件同步到独立进程）"""
        if self.sim_running:
            try:
                # 将关节角度写入文件，供独立进程读取
                joint_state_file = os.path.join(self._constants_dir, "current_joints.json")
                joint_array = np.array(joint_values) if isinstance(joint_values, list) else joint_values
                
                with open(joint_state_file, 'w') as f:
                    json.dump({'joints': joint_array.tolist(), 'timestamp': time.time()}, f)
            except Exception as e:
                # 静默失败，避免影响主程序
                pass
    
    def simulate_joint_motion(self, target_angles: np.ndarray) -> bool:
        """
        仿真关节运动
        
        Args:
            target_angles: 目标关节角度数组
            
        Returns:
            bool: 如果仿真成功返回True，否则返回False
        """
        if not SIMULATION_AVAILABLE:
            return True  # 仿真不可用，允许继续执行
        
        if not self.sim_enabled or self.sim_arm is None:
            # 如果仿真未启用，尝试初始化
            self.init_simulation()
            if not self.sim_enabled:
                # 如果初始化失败，询问用户是否继续
                reply = QMessageBox.question(
                    self,
                    "仿真初始化失败",
                    "仿真环境初始化失败，是否继续执行实际运动？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                return reply == QMessageBox.Yes
        
        try:
            # 检查关节限制
            joint_limits = np.array([
                [-1.396263, 3.490659],
                [-1.745329, 1.745329],
                [-1.570796, 1.570796],
                [0.0, 2.443461],
                [-1.570796, 1.570796],
                [-0.785398, 0.785398],
                [-1.570796, 1.570796]
            ])
            
            for i, angle in enumerate(target_angles):
                if angle < joint_limits[i][0] or angle > joint_limits[i][1]:
                    error_msg = f"关节 {i+1} 角度 {angle:.4f} 超出限制范围 [{joint_limits[i][0]:.4f}, {joint_limits[i][1]:.4f}]"
                    self.log_message(error_msg, "ERROR")
                    QMessageBox.warning(self, "仿真验证失败", error_msg)
                    return False
            
            # 更新仿真机械臂姿态（仅用于验证，不更新可视化）
            self.sim_arm.goto_given_conf(target_angles)
            
            # 验证正运动学（确保姿态有效）
            try:
                pos, rot = self.sim_arm.fk(target_angles)
                if pos is None or rot is None:
                    raise ValueError("正运动学求解失败")
            except Exception as e:
                error_msg = f"正运动学验证失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.warning(self, "仿真验证失败", error_msg)
                return False
            
            self.log_message("仿真预览：关节运动验证通过")
            return True
            
        except Exception as e:
            error_msg = f"仿真验证失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "仿真验证失败", error_msg)
            return False
    
    def simulate_cartesian_motion(self, target_pos: np.ndarray, target_rot: np.ndarray) -> bool:
        """
        仿真笛卡尔空间运动（包括IK求解验证）
        
        Args:
            target_pos: 目标位置
            target_rot: 目标旋转矩阵
            
        Returns:
            bool: 如果仿真成功（IK求解成功）返回True，否则返回False
        """
        if not SIMULATION_AVAILABLE:
            return True  # 仿真不可用，允许继续执行
        
        if not self.sim_enabled or self.sim_arm is None:
            # 如果仿真未启用，尝试初始化
            self.init_simulation()
            if not self.sim_enabled:
                # 如果初始化失败，询问用户是否继续
                reply = QMessageBox.question(
                    self,
                    "仿真初始化失败",
                    "仿真环境初始化失败，是否继续执行实际运动？\n警告：无法验证IK求解是否成功。",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                return reply == QMessageBox.Yes
        
        try:
            # 获取当前关节角度作为种子值
            seed_joints = None
            if self.is_connected and self.controller:
                try:
                    seed_joints = self.controller.get_joint_values()
                except:
                    pass
            
            # 尝试求解IK
            joint_solution = self.sim_arm.ik(
                tgt_pos=target_pos,
                tgt_rotmat=target_rot,
                seed_jnt_values=seed_joints
            )
            
            if joint_solution is None:
                error_msg = "无法求解该位置的逆运动学解（IK求解失败）\n\n可能原因：\n1. 目标位置超出工作空间\n2. 目标姿态不可达\n3. 初始种子值不合适\n\n建议：\n- 调整目标位置或姿态\n- 尝试不同的初始关节角度"
                self.log_message("仿真验证失败：IK求解失败", "ERROR")
                QMessageBox.warning(self, "IK求解失败", error_msg)
                return False
            
            # 检查关节限制
            joint_limits = np.array([
                [-1.396263, 3.490659],
                [-1.745329, 1.745329],
                [-1.570796, 1.570796],
                [0.0, 2.443461],
                [-1.570796, 1.570796],
                [-0.785398, 0.785398],
                [-1.570796, 1.570796]
            ])
            
            for i, angle in enumerate(joint_solution):
                if angle < joint_limits[i][0] or angle > joint_limits[i][1]:
                    error_msg = f"IK求解结果中关节 {i+1} 角度 {angle:.4f} 超出限制范围 [{joint_limits[i][0]:.4f}, {joint_limits[i][1]:.4f}]"
                    self.log_message(f"仿真验证失败：{error_msg}", "ERROR")
                    QMessageBox.warning(self, "IK求解失败", error_msg)
                    return False
            
            # 更新仿真机械臂姿态（仅用于验证）
            self.sim_arm.goto_given_conf(joint_solution)
            
            # 验证正运动学（确保IK解正确）
            try:
                pos_check, rot_check = self.sim_arm.fk(joint_solution)
                if pos_check is None or rot_check is None:
                    raise ValueError("IK解的正运动学验证失败")
                
                # 检查位置和姿态误差
                pos_error = np.linalg.norm(pos_check - target_pos)
                rot_error = np.linalg.norm(rot_check - target_rot)
                
                if pos_error > 0.01:  # 位置误差阈值1cm
                    error_msg = f"IK解的位置误差过大: {pos_error:.4f} m"
                    self.log_message(error_msg, "WARNING")
                    # 不阻止执行，但给出警告
                
            except Exception as e:
                error_msg = f"IK解验证失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.warning(self, "仿真验证失败", error_msg)
                return False
            
            self.log_message(f"仿真预览：IK求解成功，关节角度: {joint_solution}")
            return True
            
        except Exception as e:
            error_msg = f"仿真验证失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.warning(self, "仿真验证失败", error_msg)
            return False
    
    # ========== 预设关节组管理功能 ==========
    
    def load_joint_presets(self):
        """加载预设关节组"""
        try:
            if os.path.exists(self._presets_file):
                # 检查文件是否为空
                file_size = os.path.getsize(self._presets_file)
                if file_size == 0:
                    self.joint_presets = {}
                    self.log_message("预设文件为空，将创建新文件")
                else:
                    with open(self._presets_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:
                            self.joint_presets = {}
                            self.log_message("预设文件为空，将创建新文件")
                        else:
                            self.joint_presets = json.loads(content)
                            if not isinstance(self.joint_presets, dict):
                                self.joint_presets = {}
                                self.log_message("预设文件格式错误，已重置")
                            else:
                                self.log_message(f"已加载 {len(self.joint_presets)} 个预设关节组")
            else:
                self.joint_presets = {}
                self.log_message("预设文件不存在，将创建新文件")
            
            self.update_presets_list()
        except json.JSONDecodeError as e:
            error_msg = f"预设文件JSON格式错误: {str(e)}，已重置"
            self.log_message(error_msg, "ERROR")
            self.joint_presets = {}
            # 备份损坏的文件
            try:
                backup_file = self._presets_file + ".backup"
                if os.path.exists(self._presets_file):
                    import shutil
                    shutil.copy(self._presets_file, backup_file)
                    self.log_message(f"已备份损坏的文件到: {backup_file}")
            except:
                pass
        except Exception as e:
            error_msg = f"加载预设关节组失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            self.joint_presets = {}
    
    def save_joint_presets(self):
        """保存预设关节组到文件"""
        try:
            with open(self._presets_file, 'w', encoding='utf-8') as f:
                json.dump(self.joint_presets, f, indent=2, ensure_ascii=False)
            self.log_message(f"预设关节组已保存到: {self._presets_file}")
        except Exception as e:
            error_msg = f"保存预设关节组失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def update_presets_list(self):
        """更新预设列表显示"""
        # 检查presets_list是否已创建
        if not hasattr(self, 'presets_list') or self.presets_list is None:
            return
        
        self.presets_list.clear()
        for name in sorted(self.joint_presets.keys()):
            item = QListWidgetItem(name)
            self.presets_list.addItem(item)
    
    def save_current_joints_as_preset(self):
        """保存当前关节角度为预设"""
        self.log_message("保存当前关节按钮被点击", "INFO")
        
        if not self.is_connected or not self.controller:
            error_msg = "请先连接机械臂"
            self.log_message(error_msg, "WARNING")
            QMessageBox.warning(self, "警告", error_msg)
            return
        
        try:
            self.log_message("正在获取当前关节角度...", "INFO")
            # 获取当前关节角度
            current_joints = self.controller.get_joint_values()
            joint_list = current_joints.tolist()
            self.log_message(f"获取到关节角度: {joint_list}", "INFO")
            
            # 输入预设名称
            name, ok = QInputDialog.getText(
                self,
                "保存预设",
                "请输入预设名称:",
                QLineEdit.Normal,
                ""
            )
            
            self.log_message(f"用户输入: name='{name}', ok={ok}", "INFO")
            
            if not ok or not name.strip():
                self.log_message("用户取消或名称为空", "INFO")
                return
            
            name = name.strip()
            
            # 检查名称是否已存在
            if name in self.joint_presets:
                reply = QMessageBox.question(
                    self,
                    "确认覆盖",
                    f"预设 '{name}' 已存在，是否覆盖？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
            
            # 保存预设
            self.joint_presets[name] = joint_list
            self.save_joint_presets()
            self.update_presets_list()
            
            self.log_message(f"已保存预设: {name}, 关节角度: {joint_list}")
            QMessageBox.information(self, "成功", f"预设 '{name}' 已保存")
            
        except Exception as e:
            error_msg = f"保存预设失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def load_preset_to_inputs(self):
        """加载选中的预设到关节输入框"""
        if not hasattr(self, 'presets_list') or self.presets_list is None:
            QMessageBox.warning(self, "错误", "预设列表未初始化")
            return
        
        selected_items = self.presets_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个预设")
            return
        
        preset_name = selected_items[0].text()
        if preset_name not in self.joint_presets:
            QMessageBox.warning(self, "错误", f"预设 '{preset_name}' 不存在")
            return
        
        try:
            joint_values = self.joint_presets[preset_name]
            if len(joint_values) != len(self.joint_spins):
                QMessageBox.warning(self, "错误", f"预设关节数量不匹配: 期望 {len(self.joint_spins)}, 得到 {len(joint_values)}")
                return
            
            # 加载到输入框
            for i, spin in enumerate(self.joint_spins):
                if i < len(joint_values):
                    spin.setValue(joint_values[i])
            
            self.log_message(f"已加载预设 '{preset_name}' 到输入框")
            QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已加载到输入框")
            
        except Exception as e:
            error_msg = f"加载预设失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
    
    def execute_selected_preset(self):
        """执行选中的预设"""
        if not hasattr(self, 'presets_list') or self.presets_list is None:
            QMessageBox.warning(self, "错误", "预设列表未初始化")
            return
        
        selected_items = self.presets_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个预设")
            return
        
        if not self.is_connected or not self.controller:
            QMessageBox.warning(self, "警告", "请先连接机械臂")
            return
        
        preset_name = selected_items[0].text()
        if preset_name not in self.joint_presets:
            QMessageBox.warning(self, "错误", f"预设 '{preset_name}' 不存在")
            return
        
        try:
            joint_values = np.array(self.joint_presets[preset_name])
            
            # 仿真预览（如果启用）
            if self.sim_preview_checkbox.isChecked() and SIMULATION_AVAILABLE:
                if not self.simulate_joint_motion(joint_values):
                    return  # 仿真失败，不执行实际运动
            
            # 执行运动
            speed = self.joint_speed_spin.value()
            block = self.block_checkbox.isChecked()
            
            self.log_message(f"执行预设 '{preset_name}': {joint_values}, 速度: {speed}, 阻塞: {block}")
            self.status_bar.showMessage(f"正在执行预设 '{preset_name}'...")
            
            self.controller.move_j(
                joint_angles=joint_values,
                speed=speed,
                is_radians=True,
                block=block
            )
            
            self.log_message(f"预设 '{preset_name}' 执行完成")
            self.status_bar.showMessage(f"预设 '{preset_name}' 执行完成")
            
        except Exception as e:
            error_msg = f"执行预设失败: {str(e)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(self, "错误", error_msg)
            self.status_bar.showMessage("执行预设失败")
    
    def delete_selected_preset(self):
        """删除选中的预设"""
        if not hasattr(self, 'presets_list') or self.presets_list is None:
            QMessageBox.warning(self, "错误", "预设列表未初始化")
            return
        
        selected_items = self.presets_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择一个预设")
            return
        
        preset_name = selected_items[0].text()
        
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除预设 '{preset_name}' 吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                del self.joint_presets[preset_name]
                self.save_joint_presets()
                self.update_presets_list()
                self.preset_info_label.setText("选择一个预设以查看详细信息")
                self.log_message(f"已删除预设: {preset_name}")
                QMessageBox.information(self, "成功", f"预设 '{preset_name}' 已删除")
            except Exception as e:
                error_msg = f"删除预设失败: {str(e)}"
                self.log_message(error_msg, "ERROR")
                QMessageBox.critical(self, "错误", error_msg)
    
    def on_preset_selection_changed(self):
        """预设选择改变时的回调"""
        if not hasattr(self, 'presets_list') or self.presets_list is None:
            return
        
        selected_items = self.presets_list.selectedItems()
        if not selected_items:
            if hasattr(self, 'load_preset_btn'):
                self.load_preset_btn.setEnabled(False)
            if hasattr(self, 'execute_preset_btn'):
                self.execute_preset_btn.setEnabled(False)
            if hasattr(self, 'delete_preset_btn'):
                self.delete_preset_btn.setEnabled(False)
            if hasattr(self, 'preset_info_label'):
                self.preset_info_label.setText("选择一个预设以查看详细信息")
            return
        
        preset_name = selected_items[0].text()
        if preset_name not in self.joint_presets:
            return
        
        # 启用按钮（检查属性是否存在）
        if hasattr(self, 'load_preset_btn'):
            self.load_preset_btn.setEnabled(True)
        if hasattr(self, 'execute_preset_btn'):
            self.execute_preset_btn.setEnabled(self.is_connected)
        if hasattr(self, 'delete_preset_btn'):
            self.delete_preset_btn.setEnabled(True)
        
        # 显示预设信息
        joint_values = self.joint_presets[preset_name]
        info_text = f"预设名称: {preset_name}\n\n关节角度 (弧度):\n"
        joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
        for i, (label, value) in enumerate(zip(joint_labels, joint_values)):
            if i < len(joint_values):
                info_text += f"{label}: {value:.4f}\n"
        if hasattr(self, 'preset_info_label'):
            self.preset_info_label.setText(info_text)
    
    def on_preset_double_clicked(self, item: QListWidgetItem):
        """双击预设时加载到输入框"""
        self.load_preset_to_inputs()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 关闭仿真窗口
        if self.sim_running:
            self.stop_simulation_window()
        
        if self.is_connected:
            reply = QMessageBox.question(
                self,
                "确认退出",
                "机械臂已连接，确定要退出吗？\n将自动断开连接并禁用所有电机。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.disconnect_robot()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = OpenArmGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

