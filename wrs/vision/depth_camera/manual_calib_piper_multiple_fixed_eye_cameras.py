"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate multiple fixed eye cameras
支持多个固定眼相机的标定
"""
__VERSION__ = '0.0.1'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod

from direct.task.TaskManagerGlobal import taskMgr

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri
import numpy as np


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


class ManualCalibrationBase(ABC):
    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 component_name="arm", move_resolution=.001, rotation_resolution=rm.np.radians(5)):
        """
        Class to manually calibrate the point cloud data
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix. If it is None, the init calibration matrix will be identity matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat = np.eye(4) if init_calib_mat is None else init_calib_mat
        self._component_name = component_name

        # variable stores robot plot and the point cloud plot
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd = None

        #
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # add task
        taskMgr.doMethodLater(2, self.sync_rbt, "sync rbt", )
        taskMgr.add(self.adjust, "manual adjust the mph")
        taskMgr.doMethodLater(1, self.sync_pcd, "sync mph", )

    @abstractmethod
    def get_pcd(self) -> np.ndarray:
        """
        An abstract method to get the point cloud
        :return: An Nx3 ndarray represents the point cloud
        """
        pass

    @abstractmethod
    def get_rbt_jnt_val(self) -> np.ndarray:
        """
        An abstract method to get the robot joint angles
        :return: 1xn ndarray, n is degree of freedom of the robot
        """
        pass

    @abstractmethod
    def align_pcd(self, pcd) -> np.ndarray:
        """
        Abstract method to align the mph according to the calibration matrix
        implement the Eye-in-hand or eye-to-hand transformation here
        https://support.zivid.com/en/latest/academy/applications/hand-eye/system-configurations.html
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by moving
        :param dir: The local move motion_vec based on the calibration matrix coordinate
        :param dir_global: The global move motion_vec based on the world coordinate
        :return:
        """
        # For fixed eye camera, we adjust the camera-to-world transformation directly
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by rotating
        :param dir: The local motion_vec of the calibration matrix
        :param dir_global: The global motion_vec
        :return:
        """
        # For fixed eye camera, we adjust the camera-to-world transformation directly
        self._init_calib_mat[:3, :3] = np.dot(rm.rotmat_from_axangle(dir_global, self.rotation_resolution),
                                              self._init_calib_mat[:3, :3])

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n'):
        def add_key(keys: str or list):
            """
            Add key to  the keymap. The default keymap can be seen in visualization/panda/inputmanager.py
            :param keys: the keys added to the keymap
            """
            assert isinstance(keys, str) or isinstance(keys, list)

            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw

    def sync_pcd(self, task):
        """
        Synchronize the real robot and the simulation robot
        :return: None
        """

        self._pcd = self.get_pcd()
        self.plot()
        self.save()
        return task.again

    def sync_rbt(self, task):
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(rbt_jnt_val, update=True)
        self.plot()
        return task.again

    def save(self):
        """
        Save manual calibration results
        :return:
        """
        dump_json({'affine_mat': self._init_calib_mat.tolist()}, "manual_calibration_fixed_eye.json", reminder=False)

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot
        :param task:
        :return:
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=.8)
        self._plot_node_rbt.attach_to(base)
        pcd = self._pcd
        if pcd is not None:
            if pcd.shape[1] == 6:
                pcd, pcd_color = pcd[:, :3], pcd[:, 3:6]
                pcd_color_rgba = np.append(pcd_color, np.ones((len(pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = np.array([1, 1, 1, 1])
            pcd_r = self.align_pcd(pcd)
            self._plot_node_pcd = mgm.gen_pointcloud(pcd_r, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
        if task is not None:
            return task.again

    def adjust(self, task):
        """
        Checks for keyboard input and adjusts the calibration matrix.
        This method is optimized to prevent lag by only plotting when a change occurs
        and by removing expensive I/O operations from the loop.
        """
        was_adjusted = False

        # --- Handle Translation ---
        # Use separate 'if' statements to allow for simultaneous key presses (e.g., diagonal movement)
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_')
            was_adjusted = True

        # --- Handle Rotation ---
        if base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 0], dir_global=np.array([1, 0, 0]), key_name='x_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 0], dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 1], dir_global=np.array([0, 1, 0]), key_name='y_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, 1]), key_name='z_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mat[:3, 2], dir_global=np.array([0, 0, -1]), key_name='z_ccw')
            was_adjusted = True

        # --- Update Scene only if something changed ---
        if was_adjusted:
            self.plot()
            # IMPORTANT: We have removed self.save() from this loop.
            # Call self.save() manually when you are done with the calibration.

        return task.again


class MultiFixedEyeManualCalib(ManualCalibrationBase):
    """
    多固定眼相机标定类
    支持同时标定多个固定在世界坐标系中的相机
    """
    def __init__(self, rbt_s, rbt_x, sensor_hdls, init_calib_mats, **kwargs):
        """
        :param sensor_hdls: list of camera handlers
        :param init_calib_mats: list of 4x4 calibration matrices for each camera
        """
        super().__init__(rbt_s, rbt_x, sensor_hdls[0], init_calib_mat=init_calib_mats[0], **kwargs)
        self._sensor_hdls = sensor_hdls
        self._init_calib_mats = init_calib_mats  # list of 4x4
        self._pcds = [None] * len(sensor_hdls)
        self._current_camera_idx = 0  # 当前正在标定的相机索引

    def get_pcd(self):
        """
        获取所有相机点云
        """
        pcds = []
        for i, sensor in enumerate(self._sensor_hdls):
            pcd, pcd_color, _, _ = sensor.get_pcd_texture_depth()
            pcds.append(np.hstack((pcd, pcd_color)))
        return pcds

    def get_rbt_jnt_val(self):
        return self._rbt_x.get_joint_values()

    def align_pcd(self, pcds):
        """
        将多个固定相机的点云对齐到世界坐标
        """
        aligned_pcds = []
        for i, pcd in enumerate(pcds):
            # 对于固定眼相机，标定矩阵直接是相机到世界的变换
            c2w_mat = self._init_calib_mats[i]
            pcd_w = rm.transform_points_by_homomat(c2w_mat, points=pcd)
            aligned_pcds.append(pcd_w)
        return aligned_pcds

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        调整当前选中相机的标定矩阵
        """
        self._init_calib_mats[self._current_camera_idx][:3, 3] = \
            self._init_calib_mats[self._current_camera_idx][:3, 3] + dir_global * self.move_resolution

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        调整当前选中相机的标定矩阵
        """
        self._init_calib_mats[self._current_camera_idx][:3, :3] = np.dot(
            rm.rotmat_from_axangle(dir_global, self.rotation_resolution),
            self._init_calib_mats[self._current_camera_idx][:3, :3])

    def switch_camera(self, camera_idx):
        """
        切换当前标定的相机
        """
        if 0 <= camera_idx < len(self._sensor_hdls):
            self._current_camera_idx = camera_idx
            self._sensor_hdl = self._sensor_hdls[camera_idx]
            self._init_calib_mat = self._init_calib_mats[camera_idx]
            print(f"切换到相机 {camera_idx}")

    def save(self):
        """
        保存所有相机的标定结果
        """
        calibration_data = {
            'camera_calibrations': []
        }
        for i, calib_mat in enumerate(self._init_calib_mats):
            calibration_data['camera_calibrations'].append({
                'camera_id': i,
                'affine_mat': calib_mat.tolist()
            })
        dump_json(calibration_data, "manual_calibration_multi_fixed_eye.json", reminder=False)

    def plot(self, task=None):
        """
        同时绘制机器人和多个固定相机点云
        """
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()

        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=.8)
        self._plot_node_rbt.attach_to(base)

        # 获取点云并绘制
        self._pcds = self.get_pcd()
        aligned_pcds = self.align_pcd(self._pcds)

        # 定义多种颜色
        colors = [
            np.array([1, 0, 0, 0.5]),  # 红色
            np.array([0, 1, 0, 0.5]),  # 绿色
            np.array([0, 0, 1, 0.5]),  # 蓝色
            np.array([1, 1, 0, 0.5]),  # 黄色
            np.array([1, 0, 1, 0.5]),  # 紫色
            np.array([0, 1, 1, 0.5]),  # 青色
        ]

        self._plot_node_pcd = mgm.gen_frame()
        for i, pcd in enumerate(aligned_pcds):
            pcd_color_rgba = colors[i % len(colors)]
            pcd_node = mgm.gen_pointcloud(pcd, rgba=pcd_color_rgba)
            # 为当前标定的相机添加坐标系
            if i == self._current_camera_idx:
                mgm.gen_frame(self._init_calib_mats[i][:3, 3],
                              self._init_calib_mats[i][:3, :3], ax_length=0.1).attach_to(pcd_node)
            pcd_node.attach_to(self._plot_node_pcd)

        self._plot_node_pcd.attach_to(base)
        if task is not None:
            return task.again

    def map_key(self, x='w', x_='s', y='a', y_='d', z='q', z_='e', x_cw='z', x_ccw='x', y_cw='c', y_ccw='v', z_cw='b',
                z_ccw='n', switch_cam='t'):
        """
        添加相机切换键
        """
        def add_key(keys: str or list):
            assert isinstance(keys, str) or isinstance(keys, list)
            if isinstance(keys, str):
                keys = [keys]

            def set_keys(base, k, v):
                base.inputmgr.keymap[k] = v

            for key in keys:
                if key in base.inputmgr.keymap: continue
                base.inputmgr.keymap[key] = False
                base.inputmgr.accept(key, set_keys, [base, key, True])
                base.inputmgr.accept(key + '-up', set_keys, [base, key, False])

        add_key([x, x_, y, y_, z, z_, x_cw, x_ccw, y_cw, y_ccw, z_cw, z_ccw, switch_cam])
        self._key['x'] = x
        self._key['x_'] = x_
        self._key['y'] = y
        self._key['y_'] = y_
        self._key['z'] = z
        self._key['z_'] = z_
        self._key['x_cw'] = x_cw
        self._key['x_ccw'] = x_ccw
        self._key['y_cw'] = y_cw
        self._key['y_ccw'] = y_ccw
        self._key['z_cw'] = z_cw
        self._key['z_ccw'] = z_ccw
        self._key['switch_cam'] = switch_cam

    def adjust(self, task):
        """
        处理键盘输入，包括相机切换
        """
        was_adjusted = False

        # 处理相机切换
        if base.inputmgr.keymap[self._key['switch_cam']]:
            self._current_camera_idx = (self._current_camera_idx + 1) % len(self._sensor_hdls)
            self.switch_camera(self._current_camera_idx)
            was_adjusted = True

        # --- Handle Translation ---
        if base.inputmgr.keymap[self._key['x']]:
            self.move_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 0], 
                           dir_global=np.array([1, 0, 0]), key_name='x')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_']]:
            self.move_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 0], 
                           dir_global=np.array([-1, 0, 0]), key_name='x_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y']]:
            self.move_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 1], 
                           dir_global=np.array([0, 1, 0]), key_name='y')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_']]:
            self.move_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 1], 
                           dir_global=np.array([0, -1, 0]), key_name='y_')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z']]:
            self.move_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 2], 
                           dir_global=np.array([0, 0, 1]), key_name='z')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_']]:
            self.move_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 2], 
                           dir_global=np.array([0, 0, -1]), key_name='z_')
            was_adjusted = True

        # --- Handle Rotation ---
        if base.inputmgr.keymap[self._key['x_cw']]:
            self.rotate_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 0], 
                             dir_global=np.array([1, 0, 0]), key_name='x_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['x_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 0], 
                             dir_global=np.array([-1, 0, 0]), key_name='x_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_cw']]:
            self.rotate_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 1], 
                             dir_global=np.array([0, 1, 0]), key_name='y_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['y_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 1], 
                             dir_global=np.array([0, -1, 0]), key_name='y_ccw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_cw']]:
            self.rotate_adjust(dir=self._init_calib_mats[self._current_camera_idx][:3, 2], 
                             dir_global=np.array([0, 0, 1]), key_name='z_cw')
            was_adjusted = True
        if base.inputmgr.keymap[self._key['z_ccw']]:
            self.rotate_adjust(dir=-self._init_calib_mats[self._current_camera_idx][:3, 2], 
                             dir_global=np.array([0, 0, -1]), key_name='z_ccw')
            was_adjusted = True

        # --- Update Scene only if something changed ---
        if was_adjusted:
            self.plot()

        return task.again


def load_calibration_matrix_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    mat_list = data.get('affine_mat', None)
    if mat_list is None:
        raise ValueError("JSON 文件中没有找到 'affine_mat' 字段")
    return np.array(mat_list)


def load_multi_camera_calibration_from_json(filepath):
    """
    加载多相机标定数据
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    camera_calibrations = data.get('camera_calibrations', [])
    calib_mats = []
    for calib_data in camera_calibrations:
        calib_mats.append(np.array(calib_data['affine_mat']))
    return calib_mats


# middle:
#   ID: '243322073422'
# left:
#   ID: '243322074546'
# right:
#   ID: '243322071033'
if __name__ == "__main__":
    """
    标定多个固定眼相机，相机固定在世界坐标系中观察机器人工作区域
    """
    import numpy as np
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.robot_con.piper.piper import PiperArmController
    from wrs.robot_sim.manipulators.piper.piper import Piper
    import time

    base = wd.World(cam_pos=[0, 2, 0], lookat_pos=[0, 0, 0], lens_type=2)
    
    # 加载机器人
    rbtx = PiperArmController(can_name="can0", has_gripper=True)
    rbt = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0), name='piper')

    # 多个固定眼相机
    cam_ids = ['243322074546', '243322073422', '243322071033']  # 左、中、右相机
    sensors = [RealSenseD400(device=cid) for cid in cam_ids]
    
    # 预热相机
    for sensor in sensors:
        sensor.get_pcd_texture_depth()
        sensor.get_pcd_texture_depth()

    # 读取初始标定矩阵（如果存在）
    try:
        init_mats = load_multi_camera_calibration_from_json("/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/vision/depth_camera/manual_calibration_multi_fixed_eye.json")
    except:
        # 如果没有现有标定文件，使用初始变换矩阵
        init_mats = []
        for i in range(len(cam_ids)):
            # 假设相机在不同位置观察工作区域
            if i == 0:  # 左相机
                init_mat = np.array([
                    [1, 0, 0, -0.5],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]
                ])
            elif i == 1:  # 中相机
                init_mat = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]
                ])
            else:  # 右相机
                init_mat = np.array([
                    [1, 0, 0, 0.5],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1]
                ])
            init_mats.append(init_mat)

    mgm.gen_frame(ax_length=1).attach_to(base)

    # 创建多固定眼相机标定对象
    multi_fixed_eye_mc = MultiFixedEyeManualCalib(
        rbt_s=rbt, 
        rbt_x=rbtx, 
        sensor_hdls=sensors,
        init_calib_mats=init_mats,
        move_resolution=0.01,  # 1cm resolution
        rotation_resolution=np.radians(5)  # 5 degree resolution
    )
    
    print("多固定眼相机标定说明：")
    print("T: 切换当前标定的相机")
    print("W/S: 当前相机X轴移动")
    print("A/D: 当前相机Y轴移动") 
    print("Q/E: 当前相机Z轴移动")
    print("Z/X: 当前相机绕X轴旋转")
    print("C/V: 当前相机绕Y轴旋转")
    print("B/N: 当前相机绕Z轴旋转")
    print("标定完成后按Ctrl+C退出，标定结果会自动保存")

    base.run()



