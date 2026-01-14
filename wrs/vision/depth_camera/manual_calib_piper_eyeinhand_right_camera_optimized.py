"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate the camera - OPTIMIZED VERSION
Performance optimizations:
1. GPU-accelerated point cloud processing
2. Point cloud downsampling
3. Asynchronous processing
4. Reduced visualization frequency
5. Cached transformations
"""
__VERSION__ = '0.0.2'

import os
from pathlib import Path
import json
from abc import ABC, abstractmethod
import threading
import time
from collections import deque
import queue

from direct.task.TaskManagerGlobal import taskMgr

import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.robot_sim.robots.robot_interface as ri
import numpy as np

# GPU acceleration imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration available with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available, using CPU processing")

# Open3D for efficient point cloud processing
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
    print("Open3D available for point cloud processing")
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available, using basic processing")


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


class PointCloudProcessor:
    """GPU-accelerated point cloud processing"""
    
    def __init__(self, downsample_ratio=0.1, use_gpu=True):
        self.downsample_ratio = downsample_ratio
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.cached_transforms = {}
        
    def downsample_pointcloud(self, points, colors=None):
        """Downsample point cloud to reduce data size"""
        if len(points) == 0:
            return points, colors
            
        # Calculate number of points to keep
        num_points = len(points)
        target_points = max(1000, int(num_points * self.downsample_ratio))
        
        if target_points >= num_points:
            return points, colors
            
        # Random sampling
        indices = np.random.choice(num_points, target_points, replace=False)
        
        if colors is not None:
            return points[indices], colors[indices]
        return points[indices], None
    
    def transform_points_gpu(self, points, transform_matrix):
        """GPU-accelerated point transformation"""
        if not self.use_gpu or len(points) == 0:
            return rm.transform_points_by_homomat(transform_matrix, points)
        
        try:
            # Convert to homogeneous coordinates
            points_homo = np.hstack([points, np.ones((len(points), 1))])
            
            # Move to GPU
            points_gpu = cp.asarray(points_homo)
            transform_gpu = cp.asarray(transform_matrix)
            
            # Transform on GPU
            transformed_gpu = cp.dot(points_gpu, transform_gpu.T)
            
            # Move back to CPU
            transformed = cp.asnumpy(transformed_gpu)
            
            return transformed[:, :3]
        except Exception as e:
            print(f"GPU transformation failed, falling back to CPU: {e}")
            return rm.transform_points_by_homomat(transform_matrix, points)
    
    def process_pointcloud(self, points, colors=None, transform_matrix=None):
        """Process point cloud with downsampling and optional transformation"""
        # Downsample
        points_ds, colors_ds = self.downsample_pointcloud(points, colors)
        
        # Transform if matrix provided
        if transform_matrix is not None:
            points_ds = self.transform_points_gpu(points_ds, transform_matrix)
        
        return points_ds, colors_ds


class AsyncPointCloudCapture:
    """Asynchronous point cloud capture to prevent blocking"""
    
    def __init__(self, sensor_hdl, max_queue_size=3):
        self.sensor_hdl = sensor_hdl
        self.max_queue_size = max_queue_size
        self.pcd_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = None
        self.latest_pcd = None
        self.latest_pcd_lock = threading.Lock()
        
    def start(self):
        """Start the capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop the capture thread"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.running:
            try:
                # Get point cloud from sensor
                pcd, pcd_color, _, _ = self.sensor_hdl.get_pcd_texture_depth()
                
                # Update latest point cloud
                with self.latest_pcd_lock:
                    self.latest_pcd = (pcd, pcd_color)
                
                # Add to queue (non-blocking)
                try:
                    self.pcd_queue.put_nowait((pcd, pcd_color))
                except queue.Full:
                    # Remove oldest if queue is full
                    try:
                        self.pcd_queue.get_nowait()
                        self.pcd_queue.put_nowait((pcd, pcd_color))
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                print(f"Error in point cloud capture: {e}")
                time.sleep(0.1)
    
    def get_latest_pcd(self):
        """Get the latest point cloud data"""
        with self.latest_pcd_lock:
            return self.latest_pcd
    
    def get_pcd_from_queue(self):
        """Get point cloud from queue (non-blocking)"""
        try:
            return self.pcd_queue.get_nowait()
        except queue.Empty:
            return None


class OptimizedManualCalibrationBase(ABC):
    def __init__(self, rbt_s: ri.RobotInterface, rbt_x, sensor_hdl, init_calib_mat: rm.np.ndarray = None,
                 component_name="arm", move_resolution=.001, rotation_resolution=rm.np.radians(5),
                 downsample_ratio=0.1, visualization_fps=10):
        """
        Optimized manual calibration base class
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        :param downsample_ratio: ratio for point cloud downsampling (0.1 = keep 10%)
        :param visualization_fps: target FPS for visualization updates
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
        self._pcd_color = None

        # Performance optimizations
        self.pcd_processor = PointCloudProcessor(downsample_ratio=downsample_ratio, use_gpu=True)
        self.async_capture = AsyncPointCloudCapture(sensor_hdl)
        self.visualization_interval = 1.0 / visualization_fps
        self.last_visualization_time = 0
        
        # Caching
        self._cached_robot_pose = None
        self._cached_transform_matrix = None
        self._transform_cache_valid = False

        #
        self._key = {}
        self.map_key()
        self.move_resolution = move_resolution
        self.rotation_resolution = rotation_resolution

        # Start async capture
        self.async_capture.start()

        # add task with reduced frequency
<<<<<<< HEAD
        taskMgr.doMethodLater(5, self.sync_rbt, "sync rbt", )
        taskMgr.add(self.adjust, "manual adjust the mph")
        taskMgr.doMethodLater(5, self.sync_pcd, "sync mph", )
=======
        taskMgr.doMethodLater(2, self.sync_rbt, "sync rbt", )
        taskMgr.add(self.adjust, "manual adjust the mph")
        taskMgr.doMethodLater(1.0/visualization_fps, self.sync_pcd, "sync mph", )
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

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
        :return: An Nx3 ndarray represents the aligned point cloud
        """
        pass

    def move_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by moving
        """
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        r2w_mat = np.linalg.inv(w2r_mat)
        dir_in_robot_frame = r2w_mat[:3, :3].dot(dir_global)
        self._init_calib_mat[:3, 3] = self._init_calib_mat[:3, 3] + dir_in_robot_frame * self.move_resolution
        self._invalidate_transform_cache()

    def rotate_adjust(self, dir, dir_global, key_name=None):
        """
        The abstract method to revise the calibration matrix by rotating
        """
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        R_w_r = w2r_mat[:3, :3]  # Rotation from Robot frame to World frame
        delta_R_world = rm.rotmat_from_axangle(dir_global, np.radians(self.rotation_resolution))
        R_r_w = R_w_r.T
        delta_R_robot = R_r_w @ delta_R_world @ R_w_r
        current_R_r_c = self._init_calib_mat[:3, :3]
        self._init_calib_mat[:3, :3] = np.dot(delta_R_robot, current_R_r_c)
        self._invalidate_transform_cache()

    def _invalidate_transform_cache(self):
        """Invalidate cached transformation matrix"""
        self._transform_cache_valid = False

    def _get_cached_transform_matrix(self):
        """Get cached transformation matrix or compute if invalid"""
        if not self._transform_cache_valid or self._cached_robot_pose is None:
            rbt_pose = self._rbt_x.get_pose()
            if self._cached_robot_pose is None or not np.allclose(rbt_pose[0], self._cached_robot_pose[0]) or \
               not np.allclose(rbt_pose[1], self._cached_robot_pose[1]):
                self._cached_robot_pose = rbt_pose
                w2r_mat = rm.homomat_from_posrot(*rbt_pose)
                self._cached_transform_matrix = w2r_mat.dot(self._init_calib_mat)
                self._transform_cache_valid = True
        return self._cached_transform_matrix

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
        Synchronize the real robot and the simulation robot - OPTIMIZED
        """
        current_time = time.time()
        
        # Throttle visualization updates
        if current_time - self.last_visualization_time < self.visualization_interval:
            return task.again
            
        # Get latest point cloud from async capture
        pcd_data = self.async_capture.get_latest_pcd()
        if pcd_data is None:
            return task.again
            
        pcd, pcd_color = pcd_data
        
        # Process point cloud with optimizations
        pcd_processed, pcd_color_processed = self.pcd_processor.process_pointcloud(
            pcd, pcd_color, self._get_cached_transform_matrix()
        )
        
        self._pcd = pcd_processed
        self._pcd_color = pcd_color_processed
        self.last_visualization_time = current_time
        
        self.plot()
        self.save()
        return task.again

    def sync_rbt(self, task):
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(rbt_jnt_val, update=True)
        self._invalidate_transform_cache()  # Invalidate cache when robot moves
        self.plot()
        return task.again

    def save(self):
        """
        Save manual calibration results - OPTIMIZED (less frequent saves)
        """
        # Only save every 2 seconds to reduce I/O
        current_time = time.time()
        if not hasattr(self, '_last_save_time') or current_time - self._last_save_time > 2.0:
            dump_json({'affine_mat': self._init_calib_mat.tolist()}, 
                     "manual_calibration_realman_optimized.json", reminder=False)
            self._last_save_time = current_time

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot - OPTIMIZED
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
            
        self._plot_node_rbt = self._rbt_s.gen_meshmodel(alpha=.8)
        self._plot_node_rbt.attach_to(base)
        
        if self._pcd is not None and len(self._pcd) > 0:
            if self._pcd_color is not None and len(self._pcd_color) > 0:
                pcd_color_rgba = np.append(self._pcd_color, np.ones((len(self._pcd_color), 1)), axis=1)
            else:
                pcd_color_rgba = np.array([1, 1, 1, 1])
                
            self._plot_node_pcd = mgm.gen_pointcloud(self._pcd, rgba=pcd_color_rgba)
            mgm.gen_frame(self._init_calib_mat[:3, 3], self._init_calib_mat[:3, :3]).attach_to(self._plot_node_pcd)
            self._plot_node_pcd.attach_to(base)
            
        if task is not None:
            return task.again

    def adjust(self, task):
        """
        Checks for keyboard input and adjusts the calibration matrix - OPTIMIZED
        """
        was_adjusted = False

        # --- Handle Translation ---
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
            self.move_adjust(dir=-self._init_calib_mat[:3, 1], dir_global=np.array([0, -1, 0]), key_name='y')
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

        return task.again

    def cleanup(self):
        """Cleanup resources"""
        self.async_capture.stop()


class OptimizedRealmanManualCalib(OptimizedManualCalibrationBase):
    """
    Optimized Eye in hand example
    """

    def get_pcd(self):
        """Get point cloud from async capture"""
        pcd_data = self.async_capture.get_latest_pcd()
        if pcd_data is None:
            return np.array([])
        pcd, pcd_color = pcd_data
        return np.hstack((pcd, pcd_color))

    def get_rbt_jnt_val(self):
        return self._rbt_x.get_joint_values()

    def align_pcd(self, pcd):
        """Use cached transformation matrix for efficiency"""
        transform_matrix = self._get_cached_transform_matrix()
        return self.pcd_processor.transform_points_gpu(pcd, transform_matrix)


def load_calibration_matrix_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    mat_list = data.get('affine_mat', None)
    if mat_list is None:
        raise ValueError("JSON 文件中没有找到 'affine_mat' 字段")
    return np.array(mat_list)


# middle:
#   ID: '243322073422'
# left:
#   ID: '243322074546'
# right:
#   ID: '243322071033'
if __name__ == "__main__":

    '''
    标定右臂相机，相机看向左臂 - OPTIMIZED VERSION
    '''
    import numpy as np
    import wrs.visualization.panda.world as wd
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.robot_con.piper.piper import PiperArmController
    from wrs.robot_sim.manipulators.piper.piper import Piper
    import time

    base = wd.World(cam_pos=[0, 2, 0], lookat_pos=[0, 0, 0], lens_type=2)
    
    #加载左臂
<<<<<<< HEAD
    #rbtx_left = PiperArmController(can_name="can0", has_gripper=True)
=======
    rbtx_left = PiperArmController(can_name="can0", has_gripper=True)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    
    #加载右臂
    rbtx_right = PiperArmController(can_name="can1", has_gripper=True)

    #左臂base与世界坐标系重合
<<<<<<< HEAD
    #rbt_left = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0),pos=[0, 0.6, 0], name='piper_left')
=======
    rbt_left = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0), name='piper_left')
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
    
    #右臂在world Y轴 -0.6m处
    rbt_right = Piper(enable_cc=True, rotmat=rm.rotmat_from_euler(0, 0, 0),pos=[0, -0.6, 0], name='piper_right')

    # 读取标定矩阵
    try:
        init_mat = load_calibration_matrix_from_json("/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/vision/depth_camera/manual_calibration_piper_right.json")
    except:
        print("No existing calibration found, using identity matrix")
        init_mat = np.eye(4)

    mgm.gen_frame(ax_length=1).attach_to(base)

    # 右臂相机id
    cam_ids = '243322071033'

    rs_pipe = RealSenseD400(device=cam_ids)
    rs_pipe.get_pcd_texture_depth()  # warm-up
    rs_pipe.get_pcd_texture_depth()

    # Create optimized calibration instance
    xarm_mc = OptimizedRealmanManualCalib(
        rbt_s=rbt_right, 
        rbt_x=rbtx_right, 
        sensor_hdl=rs_pipe,
        init_calib_mat=init_mat,
        move_resolution=0.05,
        rotation_resolution=np.radians(30),
        downsample_ratio=0.1,  # Keep 10% of points
        visualization_fps=10    # 10 FPS visualization
    )
    
<<<<<<< HEAD
    # # 同步左臂的任务
    # left_arm = None

    # def sync_left_arm(task):
    #     global left_arm
    #     if left_arm is not None:
    #         left_arm.detach()
    #     rbt_jnt_val = rbtx_left.get_joint_values()
    #     rbt_left.fk(rbt_jnt_val, update=True)
    #     left_arm = rbt_left.gen_meshmodel(alpha=1).attach_to(base)
    #     return task.again

    # taskMgr.doMethodLater(1, sync_left_arm, "sync left arm")
=======
    # 同步左臂的任务
    left_arm = None

    def sync_left_arm(task):
        global left_arm
        if left_arm is not None:
            left_arm.detach()
        rbt_jnt_val = rbtx_left.get_joint_values()
        rbt_left.fk(rbt_jnt_val, update=True)
        left_arm = rbt_left.gen_meshmodel(alpha=1).attach_to(base)
        return task.again

    taskMgr.doMethodLater(0.01, sync_left_arm, "sync left arm")
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

    try:
        base.run()
    finally:
        # Cleanup resources
        xarm_mc.cleanup()
