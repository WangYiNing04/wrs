import os
import cv2
import time
import yaml
import threading
import queue
from datetime import datetime
from pathlib import Path
import numpy as np
from pynput import keyboard
from wrs.drivers.devices.realsense.realsense_d400s import *
from ultralytics import YOLO  # 导入YOLO模型
from typing import Optional, List, Tuple
import wrs.basis.robot_math as rm

class RealTimeYOLODetector:
    """实时YOLO检测器，支持多相机同时预览和YOLO实时推理"""
    
    def __init__(self, config_path='/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/config/camera_correspondence.yaml', 
                 yolo_model_path='yolov8n.pt',check_3D_keypoints=False):
        """
        初始化实时检测器
        
        Args:
            config_path: 相机配置文件路径
            yolo_model_path: YOLO模型路径
        """
        self.config_path = config_path
        self.rs_pipelines = {}
        self.detection_active = False
        self.key_press_times = {}
        self.key_repeat_threshold = 0.1  # 100ms内重复按键视为重复
        self.check_3D_keypoints = check_3D_keypoints
        
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        print(f"已加载YOLO模型: {yolo_model_path}")
        
        # 初始化相机
        self.initialize_cameras()

        if self.check_3D_keypoints:
            print("3D关键点检测已启用")

            #middle camera hand-eye matrix
            self._init_calib_mat = np.array([[0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
                                             [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
                                             [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
                                             [0.0, 0.0, 0.0, 1.0]]
                                            )

    def initialize_cameras(self):
        """初始化相机"""
        try:
            # 读取YAML配置文件
            with open(self.config_path, 'r') as file:
                camera_config = yaml.safe_load(file)

            # 从配置中提取相机ID
            camera_roles = {
                'middle': camera_config['middle_camera']['ID'],
                #'left': camera_config['left_camera']['ID'],
                #'right': camera_config['right_camera']['ID']
            }

            # 查找实际连接的设备
            available_serials, ctx = find_devices()
            print("检测到设备:", available_serials)

            # 初始化相机（用字典存储，键为角色名称）
            for role, cam_id in camera_roles.items():
                if cam_id in available_serials:
                    print(f"正在初始化 {role} 相机 (ID: {cam_id})")
                    pipeline = RealSenseD400(device=cam_id)
                    pipeline.reset()
                    time.sleep(2)
                    pipeline = RealSenseD400(device=cam_id)  # 重新初始化
                    self.rs_pipelines[role] = pipeline
                    print(f"{role} 相机初始化成功")
                else:
                    print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")
                    
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise
    
    def run_yolo_inference(self, image):
        """
        使用YOLO模型进行推理
        
        Args:
            image: 输入图像
            
        Returns:
            annotated_image: 带标注的图像
            results: 推理结果
        """
        if image is None:
            return None, None
            
        # 运行推理
        results = self.yolo_model(image)

    
        # 渲染结果
        annotated_image = results[0].plot(
            boxes=True,
            labels=True,
        )
        
        return annotated_image, results
    
    def start_detection_mode(self):
        """
        启动检测模式，支持实时预览和键盘控制
        控制说明：
        - ESC键：退出程序
        """
        print("启动实时YOLO检测模式...")
        print("控制说明：")
        print("- ESC键：退出程序")
        
        self.detection_active = True
        
  
        for role, pipeline in self.rs_pipelines.items():
            try:
                pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                # 使用YOLO进行推理
                annotated_img, results = self.run_yolo_inference(color_img)
                K =  pipeline.intr_mat
                #world2cam
                #pcd = self.align_pcd(pcd)

                keypoints = results[0].keypoints.xy
                if len(keypoints) >= 2:
                    cup_kp = keypoints[0]
                    coaster_kp = keypoints[1]
                    print(f"杯子关键点: {cup_kp}")
                    print(f"杯垫关键点: {coaster_kp}")
                
                
                point_3d = pcd[v, u]  # (x, y, z)
                pcd_3d_world = self.align_pcd(point_3d)

                print(pcd_3d_world)

            
       
                points = keypoints.cpu().numpy().reshape(-1, 2).tolist()
                
                #将2Dkeypoint转为3Dkeypoint
                keypoints_xyzkey = self.keypointsTo3D(pcd_tf=pcd,keypoints=points,color_img=color_img)
       
                print(keypoints_xyzkey)
            except Exception as e:
                print(f"从 {role} 相机获取数据失败: {e}")
               
    
    def estimate_point_from_neighborhood(
            self,
            target_pixel: np.ndarray,
            pcd_matrix: np.ndarray,
            neighborhood_size: int = 5,
            outlier_std_threshold: float = 2.0
    ) -> Optional[np.ndarray]:
        """
        Estimates the 3D point for a target pixel by analyzing its neighborhood.

        Args:
            target_pixel (np.ndarray): The (x, y) coordinate of the target pixel.
            pcd_matrix (np.ndarray): The entire point cloud matrix with shape (h, w, 3).
            neighborhood_size (int): The size of the square window (e.g., 5 for a 5x5 window).
            outlier_std_threshold (float): The number of standard deviations to use for outlier removal.

        Returns:
            Optional[np.ndarray]: The estimated 3D point (x, y, z) or None if estimation is not possible.
        """
        # Ensure neighborhood size is odd to have a central pixel
        if neighborhood_size % 2 == 0:
            neighborhood_size += 1

        h, w = pcd_matrix.shape[:2]
        px, py = target_pixel
        half_size = neighborhood_size // 2

        # 1. Extract the pixels surrounding the target pixels.
        # Define the bounding box for the neighborhood, clamping to image dimensions
        x_min = max(0, px - half_size)
        x_max = min(w - 1, px + half_size)
        y_min = max(0, py - half_size)
        y_max = min(h - 1, py + half_size)
   
        # 2. Acquire their point clouds
        neighborhood_points = pcd_matrix[
                                    int(y_min):int(y_max) + 1,
                                    int(x_min):int(x_max) + 1
                                ].reshape(-1, 3)


        neighborhood_points = np.array(neighborhood_points, dtype=float)
        # Filter out invalid (0,0,0) points which are common due to sensor noise or disparity errors
        print(11)
        mask = np.any(neighborhood_points != 0, axis=1)
        valid_points = neighborhood_points[mask]

        # If no valid points are found in the neighborhood, we cannot estimate.
        if len(valid_points) == 0:
            return None

        # 3. Remove outliers
        if len(valid_points) > 3:  # Need enough points to calculate statistics
            mean = np.mean(valid_points, axis=0)
            std = np.std(valid_points, axis=0)

            # Avoid division by zero if standard deviation is null for an axis
            std[std == 0] = 1e-6

            # Calculate Z-scores for each point on each axis
            z_scores = np.abs((valid_points - mean) / std)

            # Keep points where the Z-score for all axes is below the threshold
            inliers = valid_points[np.all(z_scores < outlier_std_threshold, axis=1)]

            # If all points were considered outliers, fall back to using all valid points
            if len(inliers) == 0:
                inliers = valid_points
        else:
            # Not enough points for robust outlier detection, use all valid points
            inliers = valid_points

        # 4. Get the average to calculate the point cloud
        if len(inliers) > 0:
            estimated_point = np.mean(inliers, axis=0)
            return estimated_point
        else:
            return None

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 停止所有相机
        for pipeline in self.rs_pipelines.values():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"停止相机时出错: {e}")
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("程序已退出")

    def keypointsTo3D(self,
                    keypoints: np.ndarray,
                    color_img: np.ndarray,
                    pcd_tf: np.ndarray,
                    neighborhood_size: int = 5,
                    ) -> Optional[np.ndarray]:
        """
        Detects keypoints and retrieves their 3D world coordinates from a point cloud.
        If a point is invalid (0,0,0), it estimates it from its neighborhood.

        Args:
            yolo_model (YOLO): The trained YOLO model.
            color_img (np.ndarray): The RGB color image.
            pcd_matrix (np.ndarray): The point cloud data, shaped (h, w, 3).
            toggle_image (bool): If True, displays the detected keypoints on the image.
            neighborhood_size (int): The window size for neighborhood analysis.

        Returns:
            Optional[List[np.ndarray]]: A list of detected 3D points (x,y,z).
        """
        #得到2Dkeypoint
        pixels_coord = keypoints
        pcd_matrix = pcd_tf.reshape(color_img.shape[0], color_img.shape[1], 3)
        detected_points_xyz = []
        for p in pixels_coord:
            # IMPORTANT: Corrected indexing from (x,y) to array's (row, col) which is (y,x)
            estimated_xyz = self.estimate_point_from_neighborhood(
                target_pixel=p,
                pcd_matrix=pcd_matrix,
                neighborhood_size=neighborhood_size
            )
            if estimated_xyz is not None:
                print(f"  -> Estimated point: {estimated_xyz}")
                detected_points_xyz.append(estimated_xyz)
            else:
                print(f"  -> Estimation failed for pixel ({p[0]},{p[1]}). Skipping.")
                # Optionally, you could append a placeholder like np.array([0,0,0])
                # or just skip it as is currently done.
        return np.asarray(detected_points_xyz)
    
    def align_pcd(self, pcd):
        c2w_mat = self._init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
   



def main():
    """主函数"""
    try:
        # 创建检测器
        detector = RealTimeYOLODetector(
            yolo_model_path='/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/model/empty_cup_place/best.pt',  # 可以替换为您自己的模型路径
            check_3D_keypoints=True
        )
        
        # 启动检测模式
        #detector.start_detection_mode()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


    import numpy as np

    def pixel_to_world(u, v, depth, K, extrinsic):
        """
        根据像素坐标和深度，计算该点在世界坐标系中的位置

        参数:
            u, v: 像素坐标
            depth: 深度值 (Zc)
            K: 3x3 相机内参矩阵
            extrinsic: 4x4 相机外参矩阵 (T_cam_world)

        返回:
            world_point: 世界坐标 (Xw, Yw, Zw)
        """
        # 内参逆矩阵
        K_inv = np.linalg.inv(K)

        # 像素坐标转相机坐标
        pixel_h = np.array([u, v, 1.0])
        cam_point = depth * (K_inv @ pixel_h)

        # 转为齐次坐标
        cam_point_h = np.append(cam_point, 1.0)

        # 相机->世界坐标 (注意外参为 T_cam_world, 所以取逆)
        T_world_cam = np.linalg.inv(extrinsic)
        world_point_h = T_world_cam @ cam_point_h

        # 去齐次
        world_point = world_point_h[:3] / world_point_h[3]

        return world_point


    # 示例：
  


    # extrinsic = np.array([[0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
    #                                          [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
    #                                          [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
    #                                          [0.0, 0.0, 0.0, 1.0]]
    #                                         )

    # u, v = 320, 240  # 图像中心
    # depth = 1.2      # 深度为1.2米

    # world_point = pixel_to_world(u, v, depth, K, extrinsic)
    # print("World point:", world_point)
    for role, pipeline in detector.rs_pipelines.items():

        pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
        # 使用YOLO进行推理
        annotated_img, results = detector.run_yolo_inference(color_img)
        print(results[0].keypoints)
        keypoints = results[0].keypoints.xy

        keypoints = keypoints.cpu().numpy().reshape(-1, 2).tolist()
        print(keypoints)
        if len(keypoints) >= 2:
            cup_kp = keypoints[0]
            coaster_kp = keypoints[1]
            print(f"杯子关键点: {cup_kp}")
            print(f"杯垫关键点: {coaster_kp}")
        else:
            raise ValueError("未检测到足够的的关键点（需要至少2个）")
        u = coaster_kp[0] 
        v = coaster_kp[1]
        point_3d = pcd[v, u]  # (x, y, z)
        pcd_3d_world = detector.align_pcd(point_3d)

        print(pcd_3d_world)


if __name__ == "__main__":
    main()