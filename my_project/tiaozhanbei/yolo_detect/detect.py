import os
import cv2
import time
import yaml
import threading
import queue
from datetime import datetime
from pathlib import Path
import numpy as np
from wrs.drivers.devices.realsense.realsense_d400s import *
from ultralytics import YOLO  # 导入YOLO模型
from typing import Optional, List, Tuple
import wrs.basis.robot_math as rm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< HEAD
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
from wrs import wd, rm, mgm, mcm, ppp, gg, gpa
=======
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

def cam_to_world(cam_point, T_cam_world):
    cam_point_h = np.append(cam_point, 1)
    world_point_h = T_cam_world @ cam_point_h
    return world_point_h[:3] / world_point_h[3]

def pixel_to_3d(u, v, depth_image, K):
    """
    将图像像素点(u, v)转换为相机坐标系下的3D点坐标
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    depth = depth_image[v, u]  # 注意OpenCV图像的索引是[y, x]
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return np.array([X, Y, Z])
class RealTimeYOLODetector:
    """实时YOLO检测器，支持多相机同时预览和YOLO实时推理"""
    
<<<<<<< HEAD
    def __init__(self, config_path=r'/home/wyn/Documents/xwechat_files/wxid_uklfyt3st8sl12_39e9/msg/file/2025-12/best(1).pt', 
=======
    def __init__(self, config_path=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\yolo_detect\config\camera_correspondence.yaml', 
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
                 yolo_model_path='yolov8n.pt',check_3D_keypoints=False, save_pointcloud=False, visualize_3d=False):
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
        self.save_pointcloud = save_pointcloud
        self.visualize_3d = visualize_3d
        self.last_visualization_time = 0  # 用于控制可视化频率
        
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        print(f"已加载YOLO模型: {yolo_model_path}")
        
        # 初始化相机
        self.initialize_cameras()

        if self.check_3D_keypoints:
            print("3D关键点检测已启用")

        #middle camera hand-eye matrix
        self.init_calib_mat = np.array([[0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
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
            
        # 运行推理，设置置信度阈值为0.9
<<<<<<< HEAD
        results = self.yolo_model(image, conf=0.3)
=======
        results = self.yolo_model(image, conf=0.9)
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

    
        # 渲染结果
        annotated_image = results[0].plot(
            boxes=True,
            labels=True,
        )
        
        return annotated_image, results

<<<<<<< HEAD
    def show_point_cloud(self, pcd, pcd_color,model_path = r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/cup.stl,pos"):
        base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(base)
        point_cloud = mgm.gen_pointcloud(pcd, rgba=pcd_color)
        point_cloud.attach_to(base)
        bj = mcm.CollisionModel(model_path)
        bj.pos = pos
        bj.attach_to(base)
        base.run()
=======
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

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
        
        try:
<<<<<<< HEAD

=======
<<<<<<< HEAD
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            # 实时显示画面
            while self.detection_active:
                display_images = {}
                
                for role, pipeline in self.rs_pipelines.items():
                    try:
                        pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
<<<<<<< HEAD
                        #self.show_point_cloud(pcd,pcd_color)
                        # 使用YOLO进行推理
                        annotated_img, results = self.run_yolo_inference(color_img)
                        

                        #world2cam
                        pcd = self.align_pcd(pcd)
                        
                        # # 裁剪世界坐标系下的点云到指定范围
                        # cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
                        #     pcd, 
                        #     x_range=(0, 0.6), 
                        #     y_range=(0, -0.6), 
                        #     z_range=(0.07, 0.08)
                        # )
                        
                        # # 打印裁剪后的点云（按高度排序）并计算中心点
                        # if cropped_pcd is not None and len(cropped_pcd) > 0:
                        #     self.print_cropped_pointcloud_with_center(cropped_pcd, role)
                            
                        #     # 可视化裁剪后的点云（控制频率，避免过于频繁）
                        #     current_time = time.time()
                        #     if current_time - self.last_visualization_time > 3.0:  # 每3秒最多可视化一次
                        #         self.visualize_pointcloud_3d(
                        #             cropped_pcd, 
                        #             title=f"裁剪后点云 - {role}相机",
                        #             height_min=0.07,
                        #             height_max=0.08,
                        #             camera_role=role
                        #         )
                        #         self.last_visualization_time = current_time
                        
                        # 提取YOLO检测框中的点云并转换到世界坐标系，同时进行高度筛选
                        # if results[0].boxes is not None and len(results[0].boxes) > 0:
                        #     bbox_pcd_world, bbox_original_count, bbox_filtered_count = self.extract_bbox_pointcloud(
                        #         results[0].boxes, 
                        #         cropped_pcd if cropped_pcd is not None else pcd,  # 使用裁剪后的点云
                        #         color_img.shape,
                        #         height_min=0.074,  # 7.4cm
                        #         height_max=0.076   # 7.6cm
                        #     )
                        #     if bbox_pcd_world is not None and len(bbox_pcd_world) > 0:
                        #         print(f"YOLO检测框高度筛选结果: 原始点云 {bbox_original_count} 个, 筛选后 {bbox_filtered_count} 个")
                        #         # 可以在这里添加点云处理逻辑
                        #         # 例如：保存点云、发送到机器人等
                                
                        #         # 可选：保存点云到文件
                        #         if hasattr(self, 'save_pointcloud') and self.save_pointcloud:
                        #             self.save_bbox_pointcloud(bbox_pcd_world, role)
                                
                        #         # 可选：3D可视化（控制频率，避免过于频繁）
                        #         if hasattr(self, 'visualize_3d') and self.visualize_3d:
                        #             current_time = time.time()
                        #             if current_time - self.last_visualization_time > 2.0:  # 每2秒最多可视化一次
                        #                 self.visualize_pointcloud_3d(
                        #                     bbox_pcd_world, 
                        #                     title=f"YOLO检测结果 - {role}相机",
                        #                     height_min=0.074,
                        #                     height_max=0.076,
                        #                     camera_role=role
                        #                 )
                        #                 self.last_visualization_time = current_time
=======
                        # 使用YOLO进行推理
                        annotated_img, results = self.run_yolo_inference(color_img)
                        
                        #world2cam
                        pcd = self.align_pcd(pcd)
                        
                        # 裁剪世界坐标系下的点云到指定范围
                        cropped_pcd, original_count, cropped_count = self.crop_pointcloud_world(
                            pcd, 
                            x_range=(0, 0.6), 
                            y_range=(0, -0.6), 
                            z_range=(0.07, 0.08)
                        )
                        
                        # 打印裁剪后的点云（按高度排序）并计算中心点
                        if cropped_pcd is not None and len(cropped_pcd) > 0:
                            self.print_cropped_pointcloud_with_center(cropped_pcd, role)
                            
                            # 可视化裁剪后的点云（控制频率，避免过于频繁）
                            current_time = time.time()
                            if current_time - self.last_visualization_time > 3.0:  # 每3秒最多可视化一次
                                self.visualize_pointcloud_3d(
                                    cropped_pcd, 
                                    title=f"裁剪后点云 - {role}相机",
                                    height_min=0.07,
                                    height_max=0.08,
                                    camera_role=role
                                )
                                self.last_visualization_time = current_time
                        
                        # 提取YOLO检测框中的点云并转换到世界坐标系，同时进行高度筛选
                        if results[0].boxes is not None and len(results[0].boxes) > 0:
                            bbox_pcd_world, bbox_original_count, bbox_filtered_count = self.extract_bbox_pointcloud(
                                results[0].boxes, 
                                cropped_pcd if cropped_pcd is not None else pcd,  # 使用裁剪后的点云
                                color_img.shape,
                                height_min=0.074,  # 7.4cm
                                height_max=0.076   # 7.6cm
                            )
                            if bbox_pcd_world is not None and len(bbox_pcd_world) > 0:
                                print(f"YOLO检测框高度筛选结果: 原始点云 {bbox_original_count} 个, 筛选后 {bbox_filtered_count} 个")
                                # 可以在这里添加点云处理逻辑
                                # 例如：保存点云、发送到机器人等
                                
                                # 可选：保存点云到文件
                                if hasattr(self, 'save_pointcloud') and self.save_pointcloud:
                                    self.save_bbox_pointcloud(bbox_pcd_world, role)
                                
                                # 可选：3D可视化（控制频率，避免过于频繁）
                                if hasattr(self, 'visualize_3d') and self.visualize_3d:
                                    current_time = time.time()
                                    if current_time - self.last_visualization_time > 2.0:  # 每2秒最多可视化一次
                                        self.visualize_pointcloud_3d(
                                            bbox_pcd_world, 
                                            title=f"YOLO检测结果 - {role}相机",
                                            height_min=0.074,
                                            height_max=0.076,
                                            camera_role=role
                                        )
                                        self.last_visualization_time = current_time
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

                       

                        display_images[role] = annotated_img if annotated_img is not None else color_img
                    except Exception as e:
                        print(f"从 {role} 相机获取图像失败: {e}")
                        display_images[role] = None
                
                # 显示所有相机画面
                for role, image in display_images.items():
                    if image is not None:
                        # 在图像上添加信息
                        info_text = f"{role} Camera"
                        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # 显示FPS信息
                        fps_text = "FPS: Calculating..."
                        if hasattr(self, 'fps'):
                            fps_text = f"FPS: {self.fps:.2f}"
                        cv2.putText(image, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow(f"{role.capitalize()} Camera (YOLO)", image)
                
                # 计算FPS
                if not hasattr(self, 'prev_time'):
                    self.prev_time = time.time()
                    self.fps_counter = 0
                else:
                    self.fps_counter += 1
                    if self.fps_counter >= 10:  # 每10帧计算一次FPS
                        curr_time = time.time()
                        self.fps = self.fps_counter / (curr_time - self.prev_time)
                        self.prev_time = curr_time
                        self.fps_counter = 0
                
                # 检查键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    self.detection_active = False
                    print("退出检测模式...")
                    break
                elif key == ord('v'):  # 'v'键切换3D可视化
                    self.visualize_3d = not self.visualize_3d
                    print(f"3D可视化: {'开启' if self.visualize_3d else '关闭'}")
                elif key == ord('c'):  # 'c'键手动触发裁剪后点云可视化
                    print("手动触发裁剪后点云可视化...")
                    # 重新获取最新的点云数据进行可视化
                    for role, pipeline in self.rs_pipelines.items():
                        try:
                            pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                            pcd = self.align_pcd(pcd)
                            cropped_pcd, _, _ = self.crop_pointcloud_world(
                                pcd, 
                                x_range=(0, 0.6), 
                                y_range=(0, -0.6), 
                                z_range=(0.07, 0.08)
                            )
                            if cropped_pcd is not None and len(cropped_pcd) > 0:
                                self.visualize_pointcloud_3d(
                                    cropped_pcd, 
                                    title=f"手动触发 - 裁剪后点云 - {role}相机",
                                    height_min=0.07,
                                    height_max=0.08,
                                    camera_role=role
                                )
                        except Exception as e:
                            print(f"手动可视化失败: {e}")
                elif key == ord('h'):  # 'h'键显示帮助
                    print("\n=== 键盘控制帮助 ===")
                    print("ESC: 退出程序")
                    print("v: 切换3D可视化")
                    print("c: 手动触发裁剪后点云可视化")
                    print("h: 显示此帮助信息")
                    print("==================\n")
                    
        except Exception as e:
            print(f"检测过程中发生错误: {e}")
        finally:
            # 清理资源
            self.cleanup()

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
        # print(11)  # 注释掉调试打印
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
                # print(f"  -> Estimated point: {estimated_xyz}")  # 注释掉3D坐标打印
                detected_points_xyz.append(estimated_xyz)
            else:
                # print(f"  -> Estimation failed for pixel ({p[0]},{p[1]}). Skipping.")  # 注释掉失败信息打印
                pass
                # Optionally, you could append a placeholder like np.array([0,0,0])
                # or just skip it as is currently done.
        return np.asarray(detected_points_xyz)
    
    # hand in eye
    # def transform_point_cloud_handeye(self,
    #             handeye_mat: np.ndarray,
    #             pcd: np.ndarray,
    #             given_conf: np.ndarray = None,
    #             component_name: Literal['rgt_arm', 'lft_arm'] = 'rgt_arm',
    #             toggle_debug=False,
    #             ):
    #     if component_name == 'rgt_arm':
    #         arm = self.rgt_arm
    #     elif component_name == 'lft_arm':
    #         arm = self.lft_arm
    #     else:
    #         raise ValueError("component_name must be either 'rgt_arm' or 'lft_arm'.")
    #     if given_conf is None:
    #         given_conf = self.rgt_arm.get_jnt_values()
    #     gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf, update=False)
    #     if hasattr(arm, 'end_effector') and arm.end_effector is not None:
    #         gl_tcp_pos = gl_tcp_pos - gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
    #     w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
    #     w2cam = w2r_mat.dot(handeye_mat)
    #     pcd_r = rm.transform_points_by_homomat(w2cam, pcd)
    #     if toggle_debug:
    #         gm.gen_frame(w2cam[:3, 3], w2cam[:3,:3]).attach_to(base)
    #     return pcd_r
    
    def align_pcd(self, pcd):
        c2w_mat = self.init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    
    def print_cropped_pointcloud_with_center(self, cropped_pcd, camera_role):
        """
        打印裁剪后的点云，按高度排序，并计算中心点
        
        Args:
            cropped_pcd: 裁剪后的点云数据 (N, 3)
            camera_role: 相机角色名称
        """
        if cropped_pcd is None or len(cropped_pcd) == 0:
            print(f"[{camera_role}相机] 没有裁剪后的点云数据")
            return
            
        print(f"\n=== [{camera_role}相机] 裁剪后点云详细信息 ===")
        print(f"点云数量: {len(cropped_pcd)}")
        
        # 按高度（Z坐标）降序排序，优先显示高度高的点
        sorted_indices = np.argsort(cropped_pcd[:, 2])[::-1]  # 降序排序
        sorted_pcd = cropped_pcd[sorted_indices]
        
        # 打印前20个最高点（避免输出过多）
        print(f"\n前20个最高点（按高度降序）:")
        print("序号    X坐标(m)    Y坐标(m)    Z坐标(m)    高度(cm)")
        print("-" * 60)
        for i in range(min(20, len(sorted_pcd))):
            point = sorted_pcd[i]
            print(f"{i+1:3d}    {point[0]:8.4f}    {point[1]:8.4f}    {point[2]:8.4f}    {point[2]*100:6.2f}")
        
        if len(sorted_pcd) > 20:
            print(f"... (还有 {len(sorted_pcd) - 20} 个点未显示)")
        
        # 计算中心点
        center_point = np.mean(cropped_pcd, axis=0)
        print(f"\n点云中心点:")
        print(f"  X: {center_point[0]:.4f} m")
        print(f"  Y: {center_point[1]:.4f} m") 
        print(f"  Z: {center_point[2]:.4f} m ({center_point[2]*100:.2f} cm)")
        
        # 计算点云范围
        min_coords = np.min(cropped_pcd, axis=0)
        max_coords = np.max(cropped_pcd, axis=0)
        print(f"\n点云范围:")
        print(f"  X: [{min_coords[0]:.4f}, {max_coords[0]:.4f}] m (跨度: {max_coords[0]-min_coords[0]:.4f} m)")
        print(f"  Y: [{min_coords[1]:.4f}, {max_coords[1]:.4f}] m (跨度: {max_coords[1]-min_coords[1]:.4f} m)")
        print(f"  Z: [{min_coords[2]:.4f}, {max_coords[2]:.4f}] m (跨度: {max_coords[2]-min_coords[2]:.4f} m)")
        
        # 计算高度统计
        heights = cropped_pcd[:, 2]
        print(f"\n高度统计:")
        print(f"  平均高度: {np.mean(heights):.4f} m ({np.mean(heights)*100:.2f} cm)")
        print(f"  最高点: {np.max(heights):.4f} m ({np.max(heights)*100:.2f} cm)")
        print(f"  最低点: {np.min(heights):.4f} m ({np.min(heights)*100:.2f} cm)")
        print(f"  高度标准差: {np.std(heights):.4f} m ({np.std(heights)*100:.2f} cm)")
        
        print("=" * 50)

    def crop_pointcloud_world(self, pcd_world, x_range=(0, 0.6), y_range=(0, -0.6), z_range=(0.07, 0.08)):
        """
        在世界坐标系下裁剪点云到指定范围
        
        Args:
            pcd_world: 世界坐标系下的点云数据 (N, 3)
            x_range: X轴范围 (min, max)，默认(0, 0.6)
            y_range: Y轴范围 (min, max)，默认(0, -0.6) 
            z_range: Z轴范围 (min, max)，默认(0.07, 0.08)
            
        Returns:
            tuple: (裁剪后的点云数据, 原始点云数量, 裁剪后点云数量)
        """
        if pcd_world is None or len(pcd_world) == 0:
            print("输入点云为空")
            return None, 0, 0
            
        # 记录原始点云数量
        original_count = len(pcd_world)
        
        # 提取坐标
        x = pcd_world[:, 0]
        y = pcd_world[:, 1] 
        z = pcd_world[:, 2]
        
        # 创建裁剪掩码
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        y_mask = (y >= y_range[1]) & (y <= y_range[0])  # 注意Y轴范围是(0, -0.6)，所以是y >= -0.6 and y <= 0
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        
        # 组合所有掩码
        combined_mask = x_mask & y_mask & z_mask
        
        # 应用掩码裁剪点云
        cropped_pcd = pcd_world[combined_mask]
        cropped_count = len(cropped_pcd)
        
        # 打印统计信息
        print(f"点云裁剪统计:")
        print(f"  原始点云数量: {original_count}")
        print(f"  裁剪后点云数量: {cropped_count}")
        print(f"  裁剪范围: X[{x_range[0]}, {x_range[1]}], Y[{y_range[1]}, {y_range[0]}], Z[{z_range[0]}, {z_range[1]}]")
        print(f"  保留比例: {cropped_count/original_count*100:.2f}%")
        
        if cropped_count > 0:
            # 打印裁剪后点云的坐标范围
            print(f"  裁剪后点云范围:")
            print(f"    X: [{cropped_pcd[:, 0].min():.4f}, {cropped_pcd[:, 0].max():.4f}]")
            print(f"    Y: [{cropped_pcd[:, 1].min():.4f}, {cropped_pcd[:, 1].max():.4f}]")
            print(f"    Z: [{cropped_pcd[:, 2].min():.4f}, {cropped_pcd[:, 2].max():.4f}]")
        else:
            print("  警告: 裁剪后没有剩余点云")
            
        return cropped_pcd, original_count, cropped_count
    
    def extract_bbox_pointcloud(self, boxes, pcd_world, img_shape, height_min=0.074, height_max=0.076):
        """
        从YOLO检测框提取点云并转换到世界坐标系，并筛选指定高度范围的点云
        
        Args:
            boxes: YOLO检测结果中的边界框
            pcd_world: 已转换到世界坐标系的点云
            img_shape: 图像尺寸 (height, width, channels)
            height_min: 最小高度阈值 (米)，默认0.074m (7.4cm)
            height_max: 最大高度阈值 (米)，默认0.076m (7.6cm)
            
        Returns:
            tuple: (筛选后的点云数据 (N, 3), 原始点云数量, 筛选后点云数量)
        """
        if boxes is None or len(boxes) == 0:
            return None, 0, 0
            
        height, width = img_shape[:2]
        bbox_points = []
        total_original_points = 0
        total_filtered_points = 0
        
        # 获取所有检测框的坐标
        boxes_xyxy = boxes.xyxy.cpu().numpy()  # 获取 (x1, y1, x2, y2) 格式的坐标
        
        for i, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = box.astype(int)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # 从点云中提取边界框内的点
            bbox_pcd = pcd_world[y1:y2, x1:x2].reshape(-1, 3)
            
            # 过滤掉无效点 (0, 0, 0)
            valid_mask = np.any(bbox_pcd != 0, axis=1)
            valid_points = bbox_pcd[valid_mask]
            total_original_points += len(valid_points)
            
            if len(valid_points) > 0:
                # 按高度筛选点云 (假设Z轴是高度方向)
                height_mask = (valid_points[:, 2] >= height_min) & (valid_points[:, 2] <= height_max)
                height_filtered_points = valid_points[height_mask]
                total_filtered_points += len(height_filtered_points)
                
                if len(height_filtered_points) > 0:
                    bbox_points.append(height_filtered_points)
                    print(f"目标框 {i+1}: 坐标({x1},{y1},{x2},{y2})")
                    print(f"  - 原始有效点云: {len(valid_points)}")
                    print(f"  - 高度筛选后点云: {len(height_filtered_points)} (高度范围: {height_min*100:.1f}-{height_max*100:.1f}cm)")
                else:
                    print(f"目标框 {i+1}: 坐标({x1},{y1},{x2},{y2}) - 无符合高度条件的点云")
        
        if bbox_points:
            # 合并所有目标框的点云
            all_points = np.vstack(bbox_points)
            print(f"总统计: 原始点云 {total_original_points} 个, 高度筛选后 {total_filtered_points} 个")
            return all_points, total_original_points, total_filtered_points
        else:
            print(f"总统计: 原始点云 {total_original_points} 个, 高度筛选后 0 个")
            return None, total_original_points, 0
    
    def save_bbox_pointcloud(self, pointcloud, camera_role, save_dir="pointcloud_output"):
        """
        保存目标框内的点云到文件
        
        Args:
            pointcloud: 点云数据 (N, 3)
            camera_role: 相机角色名称
            save_dir: 保存目录
        """
        import os
        from datetime import datetime
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{camera_role}_bbox_pointcloud_{timestamp}.ply"
        filepath = os.path.join(save_dir, filename)
        
        # 保存为PLY格式
        self.save_ply(pointcloud, filepath)
        print(f"点云已保存到: {filepath}")
    
    def save_ply(self, points, filename):
        """
        保存点云为PLY格式
        
        Args:
            points: 点云数据 (N, 3)
            filename: 保存路径
        """
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(points)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            
            for point in points:
                f.write(f'{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n')
    
    def visualize_pointcloud_3d(self, pointcloud, title="3D Point Cloud Visualization", 
                               height_min=0.074, height_max=0.076, camera_role="unknown"):
        """
        在3D散点图中可视化点云数据
        
        Args:
            pointcloud: 点云数据 (N, 3)
            title: 图表标题
            height_min: 最小高度阈值 (米)
            height_max: 最大高度阈值 (米)
            camera_role: 相机角色标识
        """
        if pointcloud is None or len(pointcloud) == 0:
            print("没有点云数据可以可视化")
            return
            
        # 创建3D图形
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 提取坐标
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]
        
        # 根据高度着色
        colors = z  # 使用Z坐标作为颜色映射
        
        # 绘制3D散点图
        scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=1, alpha=0.6)
        
        # 设置坐标轴标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # 设置标题
        ax.set_title(f'{title}\n相机: {camera_role}, 点云数量: {len(pointcloud)}\n高度范围: {height_min*100:.1f}-{height_max*100:.1f}cm')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('高度 (m)')
        
        # 设置相等的坐标轴比例
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 显示图形
        plt.tight_layout()
        plt.show(block=False)  # 非阻塞显示
        plt.pause(0.1)  # 短暂暂停以确保图形显示
        
        print(f"3D可视化已显示: {len(pointcloud)} 个点云点")
    
    def visualize_pointcloud_3d_interactive(self, pointcloud, title="3D Point Cloud Visualization", 
                                          height_min=0.074, height_max=0.076, camera_role="unknown"):
        """
        交互式3D点云可视化（可选功能）
        
        Args:
            pointcloud: 点云数据 (N, 3)
            title: 图表标题
            height_min: 最小高度阈值 (米)
            height_max: 最大高度阈值 (米)
            camera_role: 相机角色标识
        """
        if pointcloud is None or len(pointcloud) == 0:
            print("没有点云数据可以可视化")
            return
            
        try:
            # 尝试使用plotly进行交互式可视化
            import plotly.graph_objects as go
            import plotly.express as px
            
            # 创建交互式3D散点图
            fig = go.Figure(data=[go.Scatter3d(
                x=pointcloud[:, 0],
                y=pointcloud[:, 1],
                z=pointcloud[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=pointcloud[:, 2],  # 根据高度着色
                    colorscale='Viridis',
                    opacity=0.6,
                    colorbar=dict(title="高度 (m)")
                ),
                text=[f'点 {i}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})' for i, p in enumerate(pointcloud)],
                hovertemplate='%{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )])
            
            # 更新布局
            fig.update_layout(
                title=f'{title}<br>相机: {camera_role}, 点云数量: {len(pointcloud)}<br>高度范围: {height_min*100:.1f}-{height_max*100:.1f}cm',
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='cube'
                ),
                width=800,
                height=600
            )
            
            # 显示图形
            fig.show()
            print(f"交互式3D可视化已显示: {len(pointcloud)} 个点云点")
            
        except ImportError:
            print("Plotly未安装，使用matplotlib进行可视化")
            self.visualize_pointcloud_3d(pointcloud, title, height_min, height_max, camera_role)
        except Exception as e:
            print(f"交互式可视化失败: {e}")
            print("回退到matplotlib可视化")
            self.visualize_pointcloud_3d(pointcloud, title, height_min, height_max, camera_role)



def main():
    """主函数"""
    try:
        # 创建检测器
        detector = RealTimeYOLODetector(
<<<<<<< HEAD
            yolo_model_path=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/put_bottles_dustbin/best.pt', 
=======
            yolo_model_path=r'F:\wrs_tiaozhanbei\my_project\tiaozhanbei\yolo_detect\model\empty_cup_place\best.pt',  # 可以替换为您自己的模型路径
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            check_3D_keypoints=False,
            save_pointcloud=False,  # 启用点云保存功能
            visualize_3d=False  # 启用3D可视化功能
        )
        
        # 启动检测模式
        detector.start_detection_mode()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()