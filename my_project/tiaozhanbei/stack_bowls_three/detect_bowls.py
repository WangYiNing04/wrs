import os
import time
import yaml
import numpy as np
import wrs.basis.robot_math as rm
from wrs.drivers.devices.realsense.realsense_d400s import *

import cv2
from ultralytics import YOLO
import wrs.modeling.geometric_model as gm

# ------------------ 2️⃣ YOLO检测函数 ------------------
def yolo_detect_injection_pix(yolo_model, color_img, toggle_image=True, confident_threshold=0.3):
    results = yolo_model(color_img)
    if not results:
        # 如果没有结果，返回空坐标和原图（或None）
        if toggle_image:
            cv2.imshow("YOLO Keypoints", color_img)
        return None

    res = results[0]
    kp, boxes = getattr(res, "keypoints", None), getattr(res, "boxes", None)
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        print("YOLO did not detect any keypoints.")
        if toggle_image:
            cv2.imshow("YOLO Keypoints", color_img)
        return None

    # ... (检测逻辑不变)
    if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
        print("YOLO did not detect any boxes.")
        if toggle_image:
            cv2.imshow("YOLO Keypoints", color_img)
        return None
    top_idx = int(boxes.conf.squeeze().argmax().item())
    kp_xy_tensor = kp.xy[top_idx]
    if kp_xy_tensor is None or kp_xy_tensor.numel() == 0:
        if toggle_image:
            cv2.imshow("YOLO Keypoints", color_img)
        return None

    kp_conf_tensor = getattr(kp, "conf", None)
    if kp_conf_tensor is not None and len(kp_conf_tensor) > top_idx:
        keep_mask = kp_conf_tensor[top_idx] > confident_threshold
        kp_xy_tensor = kp_xy_tensor[keep_mask]
        if kp_xy_tensor.numel() == 0:
            if toggle_image:
                cv2.imshow("YOLO Keypoints", color_img)
            return None

    detected_pixel_coord = kp_xy_tensor.detach().cpu().numpy().astype(int)

    if toggle_image:
        display_img = color_img.copy()
        for i, (x, y) in enumerate(detected_pixel_coord):
            cv2.circle(display_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(display_img, str(i), (int(x) + 5, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # --- 核心修改：移除阻塞和销毁，仅显示 ---
        cv2.imshow("YOLO Keypoints", display_img)
        # 实时更新依赖于主循环中的 cv2.waitKey(1)

    return detected_pixel_coord


# ------------------ 3️⃣ 邻域点估计 ------------------
def _estimate_point_from_neighborhood(target_pixel, pcd_matrix, neighborhood_size=5, outlier_std_threshold=2.0):
    if neighborhood_size % 2 == 0:
        neighborhood_size += 1
    h, w = pcd_matrix.shape[:2]
    px, py = target_pixel
    half_size = neighborhood_size // 2
    x_min, x_max = max(0, px - half_size), min(w - 1, px + half_size)
    y_min, y_max = max(0, py - half_size), min(h - 1, py + half_size)
    neighborhood_points = pcd_matrix[y_min:y_max + 1, x_min:x_max + 1].reshape(-1, 3)
    valid_points = neighborhood_points[np.any(neighborhood_points != 0, axis=1)]
    if len(valid_points) == 0:
        return None
    if len(valid_points) > 3:
        mean, std = np.mean(valid_points, axis=0), np.std(valid_points, axis=0)
        std[std == 0] = 1e-6
        z_scores = np.abs((valid_points - mean) / std)
        inliers = valid_points[np.all(z_scores < outlier_std_threshold, axis=1)]
        if len(inliers) == 0:
            inliers = valid_points
    else:
        inliers = valid_points
    if len(inliers) > 0:
        return np.mean(inliers, axis=0)
    else:
        return None


# ------------------ 4️⃣ 适用于固定相机的点云转换函数 (Camera-to-World) ------------------
def transform_point_cloud_fixed_camera(camera_to_world_mat: np.ndarray,
                                       pcd: np.ndarray,
                                       toggle_debug=False):
    """
    将相机点云从相机坐标系转换为世界坐标系（左臂基座坐标系）。
    此函数适用于外部固定相机（Eye-to-World/Eye-to-Base），直接使用 C2W 变换矩阵。
    """

    c2w_mat = camera_to_world_mat
    pcd_r = rm.transform_points_by_homomat(c2w_mat, pcd)

    if toggle_debug:
        # 为了调试可视化，我们画出相机在世界坐标系下的位姿
        cam_pos = c2w_mat[:3, 3]
        cam_rotmat = c2w_mat[:3, :3]
        gm.gen_frame(cam_pos, cam_rotmat).attach_to(base)

    return pcd_r


# ------------------ 5️⃣ 检测流程 ------------------
def detect_keypoints_in_leftarm_frame(yolo_model, color_img, pcd_raw, camera_to_world_mat, neighborhood_size=5):
    """
    检测关键点并返回其在左臂基座坐标系下的三维坐标
    """

    pcd_left = transform_point_cloud_fixed_camera(camera_to_world_mat, pcd_raw)

    # 2. YOLO检测像素点 (此调用会进行实时画面更新)
    pixels_coord = yolo_detect_injection_pix(yolo_model, color_img, toggle_image=True)
    if pixels_coord is None:
        return None

    pcd_matrix = pcd_left.reshape(color_img.shape[0], color_img.shape[1], 3)
    detected_points_xyz = []
    for p in pixels_coord:
        est = _estimate_point_from_neighborhood(p, pcd_matrix, neighborhood_size)
        if est is not None:
            detected_points_xyz.append(est)

    if len(detected_points_xyz) == 0:
        # print("⚠️ 未提取到有效三维点") # 避免在实时循环中过多打印
        return None

    detected_points_xyz = np.asarray(detected_points_xyz)
    # print("\n✅ 检测到的关键点在左臂基座坐标系下：") # 避免在实时循环中过多打印
    # print(detected_points_xyz)
    return detected_points_xyz


yolo_model = YOLO(r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/empty_cup_place/best.pt')

class PointCloudProcessor:
    """最小化点云处理器，实现世界坐标系转换和裁剪功能"""
    
    def __init__(self, config_path=r'/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/yolo_detect/config/camera_correspondence.yaml'):
        # middle camera hand-eye matrix (相机到世界的变换矩阵)
        self._init_calib_mat = np.array([
            [0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.00295266], 
            [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.28066693000000004], 
            [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.51193761], 
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 相机相关属性
        self.config_path = config_path
        self.rs_pipelines = {}
        self.camera_active = False
        
        # 初始化相机
        self.initialize_cameras()
    
    def align_pcd(self, pcd):
        """
        将点云从相机坐标系转换到世界坐标系
        
        Args:
            pcd: 相机坐标系下的点云数据 (N, 3)
            
        Returns:
            np.ndarray: 世界坐标系下的点云数据 (N, 3)
        """
        c2w_mat = self._init_calib_mat  # 相机到世界的变换矩阵
        return rm.transform_points_by_homomat(c2w_mat, points=pcd)
    

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
    
    def get_camera_data(self, role='middle'):
        """
        从指定相机获取点云和图像数据
        
        Args:
            role: 相机角色名称，默认'middle'
            
        Returns:
            tuple: (点云数据, 彩色点云, 深度图, 彩色图) 或 (None, None, None, None) 如果失败
        """
        if role not in self.rs_pipelines:
            print(f"错误: 未找到 {role} 相机")
            return None, None, None, None
            
        try:
            pcd, pcd_color, depth_img, color_img = self.rs_pipelines[role].get_pcd_texture_depth()
            return pcd, pcd_color, depth_img, color_img
        except Exception as e:
            print(f"从 {role} 相机获取数据失败: {e}")
            return None, None, None, None
    
    def start_camera_stream(self):
        """启动相机流并开始实时处理"""
        print("启动相机流...")
        
        self.camera_active = True
        
        try:
            while self.camera_active:
                for role, pipeline in self.rs_pipelines.items():
                    try:
                        pass
                        # while True:
                        #     # 获取相机数据
                        #     pcd, pcd_color, depth_img, color_img = self.get_camera_data(role)
                            
                        #     if pcd is not None:
                        #         # 处理点云：相机坐标系 -> 世界坐标系 -> 裁剪
                        #         cropped_pcd, original_count, cropped_count = self.process_pointcloud(pcd)
                                
                        #         if cropped_pcd is not None and len(cropped_pcd) > 0:
                        #             print(f"[{role}相机] 处理完成: {len(cropped_pcd)} 个点")
                        #         else:
                        #             print(f"[{role}相机] 没有符合条件的点云")

                        #         # 打印裁剪后的点云（按高度排序）并计算中心点
                        #         if cropped_pcd is not None and len(cropped_pcd) > 0:
                        #             bowl_left,bowl_middle_bowl_right = self.print_cropped_pointcloud_with_center(cropped_pcd, role)
                        #             self.camera_active = False
                                    
                        #         else:
                        #             print(f"[{role}相机] 获取点云失败")

                        #         return bowl_left,bowl_middle_bowl_right
                            
                    except Exception as e:
                        print(f"处理 {role} 相机数据时出错: {e}")
                
                # 短暂休眠避免过度占用CPU
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"相机流处理过程中发生错误: {e}")
        finally:
            self.cleanup()
    
 

    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        # 停止所有相机
        for pipeline in self.rs_pipelines.values():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"停止相机时出错: {e}")
        
        self.camera_active = False
        print("程序已退出")



def main():
    """主函数 - 启动真实相机流"""
    try:
        # 创建处理器（会自动初始化相机）
        processor = PointCloudProcessor()
        
        # 启动相机流
        processor.start_camera_stream()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":

    main()
