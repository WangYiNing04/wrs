#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ZhangXi
# @Desc: 使用外部相机检测目标并返回其在左臂基座坐标系下的3D坐标

import numpy as np
import cv2
from ultralytics import YOLO
import wrs.basis.robot_math as rm

# ------------------ 1️⃣ 手眼标定矩阵 ------------------
handeye_mat = np.array([
    [0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.008952660000000005],
    [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.27816693000000003],
    [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.5174376099999994],
    [0.0, 0.0, 0.0, 1.0]
])

# ------------------ 2️⃣ YOLO关键点检测 ------------------
def yolo_detect_injection_pix(yolo_model, color_img, toggle_image=False, confident_threshold=0.3):
    results = yolo_model(color_img)
    if not results:
        print("YOLO returned no results.")
        return None

    res = results[0]
    kp, boxes = getattr(res, "keypoints", None), getattr(res, "boxes", None)
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        print("YOLO did not detect any keypoints.")
        return None
    if boxes is None or boxes.conf is None or len(boxes.conf) == 0:
        print("YOLO did not detect any boxes.")
        return None

    top_idx = int(boxes.conf.squeeze().argmax().item())
    kp_xy_tensor = kp.xy[top_idx]
    if kp_xy_tensor is None or kp_xy_tensor.numel() == 0:
        return None

    kp_conf_tensor = getattr(kp, "conf", None)
    if kp_conf_tensor is not None and len(kp_conf_tensor) > top_idx:
        keep_mask = kp_conf_tensor[top_idx] > confident_threshold
        kp_xy_tensor = kp_xy_tensor[keep_mask]
        if kp_xy_tensor.numel() == 0:
            return None

    detected_pixel_coord = kp_xy_tensor.detach().cpu().numpy().astype(int)

    if toggle_image:
        display_img = color_img.copy()
        for i, (x, y) in enumerate(detected_pixel_coord):
            cv2.circle(display_img, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(display_img, str(i), (int(x)+5, int(y)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("YOLO Keypoints", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_pixel_coord


# ------------------ 3️⃣ 像素邻域估计深度点 ------------------
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


# ------------------ 4️⃣ 点云坐标转换函数 ------------------
def transform_point_cloud_handeye(handeye_mat, pcd, w2r_mat=None):
    """
    将相机点云转换到左臂基座坐标系（世界坐标系）
    Args:
        handeye_mat: 相机相对于机器人基座的标定矩阵
        pcd: 相机原始点云 (N, 3)
        w2r_mat: 如果有全局位姿（例如机器人位于世界系某个位置），可以传入，否则默认为单位矩阵
    """
    if w2r_mat is None:
        w2r_mat = np.eye(4)
    w2cam = w2r_mat.dot(handeye_mat)
    return rm.transform_points_by_homomat(w2cam, pcd)


# ------------------ 5️⃣ 综合检测流程 ------------------
def detect_keypoints_in_leftarm_frame(yolo_model, color_img, pcd_raw, handeye_mat, neighborhood_size=5):
    # 1. 相机点云转左臂坐标系
    pcd_left = transform_point_cloud_handeye(handeye_mat=handeye_mat, pcd=pcd_raw)

    # 2. 从彩色图像中检测关键点像素
    pixels_coord = yolo_detect_injection_pix(yolo_model, color_img, toggle_image=True)
    if pixels_coord is None:
        print("❌ 未检测到任何关键点")
        return None

    # 3. 从点云中提取3D坐标
    pcd_matrix = pcd_left.reshape(color_img.shape[0], color_img.shape[1], 3)
    detected_points_xyz = []
    for p in pixels_coord:
        estimated_xyz = _estimate_point_from_neighborhood(p, pcd_matrix, neighborhood_size)
        if estimated_xyz is not None:
            detected_points_xyz.append(estimated_xyz)
    if not detected_points_xyz:
        print("⚠️ 没有有效的3D点被提取")
        return None

    detected_points_xyz = np.asarray(detected_points_xyz)
    print("\n✅ 检测到的关键点在左臂基座坐标系下的3D坐标：")
    print(detected_points_xyz)
    return detected_points_xyz


# ------------------ 6️⃣ 示例运行 ------------------
if __name__ == "__main__":
    # 初始化 YOLO 模型
    yolo_model = YOLO("./model/empty_cup_place/best.pt")

    # 获取相机输入
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    rs_cam = RealSenseD400(device='243322073422')  # 中间相机
    print("初始化相机完成，开始检测...")
    #color_img, pcd_raw, _, _ = rs_cam.get_pcd_texture_depth()
    pcd_raw, pcd_color, depth_img, color_img = rs_cam.get_pcd_texture_depth()
    #print(color_img)
    #print(rs_cam.get_pcd_texture_depth())
    detect_keypoints_in_leftarm_frame(yolo_model, color_img, pcd_raw, handeye_mat)
