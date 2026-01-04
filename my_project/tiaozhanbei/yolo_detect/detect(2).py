#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: ZhangXi
# @Desc: 使用外部相机检测目标并返回其在左臂基座坐标系下的3D坐标

import numpy as np
import cv2
from ultralytics import YOLO
import wrs.modeling.geometric_model as gm

# ==============================================================================
# 本地实现的矩阵变换函数
# ==============================================================================
def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray):
    """
    do homotransform on a point or an array of points using pos
    :param homomat: 齐次变换矩阵
    :param points: 1x3 nparray 或 nx3 nparray (待变换的点或点云)
    :return: 变换后的点或点云
    author: weiwei
    date: 20161213
    """
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")
    if points.ndim == 1:
        # 处理单个点 (1x3)
        homo_point = np.insert(points, 3, 1)
        return np.dot(homomat, homo_point)[:3]
    else:
        # 处理点云 (nx3)
        homo_points = np.ones((4, points.shape[0]))
        homo_points[:3, :] = points.T[:3, :]
        transformed_points = np.dot(homomat, homo_points).T
        return transformed_points[:, :3]


# ------------------ 1️⃣ 相机到世界（左臂基座）的变换矩阵 (C2W) ------------------
camera_to_world_mat = np.array([
    [0.009037022325476372, -0.6821888672799827, 0.7311201572213072, -0.008952660000000005],
    [-0.9999384009275621, -0.010877202709892496, 0.0022105256641201097, -0.27816693000000003],
    [0.006444543204378151, -0.7310950959833536, -0.6822451433307909, 0.5174376099999994],
    [0.0, 0.0, 0.0, 1.0]
])


# ------------------ 2️⃣ YOLO检测函数 ------------------
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
            cv2.putText(display_img, str(i), (int(x) + 5, int(y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow("YOLO Keypoints", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
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
    # 使用本地定义的 transform_points_by_homomat 函数
    pcd_r = transform_points_by_homomat(c2w_mat, pcd)

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

    # 2. YOLO检测像素点
    pixels_coord = yolo_detect_injection_pix(yolo_model, color_img, toggle_image=True)
    if pixels_coord is None:
        print("❌ 未检测到关键点")
        return None

    pcd_matrix = pcd_left.reshape(color_img.shape[0], color_img.shape[1], 3)
    detected_points_xyz = []
    for p in pixels_coord:
        est = _estimate_point_from_neighborhood(p, pcd_matrix, neighborhood_size)
        if est is not None:
            detected_points_xyz.append(est)

    if len(detected_points_xyz) == 0:
        print("⚠️ 未提取到有效三维点")
        return None

    detected_points_xyz = np.asarray(detected_points_xyz)
    print("\n✅ 检测到的关键点在左臂基座坐标系下：")
    print(detected_points_xyz)
    return detected_points_xyz


# ------------------ 6️⃣ 示例运行 ------------------
if __name__ == "__main__":
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    try:
        from wrs.visualization.panda.world import base
    except ImportError:
        print("Base object not found for debug frame plotting.")
        base = None  # 如果 base 不存在，跳过 gm.gen_frame 的调用

    # 初始化
    yolo_model = YOLO("./model/empty_cup_place/best.pt")
    rs_cam = RealSenseD400(device='243322073422')  # 中间相机

    pcd_raw, pcd_color, depth_img, color_img = rs_cam.get_pcd_texture_depth()
    detect_keypoints_in_leftarm_frame(yolo_model, color_img, pcd_raw, camera_to_world_mat)