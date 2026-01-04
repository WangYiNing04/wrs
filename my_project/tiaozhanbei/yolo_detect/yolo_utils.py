#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/23 22:04
# @Author : ZhangXi
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/10/23
# @Author : ZhangXi
"""
YOLO + RealSense 通用工具模块
适用于盘子、方块、苹果等任意检测任务
"""

import cv2
import numpy as np
from typing import Optional
from ultralytics import YOLO
from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400


# ---------- 通用函数 ----------

def transform_points_by_homomat(homomat: np.ndarray, points: np.ndarray):
    """4x4 齐次矩阵作用于点云"""
    if not isinstance(points, np.ndarray):
        raise ValueError("Points must be np.ndarray!")
    homo_points = np.ones((4, points.shape[0]))
    homo_points[:3, :] = points.T
    return (homomat @ homo_points).T[:, :3]


def _estimate_point_from_neighborhood(target_pixel, pcd_matrix, neighborhood_size=5):
    """用邻域均值估算深度点"""
    h, w = pcd_matrix.shape[:2]
    px, py = target_pixel
    half = neighborhood_size // 2
    x_min, x_max = max(0, px-half), min(w-1, px+half)
    y_min, y_max = max(0, py-half), min(h-1, py+half)
    neighborhood = pcd_matrix[y_min:y_max+1, x_min:x_max+1].reshape(-1, 3)
    valid = neighborhood[np.any(neighborhood != 0, axis=1)]
    if len(valid) == 0:
        return None
    return np.mean(valid, axis=0)


def yolo_detect_centers(yolo_model: YOLO, color_img: np.ndarray,
                        conf_thres=0.3, show=False):
    """返回每个检测框中心点的像素坐标和类别"""
    results = yolo_model(color_img, verbose=False)
    if not results or len(results[0].boxes) == 0:
        return None
    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    cls_ids = res.boxes.cls.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    centers = []
    img_disp = color_img.copy()

    for (x1, y1, x2, y2), cls_id, conf in zip(boxes, cls_ids, confs):
        if conf < conf_thres:
            continue
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        centers.append((cx, cy, int(cls_id)))
        if show:
            cv2.rectangle(img_disp, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.circle(img_disp, (cx, cy), 5, (0,0,255), -1)
            cv2.putText(img_disp, f"ID:{int(cls_id)}", (cx+5, cy-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    if show:
        cv2.imshow("YOLO Detection", img_disp)
        cv2.waitKey(1)

    if len(centers) == 0:
        return None
    return np.array(centers, dtype=int)


def yolo_detect_world_positions(yolo_model, color_img, pcd_world,
                                show=False, neighborhood_size=5):
    """检测物体并输出世界坐标"""
    centers = yolo_detect_centers(yolo_model, color_img, show=show)
    if centers is None:
        return None

    pcd_matrix = pcd_world.reshape(color_img.shape[0], color_img.shape[1], 3)
    world_points = []
    for cx, cy, cls_id in centers:
        est_point = _estimate_point_from_neighborhood((cx, cy), pcd_matrix, neighborhood_size)
        if est_point is not None:
            world_points.append((cls_id, est_point))
    return world_points


# ---------- 设备初始化 ----------
def init_camera(camera_id: str = "middle"):
    """
    初始化相机
    camera_id: "middle", "left" 或 "right"
    """
    device_map = {
        "middle": "242222071855",  # middle 相机的序列号
        #"left": "243322074546",  # left_hand 相机序列号
        #"right": "243322071033",  # right_hand 相机序列号
    }

    if camera_id not in device_map:
        raise ValueError(f"未知的 camera_id: {camera_id}, 可选: {list(device_map.keys())}")
    cam = RealSenseD400(device=device_map[camera_id])
    cam.get_pcd_texture_depth()
    cam.get_pcd_texture_depth()
    print(f"{camera_id} 相机预热完毕。")
    return cam


def init_yolo(weight_path: str):
    """初始化YOLO模型"""
    print(f"加载模型权重: {weight_path}")
    return YOLO(weight_path)