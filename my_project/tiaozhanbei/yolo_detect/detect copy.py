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

class RealTimeYOLODetector:
    """实时YOLO检测器，支持多相机同时预览和YOLO实时推理"""
    
    def __init__(self, config_path='./config/camera_correspondence.yaml', 
                 yolo_model_path='yolov8n.pt'):
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
        
        # 加载YOLO模型
        self.yolo_model = YOLO(yolo_model_path)
        print(f"已加载YOLO模型: {yolo_model_path}")
        
        # 初始化相机
        self.initialize_cameras()
        
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

        print(results[0].keypoints.xy)

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
        
        def on_release(key):
            if key == keyboard.Key.esc:
                self.detection_active = False
                print("退出检测模式...")
                return False  # 停止监听器
            return None
        
        # 启动键盘监听
        try:
            with keyboard.Listener(on_release=on_release) as key_listener:
                # 实时显示画面
                while self.detection_active:
                    display_images = {}
                    
                    for role, pipeline in self.rs_pipelines.items():
                        try:
                            pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                            # 使用YOLO进行推理
                            annotated_img, results = self.run_yolo_inference(color_img)
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
                    
                    # 检查ESC键
                    if cv2.waitKey(1) == 27:  # ESC退出
                        break
                        
        except Exception as e:
            print(f"检测过程中发生错误: {e}")
        finally:
            # 清理资源
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
        
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()
        
        print("程序已退出")


def main():
    """主函数"""
    try:
        # 创建检测器
        detector = RealTimeYOLODetector(
            yolo_model_path='./model/empty_cup_place/best.pt'  # 可以替换为您自己的模型路径
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