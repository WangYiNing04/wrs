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


class RGBImageCollector:
    """RGB图片采集器，支持多相机同时采集和保存"""
    
    def __init__(self, config_path='./config/camera_correspondence.yaml', save_dir='./collected_images'):
        """
        初始化RGB图片采集器
        
        Args:
            config_path: 相机配置文件路径
            save_dir: 图片保存目录
        """
        self.config_path = config_path
        self.save_dir = Path(save_dir)
        self.rs_pipelines = {}
        self.collection_active = False
        self.image_count = 0
        self.key_press_times = {}
        self.key_repeat_threshold = 0.1  # 100ms内重复按键视为重复
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
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
                'head': camera_config['head_camera']['ID'],
                'left_hand': camera_config['left_hand_camera']['ID'],
                'right_hand': camera_config['right_hand_camera']['ID']
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
    
    def save_images(self, images_dict, timestamp=None):
        """
        保存图片到指定目录
        
        Args:
            images_dict: 字典，键为相机名称，值为RGB图像
            timestamp: 时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        
        for camera_name, image in images_dict.items():
            if image is not None:
                # 创建相机专用目录
                camera_dir = self.save_dir / camera_name
                camera_dir.mkdir(exist_ok=True)
                
                # 保存图片
                filename = f"{camera_name}_{timestamp}_{self.image_count:06d}.jpg"
                filepath = camera_dir / filename
                
                # 确保图像是BGR格式（OpenCV保存需要）
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # 如果是RGB格式，转换为BGR
                    if image.dtype == np.uint8:
                        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        bgr_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                else:
                    bgr_image = image
                
                cv2.imwrite(str(filepath), bgr_image)
                print(f"保存图片: {filepath}")
    
    def collect_single_frame(self):
        """采集单帧图片"""
        images_dict = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        for role, pipeline in self.rs_pipelines.items():
            try:
                pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                images_dict[role] = color_img
            except Exception as e:
                print(f"从 {role} 相机获取图像失败: {e}")
                images_dict[role] = None
        
        # 保存图片
        self.save_images(images_dict, timestamp)
        self.image_count += 1
        print(f"已采集 {self.image_count} 张图片")
    
    def start_collection_mode(self):
        """
        启动采集模式，支持实时预览和键盘控制
        控制说明：
        - 空格键：采集当前帧
        - 'c'键：连续采集模式（每秒采集一次）
        - 's'键：停止连续采集
        - ESC键：退出程序
        """
        print("启动RGB图片采集模式...")
        print("控制说明：")
        print("- 空格键：采集当前帧")
        print("- 'c'键：连续采集模式（每秒采集一次）")
        print("- 's'键：停止连续采集")
        print("- ESC键：退出程序")
        
        self.collection_active = True
        continuous_collection = False
        collection_thread = None
        
        def continuous_collect():
            """连续采集线程"""
            while continuous_collection and self.collection_active:
                self.collect_single_frame()
                time.sleep(1.0)  # 每秒采集一次
        
        def on_press(key):
            nonlocal continuous_collection, collection_thread
            
            # 检查按键重复
            current_time = time.time()
            key_str = str(key)
            if key_str in self.key_press_times:
                if current_time - self.key_press_times[key_str] < self.key_repeat_threshold:
                    return  # 忽略重复按键
            self.key_press_times[key_str] = current_time
            
            try:
                if key == keyboard.Key.space:
                    # 采集单帧
                    self.collect_single_frame()
                    
                elif hasattr(key, 'char') and key.char:
                    char_key = key.char.lower()
                    
                    if char_key == 'c':
                        # 开始连续采集
                        if not continuous_collection:
                            continuous_collection = True
                            collection_thread = threading.Thread(target=continuous_collect, daemon=True)
                            collection_thread.start()
                            print("开始连续采集模式...")
                        else:
                            print("连续采集模式已在运行中...")
                            
                    elif char_key == 's':
                        # 停止连续采集
                        if continuous_collection:
                            continuous_collection = False
                            if collection_thread and collection_thread.is_alive():
                                collection_thread.join(timeout=1.0)
                            print("停止连续采集模式...")
                        else:
                            print("连续采集模式未在运行...")
                            
            except AttributeError:
                pass
        
        def on_release(key):
            nonlocal continuous_collection, collection_thread
            
            if key == keyboard.Key.esc:
                # 停止连续采集
                if continuous_collection:
                    continuous_collection = False
                    if collection_thread and collection_thread.is_alive():
                        collection_thread.join(timeout=1.0)
                
                self.collection_active = False
                print("退出采集模式...")
                return False  # 停止监听器
            return None
        
        # 启动键盘监听
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as key_listener:
                # 实时显示画面
                while self.collection_active:
                    display_images = {}
                    
                    for role, pipeline in self.rs_pipelines.items():
                        try:
                            pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                            display_images[role] = color_img
                        except Exception as e:
                            print(f"从 {role} 相机获取图像失败: {e}")
                    
                    # 显示所有相机画面
                    for role, image in display_images.items():
                        if image is not None:
                            # 在图像上添加信息
                            info_text = f"{role} - Count: {self.image_count}"
                            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(f"{role.capitalize()} Camera", image)
                    
                    # 检查ESC键
                    if cv2.waitKey(1) == 27:  # ESC退出
                        break
                        
        except Exception as e:
            print(f"采集过程中发生错误: {e}")
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
        
        print(f"采集完成！共采集了 {self.image_count} 张图片")
        print(f"图片保存在: {self.save_dir}")


def main():
    """主函数"""
    try:
        # 创建采集器
        collector = RGBImageCollector()
        
        # 启动采集模式
        collector.start_collection_mode()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()
