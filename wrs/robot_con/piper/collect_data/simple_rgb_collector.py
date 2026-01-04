import os
import cv2
import time
import yaml
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
from pynput import keyboard
from wrs.drivers.devices.realsense.realsense_d400s import *


class SimpleRGBCollector:
    """简化版RGB图片采集器，自动检测可用相机"""
    
    def __init__(self, save_dir='./collected_images'):
        """
        初始化RGB图片采集器
        
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = Path(save_dir)
        self.rs_pipelines = {}
        self.collection_active = False
        self.image_count = 0
        self.key_press_times = {}
        self.key_repeat_threshold = 0.1
        
        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化相机
        self.initialize_cameras()
        
    def initialize_cameras(self):
        """自动检测并初始化所有可用相机"""
        try:
            # 查找所有可用设备
            available_serials, ctx = find_devices()
            print(f"检测到 {len(available_serials)} 个设备: {available_serials}")

            if not available_serials:
                print("未检测到任何相机设备！")
                return

            # 为每个设备创建相机实例
            for i, device_id in enumerate(available_serials):
                try:
                    print(f"正在初始化相机 {i+1} (ID: {device_id})")
                    pipeline = RealSenseD400(device=device_id)
                    pipeline.reset()
                    time.sleep(2)
                    pipeline = RealSenseD400(device=device_id)  # 重新初始化
                    self.rs_pipelines[f'camera_{i+1}'] = pipeline
                    print(f"相机 {i+1} 初始化成功")
                except Exception as e:
                    print(f"初始化相机 {i+1} 失败: {e}")
                    
        except Exception as e:
            print(f"相机初始化失败: {e}")
            raise
    
    def save_images(self, images_dict, timestamp=None):
        """保存图片到指定目录"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        for camera_name, image in images_dict.items():
            if image is not None:
                # 创建相机专用目录
                camera_dir = self.save_dir / camera_name
                camera_dir.mkdir(exist_ok=True)
                
                # 保存图片
                filename = f"{camera_name}_{timestamp}_{self.image_count:06d}.jpg"
                filepath = camera_dir / filename
                
                # 确保图像格式正确
                if len(image.shape) == 3 and image.shape[2] == 3:
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
        
        for camera_name, pipeline in self.rs_pipelines.items():
            try:
                pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                images_dict[camera_name] = color_img
            except Exception as e:
                print(f"从 {camera_name} 相机获取图像失败: {e}")
                images_dict[camera_name] = None
        
        # 保存图片
        self.save_images(images_dict, timestamp)
        self.image_count += 1
        print(f"已采集 {self.image_count} 张图片")
    
    def start_collection_mode(self):
        """启动采集模式"""
        if not self.rs_pipelines:
            print("没有可用的相机，无法启动采集模式！")
            return
            
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
                time.sleep(1.0)
        
        def on_press(key):
            nonlocal continuous_collection, collection_thread
            
            # 检查按键重复
            current_time = time.time()
            key_str = str(key)
            if key_str in self.key_press_times:
                if current_time - self.key_press_times[key_str] < self.key_repeat_threshold:
                    return
            self.key_press_times[key_str] = current_time
            
            try:
                if key == keyboard.Key.space:
                    self.collect_single_frame()
                    
                elif hasattr(key, 'char') and key.char:
                    char_key = key.char.lower()
                    
                    if char_key == 'c':
                        if not continuous_collection:
                            continuous_collection = True
                            collection_thread = threading.Thread(target=continuous_collect, daemon=True)
                            collection_thread.start()
                            print("开始连续采集模式...")
                        else:
                            print("连续采集模式已在运行中...")
                            
                    elif char_key == 's':
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
                if continuous_collection:
                    continuous_collection = False
                    if collection_thread and collection_thread.is_alive():
                        collection_thread.join(timeout=1.0)
                
                self.collection_active = False
                print("退出采集模式...")
                return False
            return None
        
        # 启动键盘监听和实时显示
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as key_listener:
                while self.collection_active:
                    display_images = {}
                    
                    for camera_name, pipeline in self.rs_pipelines.items():
                        try:
                            pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                            display_images[camera_name] = color_img
                        except Exception as e:
                            print(f"从 {camera_name} 相机获取图像失败: {e}")
                    
                    # 显示所有相机画面
                    for camera_name, image in display_images.items():
                        if image is not None:
                            # 在图像上添加信息
                            info_text = f"{camera_name} - Count: {self.image_count}"
                            cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.imshow(f"{camera_name} Camera", image)
                    
                    # 检查ESC键
                    if cv2.waitKey(1) == 27:
                        break
                        
        except Exception as e:
            print(f"采集过程中发生错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        
        for pipeline in self.rs_pipelines.values():
            try:
                pipeline.stop()
            except Exception as e:
                print(f"停止相机时出错: {e}")
        
        cv2.destroyAllWindows()
        print(f"采集完成！共采集了 {self.image_count} 张图片")
        print(f"图片保存在: {self.save_dir}")


def main():
    """主函数"""
    try:
        collector = SimpleRGBCollector()
        collector.start_collection_mode()
        
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")
    finally:
        print("程序结束")


if __name__ == "__main__":
    main()
