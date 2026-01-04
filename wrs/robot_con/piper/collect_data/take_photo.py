import os
import cv2
import time
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path
from wrs.drivers.devices.realsense.realsense_d400s import *


def take_photo(save_dir='./collected_images', filename_prefix='middle_camera'):
    """
    使用middle相机拍摄照片
    
    Args:
        save_dir: 图片保存目录
        filename_prefix: 文件名前缀
    
    Returns:
        str: 保存的图片文件路径，如果失败返回None
    """
    try:
        # 读取YAML配置文件
        config_path = './config/camera_correspondence.yaml'
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return None
            
        with open(config_path, 'r') as file:
            camera_config = yaml.safe_load(file)
        
        # 获取middle相机ID（使用head_camera作为middle相机）
        middle_camera_id = camera_config['head_camera']['ID']
        print(f"Middle相机ID: {middle_camera_id}")
        
        # 查找实际连接的设备
        available_serials, ctx = find_devices()
        print("检测到设备:", available_serials)
        
        if middle_camera_id not in available_serials:
            print(f"错误: 未找到middle相机 (ID: {middle_camera_id})")
            return None
        
        # 初始化middle相机
        print(f"正在初始化middle相机 (ID: {middle_camera_id})")
        pipeline = RealSenseD400(device=middle_camera_id)
        pipeline.reset()
        time.sleep(2)  # 等待相机稳定
        pipeline = RealSenseD400(device=middle_camera_id)  # 重新初始化
        
        # 获取图像
        print("正在拍摄照片...")
        pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
        
        if color_img is None:
            print("错误: 未能获取图像")
            pipeline.stop()
            return None
        
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{filename_prefix}_{timestamp}.jpg"
        filepath = save_path / filename
        
        # 确保图像格式正确
        if len(color_img.shape) == 3 and color_img.shape[2] == 3:
            if color_img.dtype == np.uint8:
                bgr_image = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = cv2.cvtColor(color_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            bgr_image = color_img
        
        # 保存图片
        success = cv2.imwrite(str(filepath), bgr_image)
        
        if success:
            print(f"照片保存成功: {filepath}")
            pipeline.stop()
            return str(filepath)
        else:
            print("错误: 保存图片失败")
            pipeline.stop()
            return None
            
    except Exception as e:
        print(f"拍摄照片时发生错误: {e}")
        return None


def take_photo_with_preview(save_dir='./collected_images', filename_prefix='middle_camera'):
    """
    使用middle相机拍摄照片，带实时预览
    
    Args:
        save_dir: 图片保存目录
        filename_prefix: 文件名前缀
    
    Returns:
        str: 保存的图片文件路径，如果失败返回None
    """
    try:
        # 读取YAML配置文件
        config_path = './config/camera_correspondence.yaml'
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            return None
            
        with open(config_path, 'r') as file:
            camera_config = yaml.safe_load(file)
        
        # 获取middle相机ID（使用head_camera作为middle相机）
        middle_camera_id = camera_config['head_camera']['ID']
        print(f"Middle相机ID: {middle_camera_id}")
        
        # 查找实际连接的设备
        available_serials, ctx = find_devices()
        print("检测到设备:", available_serials)
        
        if middle_camera_id not in available_serials:
            print(f"错误: 未找到middle相机 (ID: {middle_camera_id})")
            return None
        
        # 初始化middle相机
        print(f"正在初始化middle相机 (ID: {middle_camera_id})")
        pipeline = RealSenseD400(device=middle_camera_id)
        pipeline.reset()
        time.sleep(2)  # 等待相机稳定
        pipeline = RealSenseD400(device=middle_camera_id)  # 重新初始化
        
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("开始实时预览，按空格键拍摄照片，ESC键退出...")
        
        # 实时预览和拍摄
        while True:
            try:
                pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                
                if color_img is not None:
                    # 在图像上添加提示信息
                    cv2.putText(color_img, "Press SPACE to take photo, ESC to exit", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Middle Camera Preview", color_img)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # 空格键拍摄
                    if color_img is not None:
                        # 生成文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = f"{filename_prefix}_{timestamp}.jpg"
                        filepath = save_path / filename
                        
                        # 确保图像格式正确
                        if len(color_img.shape) == 3 and color_img.shape[2] == 3:
                            if color_img.dtype == np.uint8:
                                bgr_image = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                            else:
                                bgr_image = cv2.cvtColor(color_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                        else:
                            bgr_image = color_img
                        
                        # 保存图片
                        success = cv2.imwrite(str(filepath), bgr_image)
                        
                        if success:
                            print(f"照片保存成功: {filepath}")
                            return str(filepath)
                        else:
                            print("错误: 保存图片失败")
                            return None
                    else:
                        print("错误: 未能获取图像")
                        return None
                        
                elif key == 27:  # ESC键退出
                    print("用户取消拍摄")
                    pipeline.stop()
                    cv2.destroyAllWindows()
                    return None
                    
            except Exception as e:
                print(f"预览过程中发生错误: {e}")
                continue
        
    except Exception as e:
        print(f"拍摄照片时发生错误: {e}")
        return None
    finally:
        cv2.destroyAllWindows()


def main():
    """主函数，演示take_photo函数的使用"""
    print("=== Middle相机拍照功能演示 ===")
    print("1. 直接拍摄（无预览）")
    print("2. 带预览拍摄")
    
    choice = input("请选择模式 (1/2): ").strip()
    
    if choice == '1':
        print("\n开始直接拍摄...")
        result = take_photo()
        if result:
            print(f"拍摄成功，图片保存在: {result}")
        else:
            print("拍摄失败")
            
    elif choice == '2':
        print("\n开始带预览拍摄...")
        result = take_photo_with_preview()
        if result:
            print(f"拍摄成功，图片保存在: {result}")
        else:
            print("拍摄失败或用户取消")
    else:
        print("无效选择")


if __name__ == "__main__":
    main()
