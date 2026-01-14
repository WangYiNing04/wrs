import sys
sys.path.insert(0, "/home/wyn/lerobot/src")
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置代理（替换为你的实际代理地址）
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "https://127.0.0.1:7890"

import time
import math
import numpy as np
import torch
import yaml
import cv2
# 1) 机械臂与相机
from wrs.robot_con.piper.piper import PiperArmController

# RealSense 可能有多种封装，这里做了多路回退导入：
realsense = None
try:
    from wrs.drivers.devices.realsense.realsense_d400s import RealSenseD400
    from wrs.drivers.devices.realsense.realsense_d400s import *
    realsense = "d400s"
except Exception:
    try:
        from wrs.drivers.devices.realsense.realsense_d400 import *
        #from wrs.drivers.devices.realsense.realsense_d400 import RealsenseD400
        realsense = "d400"
    except Exception:
        pass

# 2) SmolVLA 推理相关
from wrs.robot_con.piper.policy.smolvla.modeling_smolvla import SmolVLAPolicy

from wrs.robot_con.piper.policy.smolvla.processor_smolvla import make_smolvla_pre_post_processors


# 读取YAML配置文件
with open('/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_con/piper/collect_data/config/camera_correspondence.yaml', 'r') as file:
    camera_config = yaml.safe_load(file)

# 从配置中提取相机ID
camera_roles = {
    'head': camera_config['head_camera']['ID'],
    'left_hand': camera_config['left_hand_camera']['ID'],
    'right_hand': camera_config['right_hand_camera']['ID']
}

def init_camera():

    if realsense is None:
        raise RuntimeError("未找到 RealSense D400 封装，请检查 wrs.drivers.devices.realsense.*")

    #使用左手
    cam = RealSenseD400(device=camera_roles['left_hand'])
    # 常见 API 习惯：start() / enable_stream()，不同封装略有不同，尽量统一到 get_color() 调用即可。
    # 如果你的类需要 start()，取消下一行注释：
    # cam.start()

    return cam

def get_color_image(cam):
    # 兼容常见命名：get_rgb_image / get_color_image / get_color / get_color_img
    for name in ["get_rgb_image", "get_color_image", "get_color", "get_color_img"]:
        if hasattr(cam, name):
            img = getattr(cam, name)()
            # 确保输出是 HxWx3，且为 float32 [0,1] 或 uint8 [0,255]
            if img is None:
                continue
            img = np.asarray(img)
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            return img
    raise RuntimeError("RealSense 实例上未找到彩色图像方法（get_rgb_image/get_color_image/get_color/get_color_img）")

def rotmat_to_euler_xyz(R):
    # 与 Piper 封装一致使用 XYZ 欧拉顺序
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.atan2(sy, cy)
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float32)

def build_state_from_pose(pos_m, rotmat_3x3, max_state_dim=32):
    # 将 [x,y,z,rx,ry,rz] 放到前6维，其余补零
    euler = rotmat_to_euler_xyz(rotmat_3x3)
    state6 = np.concatenate([pos_m.astype(np.float32), euler], axis=0)
    state = np.zeros((max_state_dim,), dtype=np.float32)
    state[:6] = state6
    return state

def pick_feature_keys(policy: SmolVLAPolicy):
    # 自动选择一个视觉 key、一个 state key、一个 action key
    in_feats = policy.config.input_features
    out_feats = policy.config.output_features

    # 视觉
    visual_keys = [k for k, f in in_feats.items() if getattr(f, "type", None).name == "VISUAL"]
    if not visual_keys:
        # 如果 config 没有显式视觉键，回退到一个常用键位名
        visual_keys = ["observations/images/primary"]
    img_key = visual_keys[0]

    # 状态
    state_keys = [k for k, f in in_feats.items() if getattr(f, "type", None).name == "STATE"]
    state_key = state_keys[0] if state_keys else "observations/state"

    # 动作
    action_keys = [k for k, f in out_feats.items() if getattr(f, "type", None).name == "ACTION"]
    action_key = action_keys[0] if action_keys else "actions"
    return img_key, state_key, action_key

def main():
    '''
    初始化硬件（机械臂和相机-> 加载策略模型-> 设置控制循环:在循环中，我们从相机获取图像，从机械臂获取当前状态，预处理后输入模型得到动作，将动作转换为机械臂的位姿增量并执行。
    Returns
    -------
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # a) 初始化硬件
    arm = PiperArmController(can_name="can0", has_gripper=True, auto_enable=True)
    cam = init_camera()
    visualize_cam = False
    if visualize_cam:
        # 查找实际连接的设备
        available_serials, ctx = find_devices()
        print("检测到设备:", available_serials)

        # 初始化相机（用字典存储，键为角色名称）
        rs_pipelines = {}
        for role, cam_id in camera_roles.items():
            if cam_id in available_serials:
                print(f"正在初始化 {role} 相机 (ID: {cam_id})")
                pipeline = RealSenseD400(device=cam_id)
                pipeline.reset()
                time.sleep(5)
                pipeline = RealSenseD400(device=cam_id)  # 重新初始化
                rs_pipelines[role] = pipeline
            else:
                print(f"警告: 未找到 {role} 相机 (ID: {cam_id})")

        # 实时显示画面（使用角色名称作为窗口标题）
        while True:
            for role, pipeline in rs_pipelines.items():
                try:
                    pcd, pcd_color, depth_img, color_img = pipeline.get_pcd_texture_depth()
                    cv2.imshow(f"{role.capitalize()} Camera", color_img)
                except Exception as e:
                    print(f"从 {role} 相机获取图像失败: {e}")

            if cv2.waitKey(1) == 27:  # ESC退出
                break

        # 释放资源
        for pipeline in rs_pipelines.values():
            pipeline.stop()
        cv2.destroyAllWindows()


    #local_model_path = "/home/wyn/huggingface_datasets/smolvla_base"  # 替换为你的实际路径

    # b) 加载策略与预处理
    policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device)

    # policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(
    #     local_model_path,
    #     local_files_only=True
    # ).to(device)

    policy.eval()
    preproc, postproc = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)
    img_key, state_key, action_key = pick_feature_keys(policy)

    print(f"使用视觉键: {img_key} | 状态键: {state_key} | 动作键: {action_key}")

    # c) 控制循环
    # 提示：首次请低速/小步长运行、观察动作合理性，再逐步放开
    policy.reset()
    step_hz = 5.0
    dt = 1.0 / step_hz
    max_xyz_step = 0.01       # 每步最大 1cm
    max_rpy_step = np.deg2rad(2.0)  # 每步最大 2 deg

    # 自然语言任务（可按需修改）
    task_str = "Pick up the red block."

    try:
        while True:
            # 1. 相机 -> 图像 [H,W,3], float32 in [0,1]
            img = get_color_image(cam)
            # 转为 [3,H,W]
            img_chw = np.transpose(img, (2, 0, 1)).astype(np.float32)

            # 2. 机械臂 -> 末端位姿 -> state
            pos_m, rotmat_3x3 = arm.get_pose()
            state_vec = build_state_from_pose(pos_m, rotmat_3x3, max_state_dim=policy.config.max_state_dim)

            # 3. 组 batch 并预处理
            raw_obs = {
                img_key: torch.from_numpy(img_chw).unsqueeze(0),  # [1,3,H,W]
                state_key: torch.from_numpy(state_vec).unsqueeze(0),  # [1,S]
                "task": task_str,  # 直接放在 batch 中
            }
            print(raw_obs)
            batch = preproc(raw_obs)

            # 4. 推理（单步）
            with torch.no_grad():
                action = policy.select_action(batch)  # [1, A]
                print(action)

            # PolicyAction 是 torch.Tensor 类型，直接传递 action
            out = postproc(action)
            action_np = out.cpu().numpy()[0]  # [A]

            # 5. 将动作映射为末端位姿增量控制（保守做法）
            # 约定: 前6维为 Δ[x,y,z,rx,ry,rz]（若模型维度不足6，会自动裁切）
            delta = np.zeros(6, dtype=np.float32)
            n = min(6, action_np.shape[0])
            delta[:n] = action_np[:n]

            # 限幅（避免大步跳变）
            delta[:3] = np.clip(delta[:3], -max_xyz_step, max_xyz_step)
            delta[3:] = np.clip(delta[3:], -max_rpy_step, max_rpy_step)

            # 当前姿态 + 增量
            tgt_pos = pos_m + delta[:3]
            cur_rpy = rotmat_to_euler_xyz(rotmat_3x3)
            tgt_rpy = cur_rpy + delta[3:]
            # rpy -> rotmat
            cr, sr = math.cos(tgt_rpy[0]), math.sin(tgt_rpy[0])
            cp, sp = math.cos(tgt_rpy[1]), math.sin(tgt_rpy[1])
            cy, sy = math.cos(tgt_rpy[2]), math.sin(tgt_rpy[2])
            tgt_R = np.array([
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp,     cp * sr,                cp * cr],
            ], dtype=np.float32)

            # 6. 下发运动（线性插补或位置控制）
            arm.move_l(tgt_pos, tgt_R, is_euler=False, speed=10, block=False)

            time.sleep(dt)
    except KeyboardInterrupt:
        print("停止推理循环。")
    finally:
        try:
            arm.disable()
        except Exception:
            pass

if __name__ == "__main__":
    main()