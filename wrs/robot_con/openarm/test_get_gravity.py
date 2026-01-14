'''
Author: wang yining
Date: 2026-01-14 15:01:16
LastEditTime: 2026-01-14 15:01:26
FilePath: /wrs_tiaozhanbei/wrs/robot_con/openarm/test_get_gravity.py
Description: 
e-mail: wangyining0408@outlook.com
'''
import sys
import os
import numpy as np

# 优先从 cmeel.prefix 导入正确的 pinocchio（如果存在）
cmeel_path = None
for path in sys.path:
    if 'site-packages' in path:
        cmeel_pinocchio = os.path.join(path, 'cmeel.prefix', 'lib', 'python3.10', 'site-packages')
        if os.path.exists(cmeel_pinocchio):
            cmeel_path = cmeel_pinocchio
            break

if cmeel_path:
    sys.path.insert(0, cmeel_path)
    print(f"使用 pinocchio 路径: {cmeel_path}")

# 尝试不同的导入方式以兼容不同版本的 pinocchio
try:
    import pinocchio as pin
    
    # 方法1: 直接使用 pin.buildModelFromUrdf (pinocchio 2.x+)
    if hasattr(pin, 'buildModelFromUrdf'):
        build_model = pin.buildModelFromUrdf
        print("使用方法1: pin.buildModelFromUrdf")
    # 方法2: 从 urdf 子模块导入
    elif hasattr(pin, 'urdf') and hasattr(pin.urdf, 'buildModelFromUrdf'):
        build_model = pin.urdf.buildModelFromUrdf
        print("使用方法2: pin.urdf.buildModelFromUrdf")
    # 方法3: 尝试从 pinocchio.urdf 导入
    else:
        from pinocchio.urdf import buildModelFromUrdf as build_model
        print("使用方法3: from pinocchio.urdf import buildModelFromUrdf")
    
    # 加载模型
    urdf_path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/wrs/robot_sim/robots/openarm/openarm.urdf"
    print(f"加载 URDF 文件: {urdf_path}")
    model = build_model(urdf_path)
    data = model.createData()
    print(f"模型加载成功，自由度: {model.nq}")

    def get_gravity_torque(q):
        # q 是当前的关节弧度
        return pin.computeGeneralizedGravity(model, data, q)

    # 测试
    q_test = np.array([0, 0, 0, 0, 0, 0, 0])
    tau_g = get_gravity_torque(q_test)
    print(f"\n关节角度: {q_test}")
    print(f"重力补偿力矩: {tau_g}")
    
except AttributeError as e:
    print(f"\n错误：无法找到 buildModelFromUrdf 函数")
    print(f"请检查 pinocchio 版本和安装是否正确")
    print(f"错误详情: {e}")
    # 尝试使用 pinocchio 的其他 API
    try:
        import pinocchio.urdf as urdf
        model = urdf.buildModelFromUrdf(urdf_path)
        data = model.createData()
        tau_g = pin.computeGeneralizedGravity(model, data, q_test)
        print(f"使用替代方法成功，重力补偿力矩: {tau_g}")
    except Exception as e2:
        print(f"替代方法也失败: {e2}")
        
except Exception as e:
    print(f"\n加载 Pinocchio 模型时出错：{e}")
    print("提示：请确保安装了正确版本的 pinocchio (机器人学库)")
    import traceback
    traceback.print_exc()