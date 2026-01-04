from wrs import wd, rm, gpa, mcm, mgm
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import GraspCollection
from panda3d.core import *
from direct.showbase.DirectObject import DirectObject
import numpy as np
import os
from pathlib import Path
import random

def check_and_prepare_path(filepath):
    """检查路径是否存在，若不存在则创建目录"""
    path = Path(filepath)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"已创建目录: {path.parent}")
    return path

class GripperController(DirectObject):
    def __init__(self, gripper, base, pth=None):
        self.gripper = gripper
        self.base = base
        self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.rotmat = np.eye(3)
        self.ee_values = gripper.jaw_range[1]  # 默认张开
        self.pth = pth
        self.gripper_model = None
        self.saved_grasp_models = []  # 用于存储已保存抓取姿势的可视化模型
        self._is_closing = False

        # ✅ 可动态调整的移动和旋转速度
        self.move_speed = 0.005
        self.rot_speed = 0.05
        
        # 确保路径存在
        if self.pth:
            check_and_prepare_path(self.pth)
        
        # 加载或创建抓取集合
        try:
            if self.pth and Path(self.pth).exists():
                self.grasp_collection = GraspCollection.load_from_disk(file_name=self.pth)
                print(f"已从 {self.pth} 加载 {len(self.grasp_collection)} 个抓取姿势")
            else:
                self.grasp_collection = GraspCollection(end_effector=gripper)
                if self.pth:
                    print(f"将在 {self.pth} 创建新的抓取集合")
        except Exception as e:
            print(f"加载抓取集合失败: {e}, 创建新的抓取集合")
            self.grasp_collection = GraspCollection(end_effector=gripper)

        # 保存当前夹爪状态
        self.current_pos = self.pos.copy()
        self.current_rotmat = self.rotmat.copy()
        self.current_ee_values = self.ee_values

        self.update_gripper()
        self.setup_keyboard_controls()

        # 初始显示已保存的抓取姿势
        self.show_saved_grasps()
        
        self.accept('window-event-close', self.on_window_close)
        
    def on_window_close(self):
        """窗口关闭时的处理"""
        if self._is_closing:
            return
            
        self._is_closing = True
        print("正在保存抓取姿势...")
        if self.pth:
            self.save_grasps(self.pth)
        
        # 延迟退出，确保保存完成
        from direct.task.Task import Task
        def delayed_exit(task):
            base.userExit()
            return Task.done
            
        base.taskMgr.doMethodLater(0.1, delayed_exit, 'delayed_exit')

    def setup_keyboard_controls(self):
        # 平移控制
        self.accept('w', self.move, [[0, self.move_speed, 0]])
        self.accept('s', self.move, [[0, -self.move_speed, 0]])
        self.accept('a', self.move, [[-self.move_speed, 0, 0]])
        self.accept('d', self.move, [[self.move_speed, 0, 0]])
        self.accept('q', self.move, [[0, 0, self.move_speed]])
        self.accept('e', self.move, [[0, 0, -self.move_speed]])
        
        # 旋转控制
        self.accept('z', self.rotate, [[self.rot_speed, 0, 0]])
        self.accept('x', self.rotate, [[-self.rot_speed, 0, 0]])
        self.accept('c', self.rotate, [[0, self.rot_speed, 0]])
        self.accept('v', self.rotate, [[0, -self.rot_speed, 0]])
        self.accept('b', self.rotate, [[0, 0, self.rot_speed]])
        self.accept('n', self.rotate, [[0, 0, -self.rot_speed]])
        
        # 夹爪控制
        self.accept('f', self.adjust_gripper, [0.01])
        self.accept('g', self.adjust_gripper, [-0.01])
        
        # 记录抓取姿势
        self.accept('enter', self.record_grasp)
        self.accept('h', self.toggle_gripper_visibility)

        self.accept('p', self.save_grasps, [self.pth])  # 按p键保存

        # ✅ 新增：速度调节
        self.accept('[', self.change_move_speed, [-0.001])
        self.accept(']', self.change_move_speed, [0.001])
        self.accept(';', self.change_rot_speed, [-0.01])
        self.accept("'", self.change_rot_speed, [0.01])
    
    def change_move_speed(self, delta):
        """调整平移速度"""
        self.move_speed = max(0.001, self.move_speed + delta)
        print(f"🚀 当前移动速度: {self.move_speed:.4f}")

    def change_rot_speed(self, delta):
        """调整旋转速度"""
        self.rot_speed = max(0.01, self.rot_speed + delta)
        print(f"🌀 当前旋转速度: {self.rot_speed:.3f}")

    def move(self, delta, *args):
        self.current_pos += np.array(delta)
        self.update_gripper()
    
    def rotate(self, angles, *args):
        rotmat_x = rm.rotmat_from_axangle([1, 0, 0], angles[0])
        rotmat_y = rm.rotmat_from_axangle([0, 1, 0], angles[1])
        rotmat_z = rm.rotmat_from_axangle([0, 0, 1], angles[2])
        self.current_rotmat = self.current_rotmat @ rotmat_x @ rotmat_y @ rotmat_z
        self.update_gripper()
    
    def adjust_gripper(self, delta, *args):
        self.current_ee_values = np.clip(self.current_ee_values + delta, 
                                       self.gripper.jaw_range[0], 
                                       self.gripper.jaw_range[1])
        self.update_gripper()
    
    def update_gripper(self):
        """更新当前夹爪的可视化模型"""
        if self.gripper_model is not None:
            self.gripper_model.detach()
        
        # 使用当前状态更新夹爪
        self.gripper.grip_at_by_pose(self.current_pos, self.current_rotmat, self.current_ee_values)
        self.gripper_model = self.gripper.gen_meshmodel(alpha=1)
        self.gripper_model.attach_to(self.base)
    
    def show_saved_grasps(self):
        """显示所有已保存的抓取姿势，不影响当前夹爪状态"""
        # 先清除之前显示的所有抓取姿势
        for model in self.saved_grasp_models:
            model.detach()
        self.saved_grasp_models = []
        
        # 保存当前夹爪状态
        original_pos = self.current_pos.copy()
        original_rotmat = self.current_rotmat.copy()
        original_ee_values = self.current_ee_values
        
        # 显示所有已保存的抓取姿势
        for i, grasp in enumerate(self.grasp_collection):
            try:
                # 使用夹爪的副本或临时设置来显示已保存的抓取
                # 注意：这里可能会修改夹爪状态，但我们会在最后恢复
                self.gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat, grasp.ee_values)
                grasp_model = self.gripper.gen_meshmodel(rgb=[0, 1, 0], alpha=0.3)
                grasp_model.attach_to(self.base)
                self.saved_grasp_models.append(grasp_model)
            except Exception as e:
                print(f"显示第 {i+1} 个抓取姿势时出错: {e}")
        
        # 恢复当前夹爪状态
        self.current_pos = original_pos
        self.current_rotmat = original_rotmat
        self.current_ee_values = original_ee_values
        
        # 重新显示当前夹爪
        self.update_gripper()
    
    def record_grasp(self):
        """记录当前夹爪状态为一个新的抓取姿势"""
        try:
            # 使用当前状态创建抓取
            grasp = self.gripper.get_grasp(ac_pos=self.current_pos, ac_rotmat=self.current_rotmat)
            self.grasp_collection.append(grasp)
            print(f"记录抓取姿势 #{len(self.grasp_collection)}:")
            print(f"位置: {self.current_pos}")
            print(f"旋转矩阵:\n{self.current_rotmat}")
            print(f"夹爪宽度: {self.current_ee_values}")
            
            # 重新显示所有已保存的抓取姿势
            self.show_saved_grasps()
            
        except Exception as e:
            print(f"记录抓取姿势失败: {str(e)}")
    
    def toggle_gripper_visibility(self):
        if self.gripper_model is not None:
            if self.gripper_model.isHidden():
                self.gripper_model.show()
            else:
                self.gripper_model.hide()
    
    def save_grasps(self, filename):
        """保存抓取姿势到文件"""
        if not filename:
            print("错误: 未指定保存文件名")
            return False
            
        try:
            check_and_prepare_path(filename)
            self.grasp_collection.save_to_disk(file_name=filename)
            print(f"已成功保存 {len(self.grasp_collection)} 个抓取姿势到 {filename}")
            return True
        except Exception as e:
            print(f"保存抓取姿势失败: {str(e)}")
            return False

# 主程序
if __name__ == "__main__":
    base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame().attach_to(base)

    path = "/home/wyn/PycharmProjects/wrs_tiaozhanbei/my_project/tiaozhanbei/grasp/manual_grasps/tea_dongfang_manual_grasps.pickle"

    # 加载物体模型
    obj_cmodel = mcm.CollisionModel(r"/home/wyn/PycharmProjects/wrs_tiaozhanbei/0000_examples/objects/tiaozhanbei/tea dongfang.stl")
    obj_show = obj_cmodel.copy()

    obj_show.pos = [0.3,-0.3,0]
    obj_show.show_local_frame()
    obj_show.attach_to(base)
    obj_cmodel.attach_to(base)

    # 实例化PiperGripper
    gripper = pg.PiperGripper()

    controller = GripperController(gripper, base, pth=path)

    # 运行可视化界面
    base.run()