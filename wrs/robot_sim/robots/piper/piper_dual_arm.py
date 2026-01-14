#!/usr/bin/env python
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# @Time : 2025/10/10 18:25
# @Author : ZhangXi
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.modeling.collision_model as mcm
import numpy as np
import wrs.robot_sim.robots.robot_interface as ri
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
import wrs.modeling.model_collection as mmc
import wrs.basis.robot_math as rm

class DualPiperNoBody(ri.RobotInterface):
    """
    双臂系统（无身体），左右两只 Piper 机械臂，继承 RobotInterface
    """
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="dual_piper", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        # 左右臂位置
        lft_pos = pos + np.array([0, 0, 0])
        rgt_pos = pos + np.array([0, -0.597, 0])

        # 创建左右臂实例
        self.lft_arm = PiperSglArm(pos=lft_pos, rotmat=rotmat.copy(), enable_cc=enable_cc)
        self.rgt_arm = PiperSglArm(pos=rgt_pos, rotmat=rotmat.copy(), enable_cc=enable_cc)

        # 默认使用左臂
        self.delegator = self.lft_arm
        self.cc = self.delegator.cc

    # -------------------- delegator 切换 --------------------
    def use_lft(self):
        self.delegator = self.lft_arm
        self.cc = self.lft_arm.cc
        return self.lft_arm

    def use_rgt(self):
        self.delegator = self.rgt_arm
        self.cc = self.rgt_arm.cc
        return self.rgt_arm

    def use_all(self):
        self.delegator = None  # 使用左右臂组合时自行处理

    # -------------------- FK / IK / 关节值 --------------------
    def get_jnt_values(self):
        if self.delegator is None:
            # 拼接左右臂关节
            return np.concatenate([self.lft_arm.get_jnt_values(), self.rgt_arm.get_jnt_values()])
        return self.delegator.get_jnt_values()

    def rand_conf(self):
        return np.concatenate([self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()])

    def fk(self, jnt_values=None, toggle_jacobian=False, update=False):
        if self.delegator is None:
            raise AttributeError("Delegator not set. Use use_lft() or use_rgt().")
        if jnt_values is None:
            jnt_values = self.delegator.get_jnt_values()
        return self.delegator.fk(jnt_values, toggle_jacobian=toggle_jacobian, update=update)

    def backup_state(self):
        """
        备份左右臂状态，供 RRTConnect 或其他算法使用
        """
        if self.delegator is None:
            # 备份左右臂
            self.lft_arm.backup_state()
            self.rgt_arm.backup_state()
        else:
            # 仅备份当前 delegator
            self.delegator.backup_state()

    def restore_state(self):
        """
        恢复左右臂状态
        """
        if self.delegator is None:
            # 恢复左右臂
            self.lft_arm.restore_state()
            self.rgt_arm.restore_state()
        else:
            # 恢复当前 delegator
            self.delegator.restore_state()

    def goto_given_conf(self, jnt_values):
        if self.delegator is None:
            # 分配左右臂关节值
            n = self.lft_arm.n_dof
            self.lft_arm.goto_given_conf(jnt_values[:n])
            self.rgt_arm.goto_given_conf(jnt_values[n:])
        else:
            self.delegator.goto_given_conf(jnt_values)

    # -------------------- 可视化 --------------------
    def gen_meshmodel(self, rgb=None, alpha=1,
=======
import numpy as np
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.motion.probabilistic.rrt_connect as rrtc
from wrs.robot_sim.robots.piper.piper_single_arm import PiperSglArm
import wrs.modeling.model_collection as mmc
import wrs.modeling.collision_model as mcm
import wrs.basis.robot_math as rm  # 确保导入了 rm


class DualPiperNoBody(dari.DualArmRobotInterface):
    """
    双臂系统（没有身体），只有左右两只 Piper 机械臂。
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name='dual_piper', enable_cc=enable_cc)

        lft_pos = np.array([0, 0, 0]) + pos
        self._lft_arm = PiperSglArm(pos=lft_pos, rotmat=rotmat.copy(), enable_cc=True)
        rgt_pos = np.array([0, -0.597, 0]) + pos
        self._rgt_arm = PiperSglArm(pos=rgt_pos, rotmat=rotmat.copy(), enable_cc=True)

        if self.cc is not None:
            self.setup_cc()
        # 默认使用左臂
        self.use_lft()
        # 默认回家姿态
        self.goto_home_conf()

    @property
    def lft_arm(self):
        return self._lft_arm

    @property
    def rgt_arm(self):
        return self._rgt_arm

    def use_lft(self):
        self._delegator = self._lft_arm
        self.cc = self._delegator.cc

    def use_rgt(self):
        self._delegator = self._rgt_arm
        self.cc = self._delegator.cc

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=1,
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
<<<<<<< HEAD
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self.lft_arm.gen_meshmodel(
            rgb=rgb, alpha=alpha,
=======
        """
        生成双臂（无身体）的可视化模型。
        """
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")

        # 左臂模型
        self._lft_arm.gen_meshmodel(
            rgb=rgb,
            alpha=alpha,
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)
<<<<<<< HEAD
        self.rgt_arm.gen_meshmodel(
            rgb=rgb, alpha=alpha,
=======

        # 右臂模型
        self._rgt_arm.gen_meshmodel(
            rgb=rgb,
            alpha=alpha,
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
            toggle_tcp_frame=toggle_tcp_frame,
            toggle_jnt_frames=toggle_jnt_frames,
            toggle_flange_frame=toggle_flange_frame,
            toggle_cdprim=toggle_cdprim,
            toggle_cdmesh=toggle_cdmesh
        ).attach_to(m_col)
<<<<<<< HEAD
        return m_col

    # -------------------- 手眼点云变换 --------------------
    def transform_point_cloud_handeye(self, handeye_mat: np.ndarray, pcd: np.ndarray,
                                      given_conf: np.ndarray = None,
                                      component_name: str = 'lft_arm'):
        if component_name == 'rgt_arm':
            arm = self.rgt_arm
        else:
            arm = self.lft_arm

        if given_conf is None:
            given_conf = arm.get_jnt_values()

        gl_tcp_pos, gl_tcp_rotmat = arm.fk(given_conf)
        if hasattr(arm, 'end_effector') and arm.end_effector is not None:
            try:
                gl_tcp_pos -= gl_tcp_rotmat @ arm.manipulator.loc_tcp_pos
            except AttributeError:
                pass

        w2r_mat = rm.homomat_from_posrot(gl_tcp_pos, gl_tcp_rotmat)
        w2cam = w2r_mat @ handeye_mat
        pcd_r = rm.transform_points_by_homomat(w2cam, pcd)
        return pcd_r

    def setup_cc(self):
        """为 DualPiperNoBody 设置左右臂的自碰撞与互碰检测（无 body，至 jnt4）"""
        # === 左臂 ===
        lft_mlb = self.cc.add_cce(self.lft_arm.manipulator.jlc.anchor.lnk_list[0])
        lft_ml0 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[0].lnk)
        lft_ml1 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[1].lnk)
        lft_ml2 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[2].lnk)
        lft_ml3 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[3].lnk)
        lft_ml4 = self.cc.add_cce(self.lft_arm.manipulator.jlc.jnts[4].lnk)
        lft_ee = self.cc.add_cce(self.lft_arm.end_effector.jlc.anchor.lnk_list[0])

        # 左臂自碰检测
        from_list = [lft_ml3, lft_ml4, lft_ee]
        into_list = [lft_mlb, lft_ml0, lft_ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # === 右臂 ===
        rgt_mlb = self.cc.add_cce(self.rgt_arm.manipulator.jlc.anchor.lnk_list[0])
        rgt_ml0 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[0].lnk)
        rgt_ml1 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[1].lnk)
        rgt_ml2 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[2].lnk)
        rgt_ml3 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[3].lnk)
        rgt_ml4 = self.cc.add_cce(self.rgt_arm.manipulator.jlc.jnts[4].lnk)
        rgt_ee = self.cc.add_cce(self.rgt_arm.end_effector.jlc.anchor.lnk_list[0])

        # 右臂自碰检测
        from_list = [rgt_ml3, rgt_ml4, rgt_ee]
        into_list = [rgt_mlb, rgt_ml0, rgt_ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # === 左右臂互碰检测 ===
        from_list = [lft_ml2, lft_ml3, lft_ml4, lft_ee]
        into_list = [rgt_ml2, rgt_ml3, rgt_ml4, rgt_ee]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # 动态部分（实时更新）
        self.cc.dynamic_into_list = [
            lft_mlb, lft_ml0, lft_ml1, lft_ml2,
            rgt_mlb, rgt_ml0, rgt_ml1, rgt_ml2
        ]
        self.cc.dynamic_ext_list = []

        # 将 cc 绑定到两臂
        self.lft_arm.cc = self.cc
        self.rgt_arm.cc = self.cc
=======

        return m_col

    def setup_cc(self):
        """如果需要可以在这里添加碰撞检测"""
        # self._lft_arm.cc = None
        # self._rgt_arm.cc = None
        if self.cc is not None:
            # 可以加左右臂互碰检测
            pass
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm
    import wrs.basis.robot_math as rm
    import math

    base = wd.World(cam_pos=[2, 2, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    box1 = mcm.gen_box(xyz_lengths=[0.8, 1.4, 1], pos=np.array([0.34, -0.2985, -0.5]))
    box1.attach_to(base)
    box2 = mcm.gen_box(xyz_lengths=[0.03, 0.03, 0.555], pos=np.array([-0.05, -0.2985, 0.2775]))
    box2.attach_to(base)
<<<<<<< HEAD
    box3 = mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.23, 0.07]))
    
    box4 = mcm.gen_box(xyz_lengths=[0.08, 0.16, 0.14], pos=np.array([-0.03, -0.375, 0.07]))
    
    obs_list = [box1, box2, box3, box4]
    robot = DualPiperNoBody(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)
    base.run()
    robot.use_lft()
    # tgt_pos = np.array([0.2761, -0.2985, 0.2222])

    # # === 修复后的 IK 循环 ===
    # print("开始 IK 姿态遍历...")
    # for ii, theta in enumerate(range(0, 50, 5)):
    #     hand_x = np.array([0, 1, 0])
    #     hand_z = np.array([1.0, 0.0, 0.0])
    #     hand_y = np.cross(hand_z, hand_x)
    #     tgt_rot = np.array([hand_x, hand_y, hand_z]).T
    #     tgt_rot = rm.rotmat_from_axangle(hand_y, np.radians(theta)) @ tgt_rot
    #
    #     # IK TGT JOINT
    #     j_tgt = robot.ik(tgt_pos, tgt_rot)
    #
    #     # 🎯 修复: 检查 IK 结果是否为 None
    #     if j_tgt is not None:
    #         robot.goto_given_conf(j_tgt)
    #         print(f"✅ 角度 {theta} IK 成功并设置姿态.")
    #     else:
    #         # 打印警告，但程序继续运行下一个角度
    #         print(f"❌ 警告: 角度 {theta} (第 {ii + 1} 次尝试) IK 失败，跳过姿态设置.")
    #
    # print("IK 姿态遍历结束.")

    # 下方 RRTConnect 和 IK 测试代码保持注释状态
    # # goal_conf = robot.ik(tgt_pos=np.array([0.2747, -0.2986, 0.0143]),tgt_rotmat=np.eye(3))
    # # rrtc_planner = rrtc.RRTConnect(robot)
    # # start_conf = robot.get_jnt_values()
    # # mot_data = rrtc_planner.plan(start_conf=start_conf,
    # #                              goal_conf=goal_conf,
    # #                              obstacle_list=obs_list,
    # #                              ext_dist=.1,
    # #                              max_time=300)
    # # if mot_data is not None:
    # #     n_step = len(mot_data.mesh_list)
    # #     for i, model in enumerate(mot_data.mesh_list):
    # #         model.rgb = rm.const.winter_map(i / n_step)
    # #         model.alpha = .3
    # #         model.attach_to(base)
    # # else:
    # #     print("No available motion found.")
=======
    box3 = mcm.gen_box(xyz_lengths=[0.05, 0.05, 0.05], pos=np.array([0.2761,-0.2985, 0.0522]))
    box3.attach_to(base)
    obs_list = [box1, box2]
    robot = DualPiperNoBody(enable_cc=True)
    robot.gen_meshmodel(toggle_cdprim=True).attach_to(base)

    robot.use_lft()
    tgt_pos = np.array([0.2761, -0.2985, 0.0722])

    # 下方 RRTConnect 和 IK 测试代码保持注释状态
    goal_conf = robot.ik(tgt_pos=np.array([0.2747, -0.2986, 0.0743]),tgt_rotmat = rm.rotmat_from_euler(3.0369, -0.0483, 2.7970))
    rrtc_planner = rrtc.RRTConnect(robot)
    start_conf = robot.get_jnt_values()
    mot_data = rrtc_planner.plan(start_conf=start_conf,
                                 goal_conf=goal_conf,
                                 obstacle_list=obs_list,
                                 ext_dist=.1,
                                 max_time=300)
    if mot_data is not None:
        n_step = len(mot_data.mesh_list)
        for i, model in enumerate(mot_data.mesh_list):
            model.rgb = rm.const.winter_map(i / n_step)
            model.alpha = .3
            model.attach_to(base)
    else:
        print("No available motion found.")
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6

    # # IK 测试右臂
    # tgt_pos_r = np.array([0.3397, -0.2887, 0.2201])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # mgm.gen_frame(pos=tgt_pos_r, rotmat=tgt_rotmat).attach_to(base)
    # jnt_values_r = robot.lft_arm.ik(tgt_pos_r, tgt_rotmat)
    # print(repr(jnt_values_r))
    # if jnt_values_r is not None:
    #     robot.rgt_arm.goto_given_conf(jnt_values=jnt_values_r)
    #     robot.rgt_arm.gen_meshmodel(alpha=1, toggle_tcp_frame=True).attach_to(base)

<<<<<<< HEAD
    
    # mgm.gen_frame(pos=tgt_pos_l, rotmat=tgt_rotmat).attach_to(bas
=======
    base.run()
>>>>>>> d50fd70c0bbccf881563dcbd0209244c094ad7e6
