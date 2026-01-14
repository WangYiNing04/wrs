"""
Created on 2025/12/29
Author: (auto-generated)

OpenArm Definition for WRS Simulator
-----------------------------------

This module defines a seven-degree-of-freedom (7-DoF) model of the OpenArm
robotic arm for use within the WRS simulator.  It closely follows the
structure and coding style of the official PiPER example supplied with
the WRS framework, but replaces all geometric and kinematic parameters
with those extracted from the OpenArm URDF specification.

Each joint’s local translation, orientation, actuation axis and motion
limits are derived directly from the OpenArm URDF.  Individual link
meshes are loaded from the corresponding ``meshes`` directory and a
uniform colour scheme is applied for visualisation.

The resulting ``OpenArm`` class inherits from
``wrs.robot_sim.manipulators.manipulator_interface.ManipulatorInterface``
and can be instantiated and attached to a WRS ``World`` like any other
manipulator.  A TracIK solver may optionally be used for inverse
kinematics when available.
"""

import os
import numpy as np
import time
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
# Attempt to import the Trac IK solver.  If unavailable, numerical IK
# provided by the joint linkage controller (JLC) will be used instead.

try:
    from trac_ik import TracIK

    is_trac_ik = False
    print("Trac IK module loaded successfully")
except Exception as e:
    print(f"Trac IK module not loaded: {e}")
    is_trac_ik = False


is_trac_ik = False

class OpenArm(mi.ManipulatorInterface):
    """Model of the OpenArm 7-DoF arm for the WRS simulator."""

    def __init__(self,
                 pos: np.ndarray = np.zeros(3),
                 rotmat: np.ndarray = np.eye(3),
                 ik_solver: str = 'd',
                 name: str = 'OpenArm',
                 enable_cc: bool = True,
                 load_meshes: bool = True):
        """
        Initialise the OpenArm.

        :param pos: World position of the arm base.
        :param rotmat: World orientation of the arm base.
        :param ik_solver: Either 'd' (default WRS numerical IK) or 'a'/'j'
                          to select alternative solvers.  When the
                          trac_ik library is available the TracIK solver
                          will be used.
        :param name: Identifier for this manipulator.
        :param enable_cc: Enable self-collision checking if true.
        :param load_meshes: If False, skip loading STL mesh files (much faster,
                           but no visualization). Default: True.
        """
        print("[DEBUG] OpenArm.__init__() 开始")
        init_start_time = time.time()
        step_start = time.time()
        
        print("[DEBUG] [OpenArm] 步骤1: 调用父类初始化...")
        super().__init__(pos=pos,
                         rotmat=rotmat,
                         home_conf=np.zeros(7),
                         name=name,
                         enable_cc=enable_cc)
        print(f"[DEBUG] [OpenArm] 步骤1 完成，耗时: {time.time() - step_start:.3f} 秒")
        step_start = time.time()

        current_file_dir = os.path.dirname(__file__)

        # define a uniform colour for all links
        rgba = np.array([0.6, 0.6, 0.6, 1.0])

        # 辅助函数：条件加载STL模型
        def load_mesh_if_needed(link_obj, mesh_filename, link_name, mesh_rgba=None):
            """根据 load_meshes 参数决定是否加载STL模型
            
            注意：当 load_meshes=False 时，不能通过 setter 设置 cmodel=None，
            因为 jl.py 的 cmodel setter 会在 None 上调用 .pose 导致 AttributeError。
            我们直接跳过设置，让 _cmodel 保持初始状态（None）。
            """
            if load_meshes:
                link_start = time.time()
                mesh_path = os.path.join(current_file_dir, "meshes", mesh_filename)
                link_obj.cmodel = mcm.CollisionModel(mesh_path)
                if mesh_rgba is not None:
                    link_obj.cmodel.rgba = mesh_rgba
                elapsed = time.time() - link_start
                print(f"[DEBUG] [OpenArm] {mesh_filename} 加载完成，耗时: {elapsed:.3f} 秒")
            # 如果 load_meshes=False，跳过设置 cmodel，让它保持初始状态（None）
            # Link 对象在初始化时 _cmodel 已经是 None，所以不需要任何操作

        # --------------------------------------------------
        # anchor (link0)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤2: 加载 anchor (link0) STL 模型...")
        load_mesh_if_needed(self.jlc.anchor.lnk_list[0], "link0.stl", "anchor", rgba)
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 1 (link0 -> link1)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤3: 配置 Joint 1 和加载 link1.stl...")
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.0625])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-1.396263, 3.490659])
        load_mesh_if_needed(self.jlc.jnts[0].lnk, "link1.stl", "link1", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0,0, -0.0625])
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 2 (link1 -> link2)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤4: 配置 Joint 2 和加载 link2.stl...")
        self.jlc.jnts[1].loc_pos = np.array([-0.0301, 0.0, 0.06])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0.0, 0.0)
        #self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(np.pi/2, 0, 0)

        self.jlc.jnts[1].loc_motion_ax = np.array([-1, 0, 0.0])
        self.jlc.jnts[1].motion_range = np.array([-1.745329, 1.745329])
        load_mesh_if_needed(self.jlc.jnts[1].lnk, "link2.stl", "link2", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0.0301, 0.0, -0.1225])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 3 (link2 -> link3)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤5: 配置 Joint 3 和加载 link3.stl...")
        self.jlc.jnts[2].loc_pos = np.array([0.0301, 0.0, 0.06625])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-1.570796, 1.570796])
        load_mesh_if_needed(self.jlc.jnts[2].lnk, "link3.stl", "link3", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0, 0, -0.18875])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 4 (link3 -> link4)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤6: 配置 Joint 4 和加载 link4.stl...")
        self.jlc.jnts[3].loc_pos = np.array([0.0, 0.0315, 0.15375])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0,1, 0.0])
        self.jlc.jnts[3].motion_range = np.array([0.0, 2.443461])
        load_mesh_if_needed(self.jlc.jnts[3].lnk, "link4.stl", "link4", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0,-0.0315,-0.3425])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 5 (link4 -> link5)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤7: 配置 Joint 5 和加载 link5.stl...")
        self.jlc.jnts[4].loc_pos = np.array([0.0, -0.0315, 0.0955])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-1.570796, 1.570796])
        load_mesh_if_needed(self.jlc.jnts[4].lnk, "link5.stl", "link5", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0.0, 0.0, -0.438])
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 6 (link5 -> link6)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤8: 配置 Joint 6 和加载 link6.stl...")
        self.jlc.jnts[5].loc_pos = np.array([0.0375, 0.0, 0.1205])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([1.0, 0.0, 0.0])
        self.jlc.jnts[5].motion_range = np.array([-0.785398, 0.785398])
        load_mesh_if_needed(self.jlc.jnts[5].lnk, "link6.stl", "link6", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[5].lnk.loc_pos = np.array([-0.0375, 0.0, -0.5585])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Joint 7 (link6 -> link7)
        # --------------------------------------------------
        if load_meshes:
            print("[DEBUG] [OpenArm] 步骤9: 配置 Joint 7 和加载 link7.stl...")
        self.jlc.jnts[6].loc_pos = np.array([-0.0375, 0.0, 0.0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[6].loc_motion_ax = np.array([0.0, 1.0, 0.0])
        self.jlc.jnts[6].motion_range = np.array([-1.570796, 1.570796])
        load_mesh_if_needed(self.jlc.jnts[6].lnk, "link7.stl", "link7", np.array([0.3, 0.3, 0.3, 1.0]))
        self.jlc.jnts[6].lnk.loc_pos = np.array([0.0, 0.0, -0.5585])
        self.jlc.jnts[6].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

        # --------------------------------------------------
        # Finalise the joint linkage chain and set up IK solver
        # --------------------------------------------------
        print("[DEBUG] [OpenArm] 步骤10: finalize() 关节链 (可能较慢，初始化IK求解器等)...")
        finalize_start = time.time()
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        print(f"[DEBUG] [OpenArm] 步骤10 完成，耗时: {time.time() - finalize_start:.3f} 秒")
        step_start = time.time()
        
        print("[DEBUG] [OpenArm] 步骤11: 设置 TCP 位置和旋转...")
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.loc_tcp_rotmat = np.eye(3)
        print(f"[DEBUG] [OpenArm] 步骤11 完成，耗时: {time.time() - step_start:.3f} 秒")
        step_start = time.time()

        is_trac_ik = False
        
        print("[DEBUG] [OpenArm] 步骤12: 初始化 IK 求解器...")
        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "openarm.urdf")
            print(f"[DEBUG] [OpenArm] 加载 URDF: {urdf}...")
            urdf_start = time.time()
            self._ik_solver = TracIK("link0", "link7", urdf,
                                     timeout=0.002,
                                     solver_type="Distance")
            print(f"[DEBUG] [OpenArm] TracIK 初始化完成，耗时: {time.time() - urdf_start:.3f} 秒")
        else:
            self._ik_solver = None
            print("[DEBUG] [OpenArm] 使用数值 IK (TracIK 未启用)")
        print(f"[DEBUG] [OpenArm] 步骤12 完成，耗时: {time.time() - step_start:.3f} 秒")
        step_start = time.time()

        if self.cc is not None:
            if load_meshes:
                print("[DEBUG] [OpenArm] 步骤13: 设置碰撞检测 (setup_cc)...")
                self.setup_cc()
                print(f"[DEBUG] [OpenArm] 步骤13 完成，耗时: {time.time() - step_start:.3f} 秒")
            else:
                print("[DEBUG] [OpenArm] 步骤13: 跳过碰撞检测 (load_meshes=False，需要模型才能进行碰撞检测)")
        else:
            print("[DEBUG] [OpenArm] 步骤13: 跳过碰撞检测 (enable_cc=False)")
        
        total_time = time.time() - init_start_time
        print(f"[DEBUG] OpenArm.__init__() 完成，总耗时: {total_time:.3f} 秒")

    def setup_cc(self) -> None:
        """Configure pairs of links for self-collision checking."""
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l4, l5, l6]
        into_list = [lb, l0, l1, l2, l3]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """
        Solve the inverse kinematics for the end-effector.
        """
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            anchor_inv_homomat = np.linalg.inv(
                rm.homomat_from_posrot(self.jlc.anchor.pos,
                                       self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(
                rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]
            seed_jnt_values = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos, tgt_rotmat,
                                      seed_jnt_values=seed_jnt_values)
        else:
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    arm = OpenArm(enable_cc=True)
    mgm.gen_frame().attach_to(base)

    # arm.goto_given_conf(np.zeros(7))
    # arm.goto_given_conf(np.array([0,np.pi/4,np.pi/2,np.pi/2,0,0,0]))
    # arm.goto_given_conf(np.array([0.02003638, 1.81482826, -1.24311076, -0.06082472, 1.16155152,
    #                               -0.20125392,0]))




    # tgt_pos = np.array([0, 0, 0.5])
    # mgm.gen_sphere(radius=0.02, pos=tgt_pos, rgb=[1, 0, 0]).attach_to(base)
    # tgt_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)

    # joint_value = arm.ik(tgt_pos=tgt_pos,tgt_rotmat=tgt_rotmat)

    # print(tgt_pos)
    # print(tgt_rotmat)
    joint_value = np.array([0,0,0,0,0,0,0])
    # joint_value1 = np.array([-1.396263, -1.745329, -1.570796, 0, -1.570796,
    #                               -0.785398, -1.570796])
    
    # joint_value2 = np.array([3.490659, 1.745329, 1.570796, 2.443461, 1.570796,
    #                               0.785398, 1.570796])
    
    # joint_value = (joint_value1 + joint_value2) / 2
    # joint_value = np.array([0,0,0,0,0,0,0,0])
    print(joint_value)
    arm.goto_given_conf(joint_value)
    print(arm.fk(joint_value))
    arm.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

    arm.show_cdprim()
    base.run()
