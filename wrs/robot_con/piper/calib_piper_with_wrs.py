#piper自带的move函数与在wrs.robot_sim中的piper表示并不一样，需要校准
#实现方法，当sim中模拟与实际差不多时，读取末端执行器位置比较

from wrs.robot_con.piper.piper import PiperArmController
from wrs.robot_sim.manipulators.piper.piper import *
#home_state
#  -0.04260348704118158
#  0.12138764947620562
#  -0.005899212871740834
#  0.11210249785559578
#  -0.04365068459237818
#  -0.20212658067346329
import wrs.visualization.panda.world as wd

# Visual test for the Piper arm model.
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

left_can='can0'
right_can='can1'
left_arm_con = PiperArmController(can_name=left_can, has_gripper=True)
print(f'joint_values:{left_arm_con.get_joint_values()}')
arm = Piper(enable_cc=True,rotmat=rm.rotmat_from_euler(np.pi/2, 0, np.pi))
joint_value = [ -0.04260348704118158,
                 0.12138764947620562,
                 -0.005899212871740834,
                 0.11210249785559578,
                 -0.04365068459237818,
                 -0.20212658067346329
              ]
arm.fk(joint_value, update=True)
print("wrs通过fk解出的末端位置")
print(arm.fk(joint_value, update=True))
print("piper读到的pos")
print(left_arm_con.get_pose())

arm.gen_meshmodel().attach_to(base)
# arm.show_cdprim()
base.run()
# generate random joint configurations and test FK/IK consistency
# for _ in range(10):
#     rand_conf = arm.rand_conf()
#     pos, rotmat = arm.fk(rand_conf, update=True)
#     jnt = arm.ik(pos, rotmat)
#     print(jnt)






