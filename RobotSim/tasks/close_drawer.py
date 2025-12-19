import argparse
import numpy as np
import os
import genesis as gs
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import math
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation
import sys
sys.path.append(str(Path(__file__).parent.parent)) 
from Gaussians.render import Renderer
from Gaussians.pose_utils import SE3_exp
import torch
from PIL import Image
from utils.utils import RGB2SH, SH2RGB, get_args
from tasks.base_task import DataCollector
import shutil
import json

class CloseDrawer(DataCollector):
    def __init__(self, task='close_drawer', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0,reset_cam=0.01,single_view=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view)
        self.reset()

    def init_3d_scene(self):
        self.init_gs()
        self.white_box =self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.25, -0.13, 0.024),
                size=(0.297, 0.21, 0.048),
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(
                color=(0.9, 0.9, 0.9),
                vis_mode='visual'
            ),
        )

        self.cube_1 = self.scene.add_entity(
            material=gs.materials.Rigid(friction=1),
            morph=gs.morphs.URDF(
                file="./assets/objects/cube/small_cube.urdf",
                pos =(0.25, 0.1, 0.02),
                euler=(0.0, 0.0, 225.0),
                collision=True,
                visualization=True,
                convexify=True,
            ),
        )

        self.drawer = self.scene.add_entity(
            material=gs.materials.Rigid(friction=2),
            morph=gs.morphs.URDF(
                file="./assets/objects/scaledchouti/chouti_ab.urdf",
                pos  = (0.25, -0.095, 0.076),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                convexify=False,
            ),
        )

        self.drawer_x = 0.25
        self.drawer_y = -0.095

        self.scene.build()
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        self.scene.step()
        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 100]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 25]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -20, -20, -20, -3]),                     
            np.array([20, 20, 20, 20, 20, 3])                              
        )

        self.drawer.set_dofs_position([random.uniform(0.05, 0.08)])

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.load_cam_pose()

    def get_data(self, n_steps=40):
        # stay a while 
        for i in range(50):
            self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        ###################################################
        obj_pos = self.cube_1.get_dofs_position()[:3].cpu().numpy()
        x, y = self.get_pos(obj_pos, distance=0.0)

        banana_quat = self.cube_1.get_quat().cpu().numpy()
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])
        banana_euler = banana_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = banana_euler
        rz = rz % 180
        if(rz < 90):
            rz = rz +90
        rot_grasp = self.get_quat_grasp(roll_angle_deg=rz)[:5]

        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, 0.085]),
            quat=rot_grasp,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=700,
        )[:5]

        q_pos_start = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.1, y, 0.2]),
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=700,
        )[:5]

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.5
        q_pos_reach[2] = q_pos_reach[2] + 0.23
        q_pos_reach[3] = q_pos_reach[3] + 0.23

        self.move_action(n_steps, q_pos_start, 0.0)
        self.move_action(n_steps, q_pos_reach, 0.0)
        self.move_action(n_steps, q_pos_grasp, 0.9, mid_noise=0.005, noise=0.01)

        self.close(0.0005)

        self.move_action(n_steps, q_pos_reach, 0)

        rot_grasp = self.get_quat_grasp(roll_angle_deg=90)[:5]
        q_pos_place = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([self.drawer_x+0.01, self.drawer_y + 0.1, 0.15]),
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=700,
            quat=rot_grasp,
        )[:5]


        q_pos_place[1] = q_pos_place[1] - 0.3
        q_pos_place[2] = q_pos_place[2] + 0.1


        q_pos_reach = q_pos_place.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.5
        q_pos_reach[2] = q_pos_reach[2] + 0.23
        q_pos_reach[3] = q_pos_reach[3] + 0.23
        self.move_action(n_steps, q_pos_reach, 0, mid_noise=0.005, noise=0.01)

        self.move_action(n_steps, q_pos_place, 0, mid_noise=0.005, noise=0.01)

        self.open(0.6)
        self.move_action(n_steps, q_pos_reach, 0.6)

        q_pos_close1 = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([self.drawer_x+0.01, self.drawer_y + 0.2, 0.15]),
            rot_tol=0.05,
            pos_tol=0.1,
            max_solver_iters=700,
        )[:5]
        q_pos_close1[1] = q_pos_close1[1] - 0.3
        q_pos_close2 = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([self.drawer_x+0.01, self.drawer_y + 0.09, 0.15]),
            rot_tol=0.05,
            pos_tol=0.1,
            max_solver_iters=700,
            quat=rot_grasp,
        )[:5]

        self.move_action(n_steps, q_pos_close1, 0)
        self.move_action(n_steps, q_pos_close2, 0)

        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)
        self.move_action(n_steps, final_pose[:5], -0.174)

        if math.sqrt((self.cube_1.get_pos().cpu().numpy()[:3][0] - self.drawer_x)**2 + (self.cube_1.get_pos().cpu().numpy()[:3][1] - self.drawer_y)**2) < 0.09:
            self.succ = True
        else:
            self.succ = False

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        self.drawer.set_dofs_position([random.uniform(0.05, 0.08)])

        r_min = 0.25
        r_max = 0.31

        theta_banana = random.uniform(1*math.pi/10, 1*math.pi/5)
        r_banana = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_cube1 = r_banana * math.cos(theta_banana)
        y_cube1 = r_banana * math.sin(theta_banana)

        r_min = 0.26
        r_max = 0.28
        theta_box = random.uniform(-1*math.pi/5, -1*math.pi/9) 
        r_box = math.sqrt(random.uniform(r_min**2, r_max**2))  
        self.drawer_x = r_box * math.cos(theta_box)
        self.drawer_y = r_box * math.sin(theta_box)

        pos_cube1 = np.r_[x_cube1, y_cube1, 0.03]
        pos_box = np.r_[self.drawer_x, self.drawer_y, 0.076]
        pos_white_box = np.r_[self.drawer_x, self.drawer_y-0.035, 0.024]

        self.cube_1.set_quat(Rotation.from_euler('xyz', [0, 0, random.randint(180,225)], degrees=True).as_quat())

        self.cube_1.set_pos(pos_cube1)
        self.drawer.set_pos(pos_box)
        self.white_box.set_pos(pos_white_box)

        self.step = 0
        self.scene.step()                                       
        self.case += 1
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_args()
    collector = CloseDrawer(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir)
    collector.run(num_steps=args.num_steps)