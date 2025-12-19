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
from utils.utils import RGB2SH, SH2RGB
from tasks.base_task import DataCollector
import shutil
import json
from utils.utils import RGB2SH, SH2RGB, get_args, create_video_from_images


class Movebottle(DataCollector):
    def __init__(self, task='move_bottle', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0, reset_cam=0.01, single_view=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view)
        self.reset()

    def init_3d_scene(self):
        self.init_gs()

        self.bottle = self.scene.add_entity(
            material=gs.materials.Rigid(
                friction=1,
                rho=50,
                ),
            morph=gs.morphs.Mesh(
                file="./assets/objects/bottle_up/bottle1.obj",
                pos=(0.3, 0.0, 0.005),
                euler=(270.0, 0.0, 0.0),
                scale=1,
                collision=True,
                visualization=True,
                convexify=False,
                decimate=False,
                decompose_nonconvex=True,
                # fixed=True
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )

        self.scene.build()
        self.default_euler = self.bottle.get_quat()
        # self.arm.set_dofs_position([0, -1.57, 1.57, 1.57, -1.57, 0])
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        # self.arm.set_dofs_position([0,-3.35, 1.83, 0, 0, 2.5])
        self.scene.step()

        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 40]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 15]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -2]),                     
            np.array([20, 20, 15, 10, 10, 3])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.default_pos = self.bottle.get_dofs_position().cpu().numpy()
        self.load_cam_pose()
        
    def get_target_quat(self):
        target_rz = 0
        target_ry = 0
        target_rx = 180

        new_rot = R.from_euler('xyz', [target_rx, target_ry, target_rz], degrees=True)

        # final_matrix = new_rot.as_matrix()
        # if final_matrix[:, 2][2] < 0:
        #     correction_rot = R.from_euler('x', 180, degrees=True)
        #     new_rot = correction_rot * new_rot
    
        quat = new_rot.as_quat() 
        return [quat[3], quat[0], quat[1], quat[2]]

    def get_data(self, n_steps=40):
        # stay a while 
        for i in range(20):
            self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        ###################################################
        obj_pos = self.bottle.get_dofs_position()[:3].cpu().numpy()
        self.obj_x, self.obj_y = obj_pos[0], obj_pos[1]
        target_quat = self.get_target_quat()
        x_reach, y_reach = self.get_pos(obj_pos, distance=0.1)
        x_grasp, y_grasp = self.get_pos(obj_pos, distance=0.047)

        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x_grasp, y_grasp+0.005, 0.06]),
            quat=target_quat,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        q_pos_reach = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x_reach, y_reach+0.005, 0.18]),
            quat=target_quat,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        q_pos_target = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x_grasp+0.01, y_grasp-0.1, 0.06]),
            quat=target_quat,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        q_pos_target_final = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x_grasp-0.03, y_grasp-0.09, 0.18]),
            quat=target_quat,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        self.move_action(n_steps, q_pos_reach, 1.8, mid_noise=0.01, noise=0.01)
        self.move_action(n_steps, q_pos_grasp, 1.8, mid_noise=0.0, noise=0.01)

        self.close(0.0005)
        ###################################################

        self.move_action(n_steps+20, q_pos_target, 0.2, mid_noise=0.00, noise=0.01)

        self.open(1.2)
        
        self.move_action(n_steps, q_pos_target_final, 1.2, mid_noise=0.00, noise=0.01)

        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)

        self.move_action(n_steps+10, final_pose[:5], -0.174)

        # print('self.bottle.get_dofs_position()[2]',self.bottle.get_dofs_position()[2])
        if self.bottle.get_dofs_position()[2] < 0.015 and (self.obj_y - self.bottle.get_dofs_position()[1]) > 0.02:
            self.succ = True
        else:
            self.succ = False

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.3
        r_max = 0.35

        theta_bottle = random.uniform(1*math.pi/20, 1*math.pi/10)
        r_bottle = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_bottle = r_bottle * math.cos(theta_bottle)
        y_bottle = r_bottle * math.sin(theta_bottle)

        self.bottle.morph.scale = random.uniform(0.75, 1.25)

        pos_bottle = np.r_[x_bottle, y_bottle, 0.005]
        self.bottle_x = x_bottle
        self.bottle_y = y_bottle

        self.bottle.set_quat(self.default_euler)
        # self.bottle.set_quat(Rotation.from_euler('xyz', [270, 0, 0], degrees=True).as_quat())
        # self.box.set_quat(Rotation.from_euler('xyz', [random.randint(0,180), 0, 90], degrees=True).as_quat())

        self.bottle.set_pos(pos_bottle)

        self.step = 0
        self.scene.step()
        self.case += 1
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_args()
    collector = Movebottle(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view)
    collector.run(num_steps=args.num_steps)