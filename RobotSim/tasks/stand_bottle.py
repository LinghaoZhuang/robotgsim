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

class Standbottle(DataCollector):
    def __init__(self, task='stand_bottle', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0, reset_cam=0.01, single_view=False, use_robot_gs=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view, use_robot_gs=use_robot_gs)
        self.reset()

    def init_3d_scene(self):
        self.init_gs()

        self.bottle = self.scene.add_entity(
            material=gs.materials.Rigid(friction=5, rho=120),
            # material=gs.materials.MPM.Elastic(),
            morph=gs.morphs.Mesh(
                file="./assets/objects/bottle_thin/bottle1.obj",
                pos=(0.3, 0.0, 0.032),
                euler=(0.0, 270.0, 180.0),
                convexify=False,
                decimate=False,
                decompose_nonconvex=True,
                # fixed=True
                scale=0.98,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )

        self.scene.build()
        self.origin_euler = self.bottle.get_quat()
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        self.scene.step()

        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 50]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 15]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -5]),                     
            np.array([20, 20, 15, 10, 10, 5])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.default_pos = self.bottle.get_dofs_position().cpu().numpy()
        self.load_cam_pose()

    def get_data(self, n_steps=45):
        # stay a while 
        for i in range(20):
            self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        ###################################################
        obj_pos = self.bottle.get_dofs_position()[:3].cpu().numpy()
        x, y = self.get_pos(obj_pos, distance=0.02)
        bottle_quat = self.bottle.get_quat().cpu().numpy()
        bottle_rot = R.from_quat([bottle_quat[1], bottle_quat[2], bottle_quat[3], bottle_quat[0]])
        bottle_euler = bottle_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = bottle_euler
        if rz > 0:
            rz = rz + 90
        else:
            rz = rz % 360
            rz = rz % 90
        rot_grasp = self.get_quat_grasp(roll_angle_deg=rz)[:5]
        # rot_grasp = self.get_quat_grasp(roll_angle_deg=90)[:5]

        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, 0.085]),
            quat=rot_grasp,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=500,
        )[:5]

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.8
        q_pos_reach[2] = q_pos_reach[2] + 0.23
        q_pos_reach[3] = q_pos_reach[3] + 0.23

        self.move_action(n_steps, q_pos_reach, 0.8)
        self.move_action(n_steps, q_pos_grasp, 0.8, mid_noise=0.01, noise=0.01)

        self.close(0.0001)

        self.move_action(n_steps, q_pos_reach, 0)

        ###################################################

        q_pos_stand= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([0.3, 0, 0.09]),
            quat=self.get_target_quat(),
            rot_tol=0.01,
            pos_tol=0.05,
            max_solver_iters=500,
        )[:5]
    
        q_pos_place_reach = q_pos_stand.clone()
        q_pos_place_reach[1] = q_pos_place_reach[1] - 0.7
        q_pos_place_reach[2] = q_pos_place_reach[2] + 0.2

        self.move_action(n_steps+10, q_pos_place_reach, 0)
        self.move_action(n_steps+10, q_pos_stand, 0, mid_noise=0.01, noise=0.01)

        self.open(0.8)
        
        self.move_action(n_steps, q_pos_place_reach, 0.8, mid_noise=0.0, noise=0.0)

        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)

        self.move_action(n_steps+10, final_pose[:5], -0.174)

        if self.bottle.get_dofs_position()[2] > 0.05:
            self.succ = True
        else:
            self.succ = False

    def get_quat(self, banana_quat):
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])
    
        banana_euler = banana_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = banana_euler
        if rz > 0:
            rz = rz - 180
        else:
            rz = rz + 180
        new_rot = R.from_euler('xyz', [rz, 90, 90], degrees=True)
        correction_rot = R.from_euler('z', 180, degrees=True)
        new_rot = correction_rot * new_rot
        
        quat = new_rot.as_quat()
        return [quat[3], quat[0], quat[1], quat[2]]

    def get_target_quat(self):
        target_rz = 0
        target_ry = 0
        target_rx = 0
        # target_rx = 0

        new_rot = R.from_euler('xyz', [target_rx, target_ry, target_rz], degrees=True)

        # final_matrix = new_rot.as_matrix()
        # if final_matrix[:, 2][2] < 0:
        #     correction_rot = R.from_euler('x', 180, degrees=True)
        #     new_rot = correction_rot * new_rot
    
        quat = new_rot.as_quat() 
        return [quat[3], quat[0], quat[1], quat[2]]

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.27
        r_max = 0.33

        theta_bottle = random.uniform(-1*math.pi/5, 1*math.pi/5)
        r_bottle = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_bottle = r_bottle * math.cos(theta_bottle)
        y_bottle = r_bottle * math.sin(theta_bottle)

        pos_bottle = np.r_[x_bottle, y_bottle, 0.033]
        self.bottle_x = x_bottle
        self.bottle_y = y_bottle

        self.bottle.set_quat(Rotation.from_euler('xyz', [random.randint(-50,50), 270, 0], degrees=True).as_quat())
        # self.bottle.set_quat(self.origin_euler)
        self.bottle.set_pos(pos_bottle)

        self.step = 0
        self.scene.step()
        self.case += 1
        torch.cuda.empty_cache()

   

if __name__ == "__main__":
    args = get_args()
    collector = Standbottle(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view,
                           use_robot_gs=args.use_robot_gs)
    collector.run(num_steps=args.num_steps)