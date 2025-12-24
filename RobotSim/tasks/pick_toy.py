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
from utils.utils import RGB2SH, SH2RGB, get_args

class PickToy(DataCollector):
    def __init__(self, task='pick_toy', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0,reset_cam=0.01,single_view=False, use_robot_gs=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view, use_robot_gs=use_robot_gs)
        self.reset()

    def init_3d_scene(self):
        self.init_gs()
        self.toy = self.scene.add_entity(
            # material=gs.materials.MPM.Elastic(E=1e5, rho=100),
            material=gs.materials.Rigid(friction=4),
            morph=gs.morphs.Mesh(
                file="./assets/objects/toy/toy.obj",
                pos=(0.32, 0.1, 0.05),
                euler=(0.0, 45.0, 0.0),
                collision=True,
                visualization=True,
                convexify=False,
                decimate=False,
                decompose_nonconvex=True,
                scale=1,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )

        self.box = self.scene.add_entity(
            material=gs.materials.Rigid(),
            morph=gs.morphs.Mesh(
                file="./assets/objects/box/box.obj",
                pos=(0.2, -0.15, -0.004),
                euler=(90.0, 0.0, 180),
                scale=1,
                collision=True,
                visualization=True,
                fixed=True,
                convexify=False,
            ),
            surface=gs.surfaces.Default(vis_mode='visual'),
        )
        self.box_x = 0.2
        self.box_y = -0.15
  
        # self.arm.control_dofs_position([2.5], self.gripper_dof)
        self.scene.build()

        if self.single_view:
            # build 之后再安全读取
            self.origin_cam_pos = self.cam.pos
            self.origin_cam_lookat = self.cam.lookat
        else:
            self.origin_cam_pos_left = self.cam_left.pos
            self.origin_cam_lookat_left = self.cam_left.lookat
            self.origin_cam_pos_right = self.cam_right.pos
            self.origin_cam_lookat_right = self.cam_right.lookat

        self.origin_euler = self.toy.get_quat()
        # self.arm.set_dofs_position([0, -1.57, 1.57, 1.57, -1.57, 0])
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        # self.arm.set_dofs_position([0,-3.35, 1.83, 0, 0, 2.5])
        self.scene.step()

        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 50]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 10]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -10]),                     
            np.array([20, 20, 15, 10, 10, 15])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")

        self.default_pos = self.toy.get_dofs_position().cpu().numpy()
        # self.default_pos = np.mean(self.toy.get_particles(), axis=0)

    def get_data(self, n_steps=50):
        for i in range(70):
            self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        ###################################################
        obj_pos = self.toy.get_dofs_position()[:3].cpu().numpy()
        toy_quat = self.toy.get_quat().cpu().numpy()
        toy_rot = R.from_quat([toy_quat[1], toy_quat[2], toy_quat[3], toy_quat[0]])
        toy_euler = toy_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = toy_euler

        # if -90 < rz < 45:
        #     roll_angle_deg = rz + 90
        # else:
        #     roll_angle_deg = rz % 180 - 90

        roll_angle_deg = rz + 90
        roll_angle_deg = roll_angle_deg % 180

        rot_grasp = self.get_quat_grasp(roll_angle_deg=roll_angle_deg)[:5]
        x, y = self.get_pos(obj_pos, distance=0.02)

        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, 0.085]),
            quat=rot_grasp,
            rot_tol=0.1,
            pos_tol=0.02,
            max_solver_iters=700,
        )[:5]
        
        if (q_pos_grasp[3] < 0):
            q_pos_grasp[3] = -q_pos_grasp[3]

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 1
        q_pos_reach[2] = q_pos_reach[2] + 0.25
        q_pos_reach[3] = q_pos_reach[3] + 0.25

        self.move_action(n_steps, q_pos_reach, 1)
        self.move_action(n_steps, q_pos_grasp, 1,mid_noise=0.01, noise=0.01)
        
        self.close(0.0001)

        self.move_action(n_steps, q_pos_reach, -0.5)

        q_pos_place= self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([self.box_x, self.box_y, 0.05]),
            quat=rot_grasp,
            rot_tol=0.05,
            pos_tol=0.02,
            max_solver_iters=700,
        )[:5]

        if (q_pos_place[3] < 0):
            q_pos_place[3] = -q_pos_place[3]
        q_pos_place[1] = q_pos_place[1] - 0.6
        q_pos_place[2] = q_pos_place[2] + 0.2
        q_pos_place_reach = q_pos_place.clone()
        q_pos_place_reach[1] = q_pos_place_reach[1] - 0.5
        q_pos_place_reach[2] = q_pos_place_reach[2] + 0.2

        self.move_action(n_steps, q_pos_place_reach, -0.5)
        self.move_action(n_steps, q_pos_place, -0.5)

        self.open(0.9)

        self.move_action(n_steps, q_pos_place_reach, 1)

        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)
        self.move_action(n_steps, final_pose[:5], -0.174)

        if math.sqrt((self.toy.get_dofs_position()[:3].cpu().numpy()[0] - self.box_x)**2 + (self.toy.get_dofs_position()[:3].cpu().numpy()[1] - self.box_y)**2) < 0.08:
            self.succ = True
        else:
            self.succ = False

    def reset(self):   
        r_min = 0.25
        r_max = 0.35
        
        theta_toy = random.uniform(1*math.pi/10, 1*math.pi/6) 
        r_toy = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_toy = r_toy * math.cos(theta_toy)
        y_toy = r_toy * math.sin(theta_toy)

        r_min = 0.28
        r_max = 0.34
        theta_box = random.uniform(-1*math.pi/4, -1*math.pi/12) 
        r_box = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_box = r_box * math.cos(theta_box)
        y_box = r_box * math.sin(theta_box)

        pos_toy = np.r_[x_toy, y_toy, 0.05]
        pos_box = np.r_[x_box, y_box, -0.004]

        self.box_x = x_box
        self.box_y = y_box
        self.toy_x = x_toy
        self.toy_y = y_toy

        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        range = [(-50, 50), (130, 230)]
        # range = [(160, 200)]
        chose_range = random.choice(range)
        # self.toy.set_quat(self.origin_euler)
        self.toy.set_quat(Rotation.from_euler('xyz', [random.randint(chose_range[0], chose_range[1]), 0, 0], degrees=True).as_quat())
        self.box.set_quat(Rotation.from_euler('xyz', [random.randint(0,360), 0, 90], degrees=True).as_quat())

        self.toy.set_pos(pos_toy)
        self.box.set_pos(pos_box)

        self.step = 0
        self.scene.step()
        self.case += 1
        torch.cuda.empty_cache()

if __name__ == "__main__":
    args = get_args()
    collector = PickToy(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view,
                           use_robot_gs=args.use_robot_gs)
    collector.run(num_steps=args.num_steps) 