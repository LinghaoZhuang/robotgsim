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
import json
from utils.utils import RGB2SH, SH2RGB, get_args

class PickToy(DataCollector):
    def __init__(self, task='wiping', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0,reset_cam=0.01,single_view=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view)
        self.reset()

    def init_3d_scene(self):
        self.init_gs()
        self.towel = self.scene.add_entity(
            # material=gs.materials.Rigid(),
            material=gs.materials.MPM.Elastic(E=1e5, rho=100),
            # material=gs.materials.MPM.Elastic(),
            morph=gs.morphs.Mesh(
                file="./assets/objects/towel/towel.obj",
                pos=(0.25, -0.15, 0.02),
                euler=(0, -90.0, 0.0),
                scale=1,
                # fixed=True,
                collision=True,
                # visualization=True,
                convexify=False,
                # decimate=False,
                decompose_nonconvex=True, 
            ),
        )
        with open('./assets/objects/towel/towel.json', 'r') as f:
            towel_data = json.load(f)
        self.towel_data = {}
        for name, data in  towel_data['landmarks'].items():
            self.towel_data[name] = np.array(data['pos'])

        self.scene.build()
        # self.arm.set_dofs_position([0, -1.57, 1.57, 1.57, -1.57, 0])
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        # self.arm.set_dofs_position([0,-3.35, 1.83, 0, 0, 2.5])
        self.scene.step()

        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 100]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 25]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -3]),                     
            np.array([20, 20, 15, 10, 10, 10])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.default_pos = np.mean(self.towel.get_particles(), axis=0)
        self.load_cam_pose()

    def get_quat(self, towel_quat, fixed_angle_deg=80):
        towel_rot = R.from_quat([towel_quat[1], towel_quat[2], towel_quat[3], towel_quat[0]])
        towel_euler = towel_rot.as_euler('zxy', degrees=True)
        rz, rx, ry = towel_euler
        print(f"Towel Euler angles: rx={rx}, ry={ry}, rz={rz}")

        if rz > 0:
            rz = rz - 90
        else:
            rz = rz + 90
        rz = rz % 90

        if np.mean(self.towel.get_particles(), axis=0)[1] > 0:
            # right
            rz = -rz
        print(f"Banana Euler angles: rx={rx}, ry={ry}, rz={rz}")
        target_rz = rz
        target_ry = 90
        target_rx = 90
        new_rot = R.from_euler('zxy', [target_rz, target_rx, target_ry], degrees=True)
        final_matrix = new_rot.as_matrix()
        if final_matrix[:, 2][2] < 0:
            correction_rot = R.from_euler('x', 180, degrees=True)
            new_rot = correction_rot * new_rot
    
        quat = new_rot.as_quat()
        return [quat[3], quat[0], quat[1], quat[2]]

    def get_data(self, n_steps=40):
        for i in range(10):
            self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
            self.scene.step()

        # obj_pos = np.mean(self.towel.get_particles(), axis=0)
        obj_pos = np.mean(self.towel.get_particles(), axis=0)
        x, y = self.get_pos(obj_pos, distance=0.03)
        rot_grasp = self.get_quat_grasp(roll_angle_deg=90)[:5]
        q_pos_grasp = self.arm.inverse_kinematics(
            link=self.end_effector,
            quat=rot_grasp,
            pos=np.array([x, y, 0.085]),
        )[:5]

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.8
        q_pos_reach[2] = q_pos_reach[2] - 0.2
        q_pos_reach[3] = q_pos_reach[3] + 0.3

        self.move_action(n_steps, q_pos_reach, 0.9)

        q_pos_place= self.arm.inverse_kinematics(
            link=self.end_effector,
            quat=rot_grasp,
            pos=np.array([x, y-0.05, 0.085]),
        )[:5]

        self.move_action(n_steps, q_pos_grasp, 0.9)

        # import pdb; pdb.set_trace()
        self.close()

        self.move_action(n_steps, q_pos_place, 0)
        self.move_action(n_steps, q_pos_grasp, 0)
        self.move_action(n_steps, q_pos_place, 0)
        self.move_action(n_steps, q_pos_grasp, 0)


        self.move_action(n_steps, q_pos_reach, 0.9)
        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)
        self.move_action(n_steps, final_pose[:5], -0.174)

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.23
        r_max = 0.3
        
        theta_toy = random.uniform(-2*math.pi/5, 2*math.pi/5) 
        r_toy = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_toy = r_toy * math.cos(theta_toy)
        y_toy = r_toy * math.sin(theta_toy)

        pos_toy = np.r_[x_toy, y_toy, 0.1]

        self.towel_x = x_toy
        self.towel_y = y_toy

        # self.towel.set_quat(Rotation.from_euler('xyz', [random.randint(0,180), random.randint(0,180), random.randint(0,180)], degrees=True).as_quat())
   
        # self.towel.set_pos(pos_toy)
     
        self.step = 0
        self.scene.step()
        self.case += 1

if __name__ == "__main__":
    args = get_args()
    collector = PickToy(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view)
    collector.run(num_steps=args.num_steps)