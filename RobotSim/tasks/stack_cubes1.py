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

class StackCubes(DataCollector):
    def __init__(self, task='stack_cubes_apperance', data_augmentation=False, use_gs=True, save_dir='collected_data',
                 case=0,reset_cam=0.01,single_view=False, use_robot_gs=False):
        super().__init__(task=task, data_augmentation=data_augmentation, use_gs=use_gs, save_dir=save_dir,
                         case=case, reset_cam=reset_cam,single_view=single_view, use_robot_gs=use_robot_gs)
        self.reset()
        self.scene.viewer.show_collision_geoms = False
        self.scene.viewer.show_visual_geoms = True


    def init_3d_scene(self):
        self.init_gs()
        # self.banana = self.scene.add_entity(
        #     # material=gs.materials.MPM.Elastic(E=1e5, rho=100),
        #     material=gs.materials.Rigid(friction=5),
        #     morph=gs.morphs.Mesh(
        #         # file="./assets/objects/toy/toy.obj",
        #         file="./assets/objects/banana/banana.obj",
        #         pos=(0.32, 0.1, 0.04),
        #         euler=(0.0, 0.0, 250.0),
        #         scale=1,
        #         # collision=True,
        #         # visualization=True,
        #         convexify=False,
        #         decimate=False,
        #         decompose_nonconvex=True,
        #     ),
        #     surface=gs.surfaces.Default(vis_mode='visual'),
        # )

        # self.toy = self.scene.add_entity(
        #     # material=gs.materials.MPM.Elastic(E=1e5, rho=100),
        #     material=gs.materials.Rigid(friction=4),
        #     morph=gs.morphs.Mesh(
        #         file="./assets/objects/toy/toy.obj",
        #         pos=(0.32, -0.1, 0.05),
        #         euler=(0.0, 45.0, 0.0),
        #         collision=True,
        #         visualization=True,
        #         convexify=False,
        #         decimate=False,
        #         decompose_nonconvex=True,
        #         scale=1,
        #     ),
        #     surface=gs.surfaces.Default(vis_mode='visual'),
        # )


        self.scene_noise1 =self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.2, 0, 0.0001),
                size=(0.25, 0.3, 0.0001),
                fixed=True,
            ),
            material=gs.materials.Rigid(),
            surface=gs.surfaces.Default(
                color=(0, 0, 0),
                vis_mode='visual'
            ),
        )
        # self.scene_noise2 =self.scene.add_entity(
        #     gs.morphs.Box(
        #         pos=(0.3, -0.11, 0.0001),
        #         size=(0.3, 0.22, 0.0001),
        #         fixed=True,
        #     ),
        #     material=gs.materials.Rigid(),
        #     surface=gs.surfaces.Default(
        #         color=(0.9, 0.9, 0.9),
        #         vis_mode='visual'
        #     ),
        # )

        self.cube_1 = self.scene.add_entity(
            material=gs.materials.Rigid(friction=5),
            morph=gs.morphs.URDF(
                file="./assets/objects/cube/cube_red_texture.urdf",
                # file="./assets/objects/cube/cube_red.urdf",
                pos =(0.25, -0.1, 0.04),
                euler=(0.0, 0.0, 225.0),
                collision=True,
                visualization=True,
                convexify=True,
                # scale=0.57,
                # decimate=False,
                # decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Default(
                # color=(0.914, 0.451, 0.431), # red
                vis_mode='visual'
            ),
        )

        self.cube_2 = self.scene.add_entity(
            material=gs.materials.Rigid(friction=5),
            morph=gs.morphs.URDF(
                file="./assets/objects/cube/cube_blue_texture.urdf",
                # file="./assets/objects/cube/cube_blue.urdf",
                pos=(0.25, 0.1, 0.04),
                euler=(0.0, 0.0, 225.0),
                collision=True,
                visualization=True,
                convexify=True,
                # scale=0.57,
                # decimate=False,
                # decompose_nonconvex=True,
            ),
            surface=gs.surfaces.Default(
                # color=(0.663, 0.773, 0.933), # blue
                vis_mode='visual'
            ),
        )

        self.scene.build()
        # self.arm.set_dofs_position([0, -1.57, 1.57, 1.57, -1.57, 0])
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])
        # self.arm.set_dofs_position([0,-3.35, 1.83, 0, 0, 2.5])
        self.scene.step()

        self.arm.set_dofs_kp(np.array([300, 300, 250, 200, 160, 50]))  
        self.arm.set_dofs_kv(np.array([30, 30, 25, 20, 15, 20]))     
        self.arm.set_dofs_force_range(
            np.array([-20, -20, -15, -10, -10, -3]),                     
            np.array([20, 20, 15, 10, 10, 5])                              
        )

        self.motors_dof = np.arange(5)
        self.gripper_dof = np.array([5])
        self.end_effector = self.arm.get_link("Moving_Jaw")
        self.load_cam_pose()

    def get_data(self, n_steps=35):
        # stay a while 
        for i in range(10):
            self.arm.set_dofs_position([0, -3.32, 3.11, 1.18, 0, -0.174])
            # self.arm.set_dofs_position([0, 0, 0, 0, 0, 0])
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

        q_pos_reach = q_pos_grasp.clone()
        q_pos_reach[1] = q_pos_reach[1] - 0.6
        q_pos_reach[2] = q_pos_reach[2] + 0.23
        q_pos_reach[3] = q_pos_reach[3] + 0.23

        self.move_action(n_steps, q_pos_reach, 0.9)
        self.move_action(n_steps, q_pos_grasp, 0.9, mid_noise=0.005, noise=0.01)

        self.close(0.0005)

        self.move_action(n_steps, q_pos_reach, 0)

        ###################################################

        obj_pos = self.cube_2.get_dofs_position()[:3].cpu().numpy()
        x, y = self.get_pos(obj_pos, distance=0.0)

        banana_quat = self.cube_2.get_quat().cpu().numpy()
        banana_rot = R.from_quat([banana_quat[1], banana_quat[2], banana_quat[3], banana_quat[0]])
        banana_euler = banana_rot.as_euler('xyz', degrees=True)
        rx, ry, rz = banana_euler
        rz = rz % 180
        if(rz > 90):
            rz = rz - 90
        rot_grasp = self.get_quat_grasp(roll_angle_deg=rz)[:5]

        q_pos_pick = self.arm.inverse_kinematics(
            link=self.end_effector,
            pos=np.array([x, y, 0.12]),
            quat=rot_grasp,
            rot_tol=0.01,
            pos_tol=0.0001,
            max_solver_iters=800,
        )[:5]

        q_pos_pick_reach = q_pos_pick.clone()
        q_pos_pick_reach[1] = q_pos_pick_reach[1] - 0.6
        q_pos_pick_reach[2] = q_pos_pick_reach[2] + 0.23
        q_pos_pick_reach[3] = q_pos_pick_reach[3] + 0.23

        self.move_action(n_steps+20, q_pos_pick_reach, 0)

        self.move_action(n_steps+10, q_pos_pick, 0, mid_noise=0.001, noise=0.001)

        self.open()
        ###################################################

        self.move_action(n_steps, q_pos_pick_reach, 0.8, mid_noise=0.01, noise=0.01)


        final_pose = torch.tensor([0, -3.32, 3.11, 1.18, 0, -0.174], device=self.device)
        
        self.move_action(n_steps, final_pose[:5], -0.174)
            
        if math.sqrt((self.cube_2.get_dofs_position()[0] - self.cube_1.get_dofs_position()[0])**2 + (self.cube_2.get_dofs_position()[1] - self.cube_1.get_dofs_position()[1])**2) < 0.03:
            self.succ = True
        else:
            self.succ = False

    def reset(self):
        self.arm.set_dofs_position([0,-3.32, 3.11, 1.18, 0, -0.174])

        r_min = 0.25
        r_max = 0.31

        theta_banana = random.uniform(0, 1*math.pi/6) 
        r_banana = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_cube1 = r_banana * math.cos(theta_banana)
        y_cube1 = r_banana * math.sin(theta_banana)

        r_min = 0.25
        r_max = 0.3
        theta_box = random.uniform(-1*math.pi/6, 0) 
        r_box = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_cube2 = r_box * math.cos(theta_box)
        y_cube2 = r_box * math.sin(theta_box)

        pos_cube1 = np.r_[x_cube1, y_cube1, 0.04]
        pos_cube2 = np.r_[x_cube2, y_cube2, 0.04]

        r_min = 0.2
        r_max = 0.4

        theta_banana = random.uniform(-1*math.pi/3, 1*math.pi/3) 
        r_banana = math.sqrt(random.uniform(r_min**2, r_max**2))
        x_cube1 = r_banana * math.cos(theta_banana)
        y_cube1 = r_banana * math.sin(theta_banana)

        r_min = 0.2
        r_max = 0.4
        theta_box = random.uniform(-1*math.pi/3, 1*math.pi/3) 
        r_box = math.sqrt(random.uniform(r_min**2, r_max**2))  
        x_cube2 = r_box * math.cos(theta_box)
        y_cube2 = r_box * math.sin(theta_box)

        pos_banana = np.r_[x_cube1, y_cube1, 0.04]
        pos_toy = np.r_[x_cube2, y_cube2, 0.04]

        # if math.sqrt((pos_banana[0] - pos_cube1[0])**2 + (pos_banana[1] - pos_cube1[1])**2) < 0.075 or \
        #    math.sqrt((pos_banana[0] - pos_cube2[0])**2 + (pos_banana[1] - pos_cube2[1])**2) < 0.075 or \
        #    math.sqrt((pos_toy[0] - pos_cube2[0])**2 + (pos_toy[1] - pos_cube2[1])**2) < 0.075 or \
        #    math.sqrt((pos_toy[0] - pos_cube1[0])**2 + (pos_toy[1] - pos_cube1[1])**2) < 0.075 or \
        #    math.sqrt((pos_banana[0] - pos_toy[0])**2 + (pos_banana[1] - pos_toy[1])**2) < 0.1:
        #     return self.reset()

        if math.sqrt((pos_cube1[0] - pos_cube2[0])**2 + (pos_cube1[1] - pos_cube2[1])**2) < 0.075:
            return self.reset()

        self.cube_1.set_quat(Rotation.from_euler('xyz', [random.randint(0,360), 0, 0], degrees=True).as_quat())
        self.cube_2.set_quat(Rotation.from_euler('xyz', [random.randint(0,360), 0, 0], degrees=True).as_quat())

        self.cube_1.set_pos(pos_cube1)
        self.cube_2.set_pos(pos_cube2)

        
        a = random.uniform(0.21, 0.26)
        b = random.uniform(-0.02, 0.02)
        noise1 = [a, b, 0.0001]
        # noise2 = [a, -0.11, 0.0001]
        self.scene_noise1.set_pos(noise1)
        # self.scene_noise2.set_pos(noise2)
        # self.scene_noise1 =self.scene.add_entity(
        #     gs.morphs.Box(
        #         pos=(0.3, 0.15, 0.0001),
        #         size=(0.3, 0.3, 0.0001),
        #         fixed=True,
        #         collision=True,
        #     ),
        #     material=gs.materials.Rigid(),
        #     surface=gs.surfaces.Default(
        #         color=(0.5, 0.5, 0.5),
        #         vis_mode='visual'
        #     ),
        # )


        # self.scene_noise2.set_color([1,1,1])

        # range = [(40, 140), (220, 320)]
        # chose_range = random.choice(range)
        # self.banana.set_quat(Rotation.from_euler('xyz', [random.randint(chose_range[0], chose_range[1]), 0, 0], degrees=True).as_quat())
        # range = [(-50, 50), (130, 230)]
        # # range = [(160, 200)]
        # chose_range = random.choice(range)
        # # self.toy.set_quat(self.origin_euler)
        # self.toy.set_quat(Rotation.from_euler('xyz', [random.randint(chose_range[0], chose_range[1]), 0, 0], degrees=True).as_quat())
        # self.banana.set_pos(pos_banana)
        # self.toy.set_pos(pos_toy)

        self.step = 0
        self.scene.step()                                       
        self.case += 1
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = get_args()
    collector = StackCubes(case=args.start,use_gs=args.use_gs,data_augmentation=args.data_augmentation,
                           save_dir=args.save_dir,reset_cam=args.reset_cam,single_view=args.single_view,
                           use_robot_gs=args.use_robot_gs)
    collector.run(num_steps=args.num_steps)