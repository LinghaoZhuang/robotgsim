#!/usr/bin/env python3
"""
验证坐标系和Genesis API行为
"""
import sys
from pathlib import Path

# 尝试导入依赖
try:
    import numpy as np
    from plyfile import PlyData
    import genesis as gs
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("\n请在有numpy/genesis的环境中运行此脚本")
    print("例如: conda activate genesis_env")
    sys.exit(1)

print("=" * 60)
print("问题1: robot.ply是否和背景PLY在同一world坐标系?")
print("=" * 60)

# 检查robot.ply和背景PLY的坐标范围
robot = PlyData.read('exports/mult-view-scene/robot.ply')
left = PlyData.read('exports/mult-view-scene/left-transform2.ply')
right = PlyData.read('exports/mult-view-scene/right-transform.ply')

robot_xyz = np.stack([robot.elements[0]['x'], robot.elements[0]['y'], robot.elements[0]['z']], axis=1)
left_xyz = np.stack([left.elements[0]['x'], left.elements[0]['y'], left.elements[0]['z']], axis=1)
right_xyz = np.stack([right.elements[0]['x'], right.elements[0]['y'], right.elements[0]['z']], axis=1)

print('\n[robot.ply]坐标范围:')
print(f'  min: [{robot_xyz.min(axis=0)[0]:.4f}, {robot_xyz.min(axis=0)[1]:.4f}, {robot_xyz.min(axis=0)[2]:.4f}]')
print(f'  max: [{robot_xyz.max(axis=0)[0]:.4f}, {robot_xyz.max(axis=0)[1]:.4f}, {robot_xyz.max(axis=0)[2]:.4f}]')
print(f'  mean: [{robot_xyz.mean(axis=0)[0]:.4f}, {robot_xyz.mean(axis=0)[1]:.4f}, {robot_xyz.mean(axis=0)[2]:.4f}]')

print('\n[left-transform2.ply]坐标范围:')
print(f'  min: [{left_xyz.min(axis=0)[0]:.4f}, {left_xyz.min(axis=0)[1]:.4f}, {left_xyz.min(axis=0)[2]:.4f}]')
print(f'  max: [{left_xyz.max(axis=0)[0]:.4f}, {left_xyz.max(axis=0)[1]:.4f}, {left_xyz.max(axis=0)[2]:.4f}]')
print(f'  mean: [{left_xyz.mean(axis=0)[0]:.4f}, {left_xyz.mean(axis=0)[1]:.4f}, {left_xyz.mean(axis=0)[2]:.4f}]')

print('\n[right-transform.ply]坐标范围:')
print(f'  min: [{right_xyz.min(axis=0)[0]:.4f}, {right_xyz.min(axis=0)[1]:.4f}, {right_xyz.min(axis=0)[2]:.4f}]')
print(f'  max: [{right_xyz.max(axis=0)[0]:.4f}, {right_xyz.max(axis=0)[1]:.4f}, {right_xyz.max(axis=0)[2]:.4f}]')
print(f'  mean: [{right_xyz.mean(axis=0)[0]:.4f}, {right_xyz.mean(axis=0)[1]:.4f}, {right_xyz.mean(axis=0)[2]:.4f}]')

# 检查坐标量级
robot_mag = np.linalg.norm(robot_xyz.mean(axis=0))
left_mag = np.linalg.norm(left_xyz.mean(axis=0))
right_mag = np.linalg.norm(right_xyz.mean(axis=0))

print(f'\n坐标量级 (mean norm):')
print(f'  robot={robot_mag:.3f}, left={left_mag:.3f}, right={right_mag:.3f}')

if abs(robot_mag - left_mag) < 0.5 and abs(robot_mag - right_mag) < 0.5:
    print('\n✓ 结论: 量级接近，robot.ply可能已在同一world坐标系')
else:
    print('\n✗ 结论: 量级差异大，robot.ply可能不在同一坐标系，需要splat_to_world变换')

print("\n" + "=" * 60)
print("问题2: link.get_vverts()返回world还是local坐标?")
print("=" * 60)

# 初始化Genesis
gs.init(backend=gs.gpu)
scene = gs.Scene(
    show_viewer=False,
    vis_options=gs.options.VisOptions(show_world_frame=True),
)

scene.add_entity(gs.morphs.Plane())

arm = scene.add_entity(
    morph=gs.morphs.MJCF(
        file="./assets/so100/urdf/so_arm100.xml",
        euler=(0.0, 0.0, 90.0),
        pos=(0.0, 0.0, 0.0),
    ),
    material=gs.materials.Rigid(),
)

scene.build()

# 设置初始姿态
INITIAL_JOINTS = [0, -3.32, 3.11, 1.18, 0, -0.174]
arm.set_dofs_position(INITIAL_JOINTS)
scene.step()

# 检查Base link
print('\n[Base link测试]')
base = arm.get_link("Base")
base_verts = base.get_vverts().cpu().numpy()
base_pos = base.get_pos().cpu().numpy()
base_quat = base.get_quat().cpu().numpy()

print(f'  Link pos: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]')
print(f'  Link quat: [{base_quat[0]:.4f}, {base_quat[1]:.4f}, {base_quat[2]:.4f}, {base_quat[3]:.4f}]')
print(f'  Verts shape: {base_verts.shape}')
print(f'  Verts range:')
print(f'    min: [{base_verts.min(axis=0)[0]:.4f}, {base_verts.min(axis=0)[1]:.4f}, {base_verts.min(axis=0)[2]:.4f}]')
print(f'    max: [{base_verts.max(axis=0)[0]:.4f}, {base_verts.max(axis=0)[1]:.4f}, {base_verts.max(axis=0)[2]:.4f}]')
print(f'    mean: [{base_verts.mean(axis=0)[0]:.4f}, {base_verts.mean(axis=0)[1]:.4f}, {base_verts.mean(axis=0)[2]:.4f}]')

# 检查Rotation_Pitch (有偏移的link)
print('\n[Rotation_Pitch link测试]')
rot_pitch = arm.get_link("Rotation_Pitch")
rot_verts = rot_pitch.get_vverts().cpu().numpy()
rot_pos = rot_pitch.get_pos().cpu().numpy()
rot_quat = rot_pitch.get_quat().cpu().numpy()

print(f'  Link pos: [{rot_pos[0]:.4f}, {rot_pos[1]:.4f}, {rot_pos[2]:.4f}]')
print(f'  Link quat: [{rot_quat[0]:.4f}, {rot_quat[1]:.4f}, {rot_quat[2]:.4f}, {rot_quat[3]:.4f}]')
print(f'  Verts shape: {rot_verts.shape}')
print(f'  Verts range:')
print(f'    min: [{rot_verts.min(axis=0)[0]:.4f}, {rot_verts.min(axis=0)[1]:.4f}, {rot_verts.min(axis=0)[2]:.4f}]')
print(f'    max: [{rot_verts.max(axis=0)[0]:.4f}, {rot_verts.max(axis=0)[1]:.4f}, {rot_verts.max(axis=0)[2]:.4f}]')
print(f'    mean: [{rot_verts.mean(axis=0)[0]:.4f}, {rot_verts.mean(axis=0)[1]:.4f}, {rot_verts.mean(axis=0)[2]:.4f}]')

# 判断坐标系
verts_near_pos = np.linalg.norm(rot_verts.mean(axis=0) - rot_pos) < 0.3
verts_near_origin = np.linalg.norm(rot_verts.mean(axis=0)) < 0.1

print(f'\n判断依据:')
print(f'  |verts.mean - link.pos| = {np.linalg.norm(rot_verts.mean(axis=0) - rot_pos):.4f}')
print(f'  |verts.mean| = {np.linalg.norm(rot_verts.mean(axis=0)):.4f}')

if verts_near_pos:
    print('\n✓ 结论: get_vverts()返回WORLD坐标 (verts接近link pos)')
    print('  → KNN分割时直接使用verts，不需要变换')
elif verts_near_origin:
    print('\n✓ 结论: get_vverts()返回LOCAL坐标 (verts接近原点)')
    print('  → KNN分割时需要用link pose变换到world')
else:
    print('\n? 不确定，需要进一步检查')

print("\n" + "=" * 60)
print("附加信息: quaternion格式")
print("=" * 60)
print(f'Genesis link.get_quat()格式: {base_quat}')
print('  如果Base是identity且quat=[0,0,0,1]或[1,0,0,0]，可以判断格式')
print('  xyzw格式: [0,0,0,1] (w在最后)')
print('  wxyz格式: [1,0,0,0] (w在最前)')
