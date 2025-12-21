"""
Find Robot Initial Pose for Segmentation
==========================================

This script helps find the correct initial joint angles that match robot.ply.

Usage:
    python scripts/find_robot_pose.py

Then paste joint angles from pose_adjuster.html
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import genesis as gs
import numpy as np
import open3d as o3d
from Gaussians.util_gau import load_ply
from robot_gaussian.forward_kinematics import LINK_NAMES


def get_urdf_point_cloud(arm, link_names):
    """Extract all URDF mesh vertices."""
    all_points = []
    all_colors = []

    link_colors = [
        [1.0, 0.0, 0.0],  # Base - Red
        [0.0, 1.0, 0.0],  # Rotation_Pitch - Green
        [0.0, 0.0, 1.0],  # Upper_Arm - Blue
        [1.0, 1.0, 0.0],  # Lower_Arm - Yellow
        [1.0, 0.0, 1.0],  # Wrist_Pitch_Roll - Magenta
        [0.0, 1.0, 1.0],  # Fixed_Jaw - Cyan
        [1.0, 0.5, 0.0],  # Moving_Jaw - Orange
    ]

    for i, name in enumerate(link_names):
        link = arm.get_link(name)
        verts = link.get_vverts().cpu().numpy()
        all_points.append(verts)
        all_colors.append(np.tile(link_colors[i], (len(verts), 1)))

    return np.vstack(all_points), np.vstack(all_colors)


def visualize_alignment(urdf_points, urdf_colors, robot_ply_points, joint_angles):
    """Visualize URDF mesh and robot.ply together."""
    pcd_urdf = o3d.geometry.PointCloud()
    pcd_urdf.points = o3d.utility.Vector3dVector(urdf_points)
    pcd_urdf.colors = o3d.utility.Vector3dVector(urdf_colors)

    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(robot_ply_points)
    pcd_robot.paint_uniform_color([0.7, 0.7, 0.7])

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    print("\n" + "=" * 60)
    print("Visualization Window")
    print("=" * 60)
    print("\nLegend:")
    print("  彩色点 = URDF mesh (按link着色)")
    print("  灰色点 = robot.ply")
    print("\n如果对齐正确:")
    print("  - 彩色和灰色点应该重叠")
    print("  - 形状和关节位置相同")
    print(f"\n当前关节角度: {[f'{a:.3f}' for a in joint_angles]}")
    print("\n关闭窗口继续...")

    o3d.visualization.draw_geometries(
        [pcd_urdf, pcd_robot, coord_frame],
        window_name="URDF (彩色) vs robot.ply (灰色)",
        width=1200,
        height=800
    )


def try_pose(arm, scene, joint_angles, robot_ply_points, pose_name):
    """Try a specific pose and visualize."""
    print(f"\n{'='*60}")
    print(f"测试姿态: {pose_name}")
    print(f"关节角度: {[f'{a:.3f}' for a in joint_angles]}")
    print(f"{'='*60}")

    arm.set_dofs_position(joint_angles)
    scene.step()

    urdf_points, urdf_colors = get_urdf_point_cloud(arm, LINK_NAMES)
    visualize_alignment(urdf_points, urdf_colors, robot_ply_points, joint_angles)


def main():
    print("=" * 60)
    print("寻找机械臂初始姿态")
    print("=" * 60)

    # Initialize Genesis
    print("\n[1/3] 初始化Genesis...")
    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(substeps=60),
        renderer=gs.renderers.Rasterizer(),
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

    # Load robot.ply
    print("\n[2/3] 加载robot.ply...")
    robot_gau = load_ply('exports/mult-view-scene/robot.ply')
    robot_ply_points = robot_gau.xyz
    print(f"   加载了 {len(robot_ply_points)} 个Gaussians")

    # Try different poses
    print("\n[3/3] 测试不同姿态...")

    poses = [
        ([0, 0, 0, 0, 0, 0], "零位姿态"),
        ([0, -3.32, 3.11, 1.18, 0, -0.174], "当前初始姿态"),
        ([0, -np.pi/4, np.pi/4, 0, 0, 0], "Home姿态"),
    ]

    for joint_angles, pose_name in poses:
        try_pose(arm, scene, joint_angles, robot_ply_points, pose_name)

    # Manual input
    print("\n" + "=" * 60)
    print("手动调整")
    print("=" * 60)
    print("\n推荐: 用浏览器打开 scripts/pose_adjuster.html")
    print("      调整滑块后复制角度值粘贴到这里\n")
    print("格式: j0 j1 j2 j3 j4 j5 (空格或逗号分隔，弧度)")
    print("示例: 0 -3.32 3.11 1.18 0 -0.174")
    print("示例: 0, 0, 0, 0, 0, 0")
    print("\n按Enter跳过")

    user_input = input("\n关节角度: ").strip()

    if user_input:
        try:
            user_input = user_input.replace(',', ' ')
            manual_angles = [float(x) for x in user_input.split()]

            if len(manual_angles) != 6:
                print(f"错误: 需要6个角度值 (当前{len(manual_angles)}个)")
            else:
                try_pose(arm, scene, manual_angles, robot_ply_points, "手动输入")

                confirm = input("\n这个姿态正确吗? (y/n): ").strip().lower()
                if confirm == 'y':
                    print("\n" + "=" * 60)
                    print("✅ 更新以下文件中的初始姿态:")
                    print("=" * 60)
                    print(f"\nINITIAL_JOINTS = {manual_angles}")
                    print("\n需要更新的文件:")
                    print("  1. scripts/segment_robot.py (line 71)")
                    print("  2. robot_gaussian/robot_gaussian_model.py (line 424)")
                    print("  3. tasks/base_task.py (_init_robot_gs, line 434)")
                    print("\n然后重新运行:")
                    print("  python scripts/segment_robot.py")
                    print("  python scripts/visualize_segmentation.py")
        except ValueError as e:
            print(f"错误: 输入格式错误 - {e}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
