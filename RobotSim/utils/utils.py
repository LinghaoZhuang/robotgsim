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
import torch
from PIL import Image
import cv2
from natsort import natsorted
import trimesh
import glob
import re

def generate_joint_points(parent_link, child_link, n_points=10000, radius=0.01):
    parent_pos = parent_link.get_pos().cpu()
    parent_quat = parent_link.get_quat().cpu()
    
    child_pos = child_link.get_pos().cpu()
    child_quat = child_link.get_quat().cpu()
    
    joint_center = (parent_pos + child_pos) / 2.0
    
    parent_rot = Rotation.from_quat(parent_quat).as_matrix()
    parent_dir = parent_rot @ np.array([0, 0, 1])  
    
    child_rot = Rotation.from_quat(child_quat).as_matrix()
    child_dir = child_rot @ np.array([0, 0, 1])
    
    joint_normal = (parent_dir + child_dir) / 2.0
    joint_normal = joint_normal / np.linalg.norm(joint_normal)
    
    points = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        
        radial_offset = radius * np.array([
            np.cos(angle) * np.sin(np.pi/4), 
            np.sin(angle) * np.sin(np.pi/4),
            0
        ])
        
        depth_offset = 0.03 * np.array([0, 0, (i % 2 - 0.5)])  # 沿z轴错位分布
        
        point = joint_center + radial_offset + depth_offset
        points.append(point)
    
    return np.array(points)

def process_joint_connections(arm_entity, point_cloud):
    # 定义机械臂层级关系
    joint_links = [
        ("Base", "Rotation_Pitch"), 
        ("Rotation_Pitch", "Upper_Arm"),
        ("Upper_Arm", "Lower_Arm"),
        ("Lower_Arm", "Wrist_Pitch_Roll"),
        ("Wrist_Pitch_Roll", "Fixed_Jaw"),
        ("Fixed_Jaw", "Moving_Jaw")
    ]
    
    connection_points = []
    
    for parent_name, child_name in joint_links:
  
        parent_link = arm_entity.get_link(parent_name)
        child_link = arm_entity.get_link(child_name)
        
        if parent_link is None or child_link is None:
            continue

        joint_points = generate_joint_points(parent_link, child_link)
        
        if joint_points is not None:
            connection_points.append(joint_points)
            print(f"在 {parent_name}-{child_name} 关节处生成 {len(joint_points)} 个连接点")


    if connection_points:
        connection_points = np.vstack(connection_points)
        print(f"总共生成 {len(connection_points)} 个连接点")
        return np.vstack([point_cloud, connection_points])
    
    return point_cloud

def export_robot_point_cloud(arm_entity, filename="robot_point_cloud.ply"):
    import numpy as np
    import pandas as pd
    from pyntcloud import PyntCloud
    
    point_cloud = []
    
    # 机械臂所有链接名称列表
    link_names = [
        "Base", "Rotation_Pitch", "Upper_Arm", "Lower_Arm", 
        "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw"
    ]
    
    print(f"开始导出机械臂点云，链接数量: {len(link_names)}")
    
    for name in link_names:
        try:
            link = arm_entity.get_link(name)
            if link is None:
                print(f"警告: 未找到链接 '{name}'")
                continue
        
            vertices = None
            
            print(f"正在处理链接 '{name}'，尝试获取可视化顶点...")
            try:
                vertices = link.get_vverts()
                if vertices is not None and len(vertices) > 0:
                    print(f"  成功获取 {len(vertices)} 个可视化顶点")
            except Exception as e:
                print(f"  无法获取可视化顶点: {str(e)}")
                vertices = None
            
            if vertices is None or len(vertices) == 0:
                print(f"尝试获取碰撞顶点...")
                try:
                    vertices = link.get_verts()
                    if vertices is not None and len(vertices) > 0:
                        print(f"  成功获取 {len(vertices)} 个碰撞顶点")
                except Exception as e:
                    print(f"  无法获取碰撞顶点: {str(e)}")
                    continue
            
            for vertex in vertices:
                v = vertex if isinstance(vertex, np.ndarray) else np.array(vertex.cpu())
                try:
                    world_vertex = gs.math.transform_point(pos, quat, v)
                    point_cloud.append(world_vertex[:3])
                except:
                    point_cloud.append(v[:3])
                    
        except Exception as e:
            import traceback
            print(f"处理链接 '{name}' 时出错: {str(e)}")
            traceback.print_exc()
            continue
            
    if not point_cloud:
        print("错误: 未收集到任何点云数据")
        return
    
    point_cloud = np.array(point_cloud)
    
    print("点云统计信息:")
    print(f"  总点数: {len(point_cloud)}")
    print(f"  X范围: [{np.min(point_cloud[:,0]):.4f}, {np.max(point_cloud[:,0]):.4f}]")
    print(f"  Y范围: [{np.min(point_cloud[:,1]):.4f}, {np.max(point_cloud[:,1]):.4f}]")
    print(f"  Z范围: [{np.min(point_cloud[:,2]):.4f}, {np.max(point_cloud[:,2]):.4f}]")
    
    # complete_point_cloud = process_joint_connections(arm_entity, point_cloud)
    # print("complete_point_cloud:")
    # print(f"  总点数: {len(complete_point_cloud)}")
    # print(f"  X范围: [{np.min(complete_point_cloud[:,0]):.4f}, {np.max(complete_point_cloud[:,0]):.4f}]")
    # print(f"  Y范围: [{np.min(complete_point_cloud[:,1]):.4f}, {np.max(complete_point_cloud[:,1]):.4f}]")
    # print(f"  Z范围: [{np.min(complete_point_cloud[:,2]):.4f}, {np.max(complete_point_cloud[:,2]):.4f}]")

    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
    cloud = PyntCloud(df)
    cloud.to_file(filename)
    print(f"已保存机械臂点云到: {filename}")

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

def img2video(path='collected_data/2/0', save_path='output.mp4', fps=30):
    image_folder = path
    video_name = save_path
    fps = fps

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort()  
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)  # 自动处理数字排序

    # print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def collision_obj(input_file_path, output_file_path):

    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 核心处理逻辑 ---
    print(f"正在加载模型: {input_file_path}")
    # 1. 加载原始的、高精度的模型
    original_mesh = trimesh.load_mesh(input_file_path)

    # 2. 计算其凸包 - 这是最关键的一步！
    #    .convex_hull 会返回一个新的trimesh对象，它就是原始模型的凸包
    convex_hull_mesh = original_mesh.convex_hull

    print("凸包计算完成。")
    print(f"原始模型面数: {len(original_mesh.faces)}")
    print(f"凸包模型面数: {len(convex_hull_mesh.faces)}")
    convex_hull_mesh.export(output_file_path)
    print(f"已成功将凸包碰撞体保存到: {output_file_path}")

    convex_mesh = trimesh.load_mesh(output_file_path)
    assert convex_mesh.is_convex, "碰撞体必须为凸包！"

    # 强制二次凸化处理（防止原始凸包质量问题）
    super_convex = convex_mesh.convex_hull
    super_convex.export("banana_collision_final.obj")

import argparse
def get_args(benchmark=False, use_rlg_config=False):
    parser = argparse.ArgumentParser(description="RoboSimGS Training and Evaluation Arguments")
    parser.add_argument("--start", type=int, default=0, help="")
    parser.add_argument("--num_steps", type=int, default=100, help="")
    parser.add_argument("--use_gs", default=True, help="")
    parser.add_argument("--data_augmentation", action='store_true', help="")
    parser.add_argument("--save_dir", default="collected_data", help="")
    parser.add_argument("--reset_cam", type=float, default=0.01, help="")
    parser.add_argument("--single_view", action='store_true', help="")
    parser.add_argument("--use_robot_gs", action='store_true', help="Pure 3DGS rendering for robot (no overlay)")
    args = parser.parse_args()
    return args

def edit_mesh():
    import trimesh
    mesh = trimesh.load("assets/objects/bottle_thin/bottle.obj")

    # 2. 平移 mesh（例如沿 X 轴平移 1.0 单位）
    translation = [0.0, -0.10, 0.0]  # [x, y, z]
    mesh.apply_translation(translation)


    angle = np.radians(90)  # 角度转弧度
    rotation_matrix = np.array([
        [np.cos(angle),  0, np.sin(angle), 0],  # X 轴旋转分量
        [0,              1, 0,             0],  # Y 轴不变
        [-np.sin(angle), 0, np.cos(angle), 0],  # Z 轴旋转分量
        [0,              0, 0,             1]   # 齐次坐标
    ])
    mesh.apply_transform(rotation_matrix)
    # 3. 导出为新的 OBJ 文件
    mesh.export("assets/objects/bottle_thin/bottle1.obj")

def create_video_from_images(image_folder, output_video, fps=30, lossless=False, delete_after=True):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    images = sorted(
        glob.glob(os.path.join(image_folder, "*.png")) +
        glob.glob(os.path.join(image_folder, "*.jpg")) +
        glob.glob(os.path.join(image_folder, "*.jpeg")) +
        glob.glob(os.path.join(image_folder, "*.bmp")),
        key=natural_sort_key
    )
    
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    if lossless:
        fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print(f"错误: 无法创建视频文件 '{output_video}'")
        return False
    
    processed_count = 0
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
            
        video.write(img)
        processed_count += 1
        
    video.release()
    if delete_after:    
        deleted_count = 0
        for image_path in images:
            try:
                os.remove(image_path)
                deleted_count += 1
            except Exception as e:
                print(f"警告: 无法删除 '{image_path}': {e}")
        return True
    return True

def create_video_from_images_multi_view(image_folder, output_video, fps=30, lossless=True, delete_after=True, view='left'):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    images = sorted(
        glob.glob(os.path.join(image_folder, f"*_{view}.png")) +
        glob.glob(os.path.join(image_folder, f"*_{view}.jpg")),
        key=natural_sort_key
    )

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape
    
    if lossless:
        fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not video.isOpened():
        print(f"错误: 无法创建视频文件 '{output_video}'")
        return False
    
    processed_count = 0
    for i, image_path in enumerate(images):
        img = cv2.imread(image_path)
            
        video.write(img)
        processed_count += 1
        
    video.release()
    if delete_after:    
        deleted_count = 0
        for image_path in images:
            try:
                os.remove(image_path)
                deleted_count += 1
            except Exception as e:
                print(f"警告: 无法删除 '{image_path}': {e}")
        return True
    return True

# def create_video_from_images_multi_view(image_folder, output_video, fps=30, lossless=False, delete_after=True, view='left'):
#     os.makedirs(os.path.dirname(output_video), exist_ok=True)
#     def natural_sort_key(s):
#         return [int(text) if text.isdigit() else text.lower()
#                 for text in re.split(r'(\d+)', s)]
#     images = sorted(
#         glob.glob(os.path.join(image_folder, f"*_{view}.png")) +
#         glob.glob(os.path.join(image_folder, f"*_{view}.jpg")),
#         key=natural_sort_key
#     )

#     frame = cv2.imread(images[0])
#     height, width, _ = frame.shape

#     frame_array = np.empty((len(images), height, width, 3), dtype=np.uint8)
    
#     import tqdm
#     progress = tqdm.tqdm(total=len(images), desc="处理图像帧", leave=False)

#     for i, image_path in enumerate(images):
#         img = cv2.imread(image_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         frame_array[i] = img_rgb
#         progress.update(1)

#     progress.close()
#     np.save(output_video, frame_array)

#     if delete_after:    
#         deleted_count = 0
#         for image_path in images:
#             try:
#                 os.remove(image_path)
#                 deleted_count += 1
#             except Exception as e:
#                 print(f"警告: 无法删除 '{image_path}': {e}")
#         return True
#     return True

def extract_images_from_video(video_path, output_folder, image_format='png', start_index=0, zero_pad=6):
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'")
        return 0
    
    # 设置文件名格式
    file_template = f"{{:0{zero_pad}d}}.{image_format.lower()}"
    count = start_index
    
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存当前帧
        output_path = os.path.join(output_folder, file_template.format(count))
        success = cv2.imwrite(output_path, frame)
        
        if success:
            saved_count += 1
        else:
            print(f"警告: 无法保存图片 '{output_path}'")
        
        count += 1
    
    cap.release()
    print(f"从视频中成功提取 {saved_count} 帧图片，保存在 {output_folder}")
    return saved_count

if __name__ == "__main__":
    # img2video(path='collected_data/2/0', save_path='output.mp4', fps=30)
    # edit_mesh()
    # extract_images_from_video(video_path='/media/haoyu/BaseBig/code/RoboSimGS/collected_data/1/615/video.mp4', output_folder='/media/haoyu/BaseBig/code/RoboSimGS/collected_data/1/615/1', image_format='png', start_index=0, zero_pad=4)
    create_video_from_images(image_folder='/media/haoyu/BaseBig/code/RoboSimGS/collected_data/9/326', output_video='/media/haoyu/BaseBig/code/RoboSimGS/collected_data/9/326/video.mp4', fps=30)
    # collision_obj('/home/haoyu/code/RoboSimGS/assets/objects/banana/banana.obj', '/home/haoyu/code/RoboSimGS/assets/objects/banana/banana_collision.obj')

