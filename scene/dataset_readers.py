#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from scene.colmap_loader import rotmat2qvec
import numpy as np
import cv2
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import utils.camera as camera
import torch
import pickle

from utils.darf_noise import generate_noisy_pose, matrix_to_euler_angles, euler_angles_to_matrix

from duster.dust3r.inference import inference, load_model
from duster.dust3r.utils.image import load_images
from duster.dust3r.image_pairs import make_pairs
from duster.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from duster.dust3r.utils.device import to_numpy


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def init_filestructure(save_path):
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)
    
    images_path = save_path / 'images'
    masks_path = save_path / 'masks'
    sparse_path = save_path / 'sparse/0'
    
    images_path.mkdir(exist_ok=True, parents=True)
    masks_path.mkdir(exist_ok=True, parents=True)    
    sparse_path.mkdir(exist_ok=True, parents=True)
    
    return save_path, images_path, masks_path, sparse_path

def save_images_masks(imgs, masks, images_path, masks_path):
    # Saving images and optionally masks/depth maps
    for i, (image, mask) in enumerate(zip(imgs, masks)):
        image_save_path = images_path / f"{i}.png"
        
        mask_save_path = masks_path / f"{i}.png"
        # image[~mask] = 1.
        rgb_image = cv2.cvtColor(image*255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)
        
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2)*255
        Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)
        
        
def save_cameras(focals, principal_points, sparse_path, imgs_shape):
    # Save cameras.txt
    cameras_file = sparse_path / 'cameras.txt'
    with open(cameras_file, 'w') as cameras_file:
        cameras_file.write("# Camera list with one line of data per camera:\n")
        cameras_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, (focal, pp) in enumerate(zip(focals, principal_points)):
            cameras_file.write(f"{i} PINHOLE {imgs_shape[2]} {imgs_shape[1]} {focal[0]} {focal[0]} {pp[0]} {pp[1]}\n")
    
def save_imagestxt(world2cam, sparse_path):
     # Save images.txt
    images_file = sparse_path / 'images.txt'
    # Generate images.txt content
    with open(images_file, 'w') as images_file:
        images_file.write("# Image list with two lines of data per image:\n")
        images_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        images_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(world2cam.shape[0]):
            # Convert rotation matrix to quaternion
            rotation_matrix = world2cam[i, :3, :3]
            qw, qx, qy, qz = rotmat2qvec(rotation_matrix)
            tx, ty, tz = world2cam[i, :3, 3]
            images_file.write(f"{i} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i} {i}.png\n")
            images_file.write("\n") # Placeholder for points, assuming no points are associated with images here

def save_pointcloud_with_normals(imgs, pts3d, msk, sparse_path):
    pc, pts, col = get_pc(imgs, pts3d, msk)  # Assuming get_pc is defined elsewhere and returns a trimesh point cloud

    # Define a default normal, e.g., [0, 1, 0]
    default_normal = [0, 1, 0]

    # Prepare vertices, colors, and normals for saving
    vertices = pc.vertices
    colors = pc.colors
    normals = np.tile(default_normal, (vertices.shape[0], 1))

    save_path = sparse_path / 'points3D.ply'

    # Construct the header of the PLY file
    header = """ply
format ascii 1.0
element vertex {}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
""".format(len(vertices))

    # Write the PLY file
    with open(save_path, 'w') as ply_file:
        ply_file.write(header)
        for vertex, color, normal in zip(vertices, colors, normals):
            ply_file.write('{} {} {} {} {} {} {} {} {}\n'.format(
                vertex[0], vertex[1], vertex[2],
                normal[0], normal[1], normal[2],
                int(color[0]), int(color[1]), int(color[2]),
            ))
        
import trimesh
def get_pc(imgs, pts3d, mask):
    imgs = to_numpy(imgs)
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    
    # no masking
    # pts = np.concatenate([p[np.ones_like(m, dtype=bool)] for p, m in zip(pts3d, mask)])
    # col = np.concatenate([p[np.ones_like(m, dtype=bool)] for p, m in zip(imgs, mask)])

    pts = pts.reshape(-1, 3)[::3]
    col = col.reshape(-1, 3)[::3]
    
    #mock normals:
    normals = np.tile([0, 1, 0], (pts.shape[0], 1))
    
    pct = trimesh.PointCloud(pts, colors=col)
    pct.vertices_normal = normals  # Manually add normals to the point cloud
    
    return pct, pts, col

def save_pointcloud(imgs, pts3d, msk, sparse_path):
    save_path = sparse_path / 'points3D.ply'
    pc, pts, col = get_pc(imgs, pts3d, msk)
    
    pc.export(save_path)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE" or intr.model=="OPENCV" or intr.model=="RADIAL":
            print(intr.model)
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))


    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def runDust3r(image_path, cam_infos, device, args_dict):
    # Run dust3r
    ply_path = None
    if args_dict is not None:
        model_path = "./duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
        device = 'cuda'
        batch_size = 1
        schedule = 'cosine'
        lr = 0.01
        niter = 300
        model = load_model(model_path, device)
        image_path = image_path if args_dict["own_data"] else [img_path.image_path for img_path in cam_infos] 
        images = load_images(image_path, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    
    del model, output, images, pairs

    return scene

def readColmapSceneInfo(path, images, eval, llffhold=8, args_dict=None, log_dir=None):
    image_path = None
    if args_dict['own_data']:
        # read all the image paths from the images folder
        image_path = [os.path.join(path, img) for img in os.listdir(path) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".JPG")]
        train_cam_infos, test_cam_infos, cam_infos, cam_infos_unsorted, cam_extrinsics, cam_intrinsics = None, None, None, None, None, None
    else:
        sfm_path = path.split("_dust3r")[0]
        # construct colmap dataset from dust3r
        try:
            cameras_extrinsic_file = os.path.join(sfm_path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(sfm_path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(sfm_path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(sfm_path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        reading_dir = "images" if images == None else images
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(sfm_path, reading_dir))
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name))

        if args_dict is not None and args_dict['few_shot'] != -1:
            indices = []
            dataset_length = len(cam_infos)
            num_samples = args_dict['few_shot']*2
            index = np.linspace(0, dataset_length-1, num_samples, dtype=int)
            cam_infos = [cam_infos[i] for i in index]
            print(f"Training with few-shot! The selected views are: {index}")

        if eval:
            full_length = len(cam_infos)
            train_cam_infos = [cam_infos[i]  for i in range(full_length) if i % 2 == 0]
            test_cam_infos = [cam_infos[i]  for i in range(full_length) if i % 2 == 1]
        else:
            full_length = len(cam_infos)
            train_cam_infos = [cam_infos[i]  for i in range(full_length) if i % 2 == 0]
            test_cam_infos = []

        
        with open(f'{log_dir}/cam_gt.pkl', 'wb') as f:
            pickle.dump([(c.R, c.T, c.FovY, c.FovX, c.width, c.height,c.image_name) for c in train_cam_infos], f)
    
    # # Run dust3r
    # ply_path = None
    # if args_dict is not None:
    #     model_path = "./duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    #     device = 'cuda'
    #     batch_size = 1
    #     schedule = 'cosine'
    #     lr = 0.01
    #     niter = 300
    #     model = load_model(model_path, device)
    #     image_path = image_path if args_dict["own_data"] else [img_path.image_path for img_path in train_cam_infos] 
    #     images = load_images(image_path, size=512)
    #     pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    #     output = inference(pairs, model, device, batch_size=batch_size)

    #     scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    #     loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            
    def inv(mat):
        """ Invert a torch or numpy matrix
        """
        if isinstance(mat, torch.Tensor):
            return torch.linalg.inv(mat)
        if isinstance(mat, np.ndarray):
            return np.linalg.inv(mat)
        raise ValueError(f'bad matrix type = {type(mat)}')
    
    # Run dust3r
    scene = runDust3r(image_path, train_cam_infos, "cuda", args_dict)

    if eval:
        test_scene = runDust3r(image_path, test_cam_infos, "cuda", args_dict)

        # Make test_cam_infos from dust3r
        test_world2cam = inv(test_scene.get_im_poses().detach()).cpu().numpy()
        test_focals = test_scene.get_focals().detach().cpu().numpy()
        for idx in range(len(test_cam_infos)):
            FovY = focal2fov(test_focals[idx], test_scene.imgs[idx].shape[0])
            FovX = focal2fov(test_focals[idx], test_scene.imgs[idx].shape[1])
            test_cam_infos[idx] = test_cam_infos[idx]._replace(R=test_world2cam[idx, :3, :3].transpose(), T=test_world2cam[idx, :3, 3], FovY=FovY, FovX=FovX)

    intrinsics = scene.get_intrinsics().detach().cpu().numpy()
    world2cam = inv(scene.get_im_poses().detach()).cpu().numpy()
    principal_points = scene.get_principal_points().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()
    imgs = np.array(scene.imgs)
    gt_imgs = np.array([np.array(c.image) for c in train_cam_infos], dtype=np.float32) if not args_dict["own_data"] else np.array([np.array(Image.open(img)) for img in image_path], dtype=np.float32)
    gt_imgs = gt_imgs / 255.0
    
    pts3d = [i.detach() for i in scene.get_pts3d()]
    depth_maps = [i.detach() for i in scene.get_depthmaps()]

    min_conf_thr = 0  # instant splat does not use this maybe... (horse with thr 20 = PSNR 23.5, with thr 0 = PSNR 25.3)
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    masks = to_numpy(scene.get_masks())
    
    save_dir = Path(path)
    save_dir.mkdir(exist_ok=True, parents=True)

    save_path, images_path, masks_path, sparse_path = init_filestructure(save_dir)

    train_imgs = imgs
    if args_dict['og_scale']:
        print("Using original scale")
        dust3r_img_shape = train_imgs.shape
        # principal_points = np.array([[gt_imgs.shape[2]//2, gt_imgs.shape[1]//2] for _ in range(len(gt_imgs))], dtype=np.float32)  # use gt imgs
        train_imgs = gt_imgs

    save_images_masks(train_imgs, masks, images_path, masks_path)
    save_cameras(focals, principal_points, sparse_path, imgs_shape=dust3r_img_shape)
    # save_cameras(focals, principal_points, sparse_path, imgs_shape=train_imgs.shape)
    save_imagestxt(world2cam, sparse_path)
    # save_pointcloud(imgs, pts3d, masks, sparse_path)
    save_pointcloud_with_normals(imgs, pts3d, masks, sparse_path)
    
    del scene, images, principal_points, focals, imgs, pts3d, masks, depth_maps, intrinsics, train_cam_infos, cam_infos, cam_infos_unsorted, cam_extrinsics, cam_intrinsics

    # make empty points3D.txt file
    with open(sparse_path / 'points3D.txt', 'w') as f:
        f.write("")
    # make .bin files
    os.system(f"colmap model_converter --input_path {sparse_path} --output_path {sparse_path} --output_type BIN")
    # delete points3D.txt file
    os.remove(sparse_path / 'points3D.txt')
    os.remove(sparse_path / 'points3D.bin')

    # read colmap dataset
    path = save_dir
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name))

    train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
}