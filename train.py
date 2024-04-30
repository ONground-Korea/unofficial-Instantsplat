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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui, r6d2mat
import sys
from scene import Scene, GaussianModel, CameraPoses
from utils.general_utils import safe_state
import uuid
import numpy as np
import wandb
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_sets
import pickle
import camera_barf
from easydict import EasyDict as edict
import random
import cv2
import torchvision
import torch.nn as nn
from utils.schedular import ExponentialDecayScheduler

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations ,saving_iterations, checkpoint_iterations ,checkpoint, debug_from, args_dict):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, args_dict['output_path'], args_dict['exp_name'], args_dict['project_name'])
    gaussians = GaussianModel(dataset.sh_degree)
    
    if args_dict['ours']:
        divide_ratio = 0.7
    else:
        divide_ratio = 0.8
    print(f"Set divide_ratio to {divide_ratio}")
    
    if args_dict['pretrained_scene'] is not None:
        scene = Scene(dataset, gaussians,load_iteration=30000,args_dict=args_dict)
    else:
        scene = Scene(dataset, gaussians, args_dict=args_dict)

    gaussians.training_setup(opt) # opt contains densify from iter and densify until iter
    pose_network = CameraPoses(scene.getTrainCameras(), args_dict['pose_representation']).cuda()

    camera_pose_optimizer = torch.optim.AdamW(pose_network.parameters(), lr=1e-5, betas=(0.9, 0.999))
    camera_pose_scheduler = ExponentialDecayScheduler(0, 1000, 5e-7).get_scheduler(camera_pose_optimizer, 1e-5)
    # camera_pose_optimizer = torch.optim.SGD(pose_network.parameters(), lr=5e-6, momentum=0.9, weight_decay=1e-4)
    # camera_pose_scheduler = torch.optim.lr_scheduler.ExponentialLR(camera_pose_optimizer, gamma=0.97)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    random_index = torch.randperm(len(viewpoint_stack))

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1
    num_gaussians = []
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, render_only=True)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        num_gaussians.append(gaussians.get_xyz.shape[0])
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # if args_dict['DSV']:
        #     if iteration % 100 == 0:
        #         gaussians.oneupSHdegree()
        # elif args_dict['ours']:
        #     if iteration >= 5000:
        #         if iteration % 1000 == 0:
        #             gaussians.oneupSHdegree()
        
        if iteration % len(random_index) == 0:
            random_index = torch.randperm(len(pose_network))
        
        cam_idx = random_index[iteration % len(random_index)]
        viewpoint_cam = viewpoint_stack[cam_idx]
        
        ## camera pose learning
        camera_pose = pose_network(cam_idx)
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose = camera_pose,render_only=False, pose_rep = args_dict['pose_representation'])
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        # pose regularization: To maintain a close adherence to the original camera pose.
        original_campose = torch.eye(4, device="cuda")
        original_campose[:3,:3] = torch.tensor(viewpoint_cam.R.T, device="cuda")
        original_campose[:3,3] = torch.tensor(viewpoint_cam.T, device="cuda")
        pose_reg = l1_loss(camera_pose.T, original_campose)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.1 * pose_reg
        loss.backward()
        

        iter_end.record()

        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","pose_reg" : f"{pose_reg}" , "num_gaussians" : f"{gaussians.get_xyz.shape[0]}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Optimizer step
        if iteration < opt.iterations:
            camera_pose_optimizer.step()
            gaussians.optimizer.step()
            if args_dict['pose_representation'] == 'quaternion':
                pose_network.normalize_quat()
            camera_pose_optimizer.zero_grad(set_to_none = True)
            gaussians.optimizer.zero_grad(set_to_none = True)
            camera_pose_scheduler.step()
            # # Learning rate scheduler
            # if iteration % 1000 == 0:
            #     camera_pose_scheduler.step()

        # Log and save
        if iteration % 50 == 0:
            if args_dict['pose_representation'] == '9D':
                all_9d = pose_network.get_all_poses()
                all_w2c = torch.stack([r6d2mat(pose_9d.unsqueeze(dim=0)).unsqueeze(dim=0) for pose_9d in all_9d],dim=0).squeeze().transpose(1,2)
                all_R = all_w2c[:,:3,:3]
                all_t = all_w2c[:,:3,3]
            elif args_dict['pose_representation'] == 'quaternion':
                all_R, all_t = pose_network.get_all_poses()
            with torch.no_grad():
                if args_dict['own_data']:
                    rot_error, trans_error = 0.0, 0.0
                else:
                    rot_error, trans_error,sim3 = pose_error_report(tb_writer, iteration, scene, dataset.model_path, all_R,all_t)  
                    
                    
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),dataset.model_path,viewpoint_stack, rot_error, trans_error, pose_network=pose_network, sim3=sim3, args_dict=args_dict)

        with torch.no_grad():
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                np_num_gaussians = np.array(num_gaussians)
                np.save(scene.model_path + "/num_gaussians.npy", np_num_gaussians)

            # # Densification
            # if iteration < opt.densify_until_iter and args_dict['no_densify'] == False:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                
            #     if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    # Save video 
    # Slerp interpolation
    from scipy.spatial.transform import Rotation as R
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Slerp
    
    poses = pose_network.get_all_poses()
    all_w2c = torch.stack([r6d2mat(pose_9d.unsqueeze(dim=0)).unsqueeze(dim=0) for pose_9d in poses],dim=0).squeeze().transpose(1,2)
    all_R = all_w2c[:,:3,:3].clone().detach().cpu().numpy()
    all_t = all_w2c[:,:3,3].clone().detach().cpu().numpy()
    key_rots = R.from_matrix(all_R)
    key_trans = all_t

    key_times = [i for i in range(0, len(viewpoint_stack))]
    slerp = Slerp(key_times, key_rots)
    interp_func = interp1d(key_times, key_trans, axis=0)

    num_frames = 200
    times = np.linspace(0, len(viewpoint_stack)-1, num_frames)
    interp_rots = slerp(times).as_matrix()
    interp_trans = interp_func(times)
    interpolated_poses = torch.zeros((num_frames, 4, 4))
    for i in range(num_frames):
        interpolated_poses[i][:3,:3] = torch.tensor(interp_rots[i])
        interpolated_poses[i][:3,3] = torch.tensor(interp_trans[i])
        interpolated_poses[i][3,3] = 1.0
    
    # Rendering and saving video
    torch.cuda.empty_cache()
    video_imgs = []
    for i in range(num_frames):
        # [T, H, W, C]
        video_img = torch.clamp(render(viewpoint_stack[0], scene.gaussians, pipe, background, camera_pose = interpolated_poses[i], render_only=False, pose_rep="matrix")["render"], 0.0, 1.0).detach().cpu().permute(1,2,0)
        # if width or height is odd, resize to even
        if video_img.shape[0] % 2 != 0:
            video_img = video_img[:-1]
        if video_img.shape[1] % 2 != 0:
            video_img = video_img[:,:-1]
        video_imgs.append(video_img)
        # video_imgs.append(torch.clamp(render(viewpoint_stack[0], scene.gaussians, pipe, background, camera_pose = interpolated_poses[i], render_only=False, pose_rep="matrix")["render"], 0.0, 1.0).detach().cpu().permute(1,2,0))
    
    imgs_imgs = torch.stack(video_imgs)
    video_imgs = imgs_imgs * 255.0
    # torchvision.utils.save_image(imgs_imgs[0].permute(2,0,1), f'{args.output_path}/{args.exp_name}/interpolate_start.jpg')
    # torchvision.utils.save_image(imgs_imgs[50].permute(2,0,1), f'{args.output_path}/{args.exp_name}/interpolate_middle1.jpg')
    # torchvision.utils.save_image(imgs_imgs[100].permute(2,0,1), f'{args.output_path}/{args.exp_name}/interpolate_middle2.jpg')
    # torchvision.utils.save_image(imgs_imgs[150].permute(2,0,1), f'{args.output_path}/{args.exp_name}/interpolate_middle3.jpg')
    # torchvision.utils.save_image(imgs_imgs[-1].permute(2,0,1), f'{args.output_path}/{args.exp_name}/interpolate_end.jpg')
    torchvision.io.write_video(f'{args.output_path}/{args.exp_name}/video.mp4', video_imgs, int(30.0 / (len(key_times)/12)))

def prepare_output_and_logger(args, output_path, exp_name, project_name):
    if (not args.model_path) and (not exp_name):
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    elif (not args.model_path) and exp_name:
        args.model_path = os.path.join("./output", exp_name) 
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, 'command_line.txt'), 'w') as file:
        # Write the log information to the file.
        file.write(' '.join(sys.argv))

    # Create Tensorboard writer
    tb_writer = None
    # Prepare Wandb
    wandb.init(project=project_name, name=exp_name, dir=args.model_path, config=args, sync_tensorboard=True)
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("Logging progress to Tensorboard at {}".format(args.model_path))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def prealign_cameras(pose,pose_GT):
        # compute 3D similarity transform via Procrustes analysis
        center = torch.zeros(1,1,3)
        center_pred = camera_barf.cam2world(center,pose)[:,0] # [N,3]
        center_GT = camera_barf.cam2world(center,pose_GT)[:,0] # [N,3]
        try:
            sim3 = camera_barf.procrustes_analysis(center_GT,center_pred)
        except:
            print("warning: SVD did not converge...")
            sim3 = edict(t0=0,t1=0,s0=1,s1=1,R=torch.eye(3))
        # align the camera poses
        center_aligned = (center_pred-sim3.t1)/sim3.s1@sim3.R.t()*sim3.s0+sim3.t0
        R_aligned = pose[...,:3]@sim3.R.t()
        t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
        pose_aligned = camera_barf.pose(R=R_aligned,t=t_aligned)
        return pose_aligned,sim3

def evaluate_camera_alignment(pose_aligned,pose_GT):
    # measure errors in rotation and translation
    R_aligned,t_aligned = pose_aligned.split([3,1],dim=-1)
    R_GT,t_GT = pose_GT.split([3,1],dim=-1)
    R_error = camera_barf.rotation_distance(R_aligned,R_GT)
    t_error = (t_aligned-t_GT)[...,0].norm(dim=-1)
    error = edict(R=R_error,t=t_error)
    return error

def pose_error_report(tb_writer, iteration, scene,log_dir,all_R,all_t):
    with open(f'{log_dir}/cam_gt.pkl', 'rb') as f:
        gt_camera_poses_pkl = pickle.load(f)
        

    gt_camera_pose_R = np.array([gt_camera_pose[0].T for gt_camera_pose in gt_camera_poses_pkl])
    gt_camera_pose_T = np.array([gt_camera_pose[1] for gt_camera_pose in gt_camera_poses_pkl])
    gt_camera_pose_R = torch.tensor(gt_camera_pose_R, dtype=torch.float32)
    gt_camera_pose_T = torch.tensor(gt_camera_pose_T, dtype=torch.float32)

    pred_camera_pose_R = all_R.clone().detach().cpu()
    pred_camera_pose_T = all_t.clone().detach().cpu()
    
    
    
    gt_camera_poses = camera_barf.pose(gt_camera_pose_R,gt_camera_pose_T)
    pred_camera_poses = camera_barf.pose(pred_camera_pose_R, pred_camera_pose_T)

    pred_camera_poses_aligned, sim3 = prealign_cameras(pred_camera_poses, gt_camera_poses)
    error = evaluate_camera_alignment(pred_camera_poses_aligned, gt_camera_poses)

    print(f'rotation error(degrees) : {np.rad2deg(error.R.mean())} , translation error(m) : {error.t.mean()}')
    
    return np.rad2deg(error.R.mean()), error.t.mean(), sim3

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,log_dir,pose_noise, rotation_error, trans_error, pose_network=None, sim3=None, args_dict= None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('camera_pose_error/rotation_error', rotation_error, iteration)
        tb_writer.add_scalar('camera_pose_error/translation_error', trans_error, iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                {'name': 'train', 'cameras' : scene.getTrainCameras()})

        
                            #   {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            # if config['cameras'] and len(config['cameras']) > 0:
            if len(config['cameras'])>0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if pose_network is not None and config['name'] == 'train':
                        trained_pose = pose_network(idx)
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, camera_pose = trained_pose,render_only=False, pose_rep=args_dict['pose_representation'], *renderArgs)["render"], 0.0, 1.0)
                    elif pose_network is not None and config['name'] == 'test':
                        test_pose = viewpoint.world_view_transform
                        gt_image = viewpoint.original_image.to("cuda")
                        refine_test_pose = refine_pose(test_pose, sim3, scene,gt_image,idx, renderArgs)
                        image = torch.clamp(renderFunc( scene.getTrainCameras()[0], scene.gaussians, camera_pose = refine_test_pose,render_only=False, pose_rep=args_dict['pose_representation'], *renderArgs)["render"], 0.0, 1.0)
                    else:
                        image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    # save image
                    # cv2.imwrite(f'{args.output_path}/{args.exp_name}/{viewpoint.image_name}.png', (image.cpu().numpy()*255).astype(np.uint8).transpose(1,2,0))
                    # cv2.imwrite(f'{args.output_path}/{args.exp_name}/{viewpoint.image_name}_gt.png', (gt_image.cpu().numpy()*255).astype(np.uint8).transpose(1,2,0))
                    
                    torchvision.utils.save_image(image, f'{args.output_path}/{args.exp_name}/{viewpoint.image_name}.png')
                    torchvision.utils.save_image(gt_image, f'{args.output_path}/{args.exp_name}/{viewpoint.image_name}_gt.png')
                    ## error map
                    error_map = torch.abs(image - gt_image)
                    torchvision.utils.save_image(error_map, f'{args.output_path}/{args.exp_name}/{viewpoint.image_name}_error_map.png')

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                with open(os.path.join(args.output_path, args.exp_name, 'log_file.txt'), 'a') as file:
                    # Write the log information to the file.
                    file.write("\n[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
class TestCameraPose(nn.Module):
    def __init__(self, test_pose):
        super(TestCameraPose, self).__init__()
        self.test_pose = test_pose
        eye_pose = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1., 0.]).to(test_pose.device) + torch.randn(9).to(test_pose.device)*1e-3
        self.resi_pose = nn.Parameter(eye_pose, requires_grad=True)
        
    def forward(self):
        resi_pose_optim = r6d2mat(self.resi_pose.unsqueeze(dim=0)).squeeze(dim=0).transpose(1,0)
    
        test_pose_tmp = resi_pose_optim @ self.test_pose
        test_pose_tmp = test_pose_tmp.transpose(1,0)
        return test_pose_tmp
        
def refine_pose(test_pose, sim3, scene, gt_image, idx,renderArgs):
    test_pose = test_pose.clone().transpose(1,0)
    sim3 = edict({k:v.cuda() for k,v in sim3.items()})
    
    test_pose_barf = camera_barf.pose(test_pose[:3,:3],test_pose[:3,3])
    center = camera_barf.cam2world(torch.zeros(1,1,3, device=test_pose.device),test_pose_barf)[:,0]
    center_aligned = (center-sim3.t0)/sim3.s0@sim3.R*sim3.s1+sim3.t1
    R_aligned = test_pose_barf[...,:3]@sim3.R
    t_aligned = (-R_aligned@center_aligned[...,None])[...,0]
    
    
    test_pose = torch.eye(4,4,device=test_pose.device)
    test_pose[:3,:3] = R_aligned
    test_pose[:3,3] = t_aligned
    
    test_camera_posenet = TestCameraPose(test_pose)
    test_pose_optimizer = torch.optim.AdamW(test_camera_posenet.parameters(), lr=1e-4, betas=(0.9, 0.999))
    camera_pose_scheduler = ExponentialDecayScheduler(0, 1000, 5e-7).get_scheduler(test_pose_optimizer, 1e-4)
    ## refine test_pose with render
    
    iterator = tqdm(range(1000))
    
    for it in iterator:
        test_pose_optimizer.zero_grad()
        
        test_pose_tmp = test_camera_posenet()
        test_image = render(scene.getTrainCameras()[0], scene.gaussians, camera_pose = test_pose_tmp,render_only=False, pose_rep='9D', *renderArgs)["render"]
        Ll1 = l1_loss(test_image, gt_image)
        loss = (1.0 -0.2) * Ll1 + 0.2 * (1.0 - ssim(test_image, gt_image))
        if it%10==0:
            iterator.set_postfix({"Loss": f"{loss.item():.{7}f}"})
        loss.backward()
        test_pose_optimizer.step()
        camera_pose_scheduler.step()
        
    test_pose = test_camera_posenet().clone().detach()
    return test_pose

def pose_to_d9(pose: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrix to 9D representation. 

    We take the two first ROWS of the rotation matrix, 
    along with the translation vector.
    ATTENTION: row or vector needs to be consistent from pose_to_d9 and r6d2mat
    """
    nbatch = pose.shape[0]
    R = pose[:, :3, :3]  # [N, 3, 3]
    t = pose[:, :3, -1]  # [N, 3]

    r6 = R[:, :2, :3].reshape(nbatch, -1)  # [N, 6]

    d9 = torch.cat((t, r6), -1)  # [N, 9]
    # first is the translation vector, then two first ROWS of rotation matrix

    return d9

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--output_path", type=str,default='./output/')
    parser.add_argument("--random_pointcloud", action="store_true")
    parser.add_argument("--white_bg", action="store_true")
    parser.add_argument("--num_views", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="gaussian-splatting")
    parser.add_argument("--rgb_fix", action="store_true")
    parser.add_argument("--xyz_fix", action="store_true")
    parser.add_argument("--rgb_free", action="store_true")
    parser.add_argument("--zero_init", action="store_true")
    parser.add_argument("--no_densify", action="store_true")
    parser.add_argument("--pose_noise", action="store_true")
    parser.add_argument("--opacity_variation_fix", action="store_true")
    parser.add_argument("--pose_representation", type=str, default='9D', choices=['9D','quaternion'])
    parser.add_argument("--pretrained_scene", type=str, default=None)
    parser.add_argument('--DSV', action='store_true', help="Use the initialisation from the paper")
    parser.add_argument("--ours", action="store_true", help="Use our initialisation")
    parser.add_argument("--c2f", action="store_true", default=False)
    parser.add_argument("--c2f_every_step", type=float, default=1000, help="Recompute low pass filter size for every c2f_every_step iterations")
    parser.add_argument("--c2f_max_lowpass", type=float, default= 300, help="Maximum low pass filter size")
    parser.add_argument("--num_gaussians", type=int, default=1000000, help="Number of random initial gaussians to start with (default=1M for DSV)")
    parser.add_argument("--few_shot", type=int, default=-1, help="Percent of the dataset to use for training")
    parser.add_argument("--og_scale", action="store_true", help="True if original image is used for training instead of dust3r input images scale", default=True)
    parser.add_argument("--own_data", action="store_true", help="True if using own data")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.white_background = args.white_bg
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    if args.pretrained_scene:
        args.rgb_fix=True
        args.xyz_fix=True
        args.opacity_variation_fix=True
        args.no_densify=True

    if args.ours:
        print("========= USING OUR INITIALISATION =========")
        args.c2f = True
        args.c2f_every_step = 1000
        args.c2f_max_lowpass = 300
        args.eval = True
        args.num_gaussians = 10
    
    if args.own_data is False:
        if '_dust3r' not in args.source_path:
            args.source_path = args.source_path + '_dust3r'
            print(f"Changed source path to {args.source_path}")

    if not args.DSV and not args.ours:
        parser.error("Please specify either --DSV or --ours")
    print(f"args: {args}")
    
    while True :
        try:
            network_gui.init(args.ip, args.port)
            print(f"GUI server started at {args.ip}:{args.port}")
            break
        except Exception as e:
            args.port = args.port + 1
            print(f"Failed to start GUI server, retrying with port {args.port}...")
            
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations ,args.save_iterations, args.checkpoint_iterations ,args.start_checkpoint, args.debug_from, args.__dict__)

    # All done
    print("\nTraining complete.")
