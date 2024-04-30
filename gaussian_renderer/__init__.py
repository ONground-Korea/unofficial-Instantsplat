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

import torch
import math
import sys
sys.path.append('..')
from submodules.diff_gaussian_rasterization.diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F
import numpy as np

def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
            first two rows of the rotation matrix. 
    Returns:
        batch of rotation matrices of size (*, 3, 3)
    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    t = d6[..., :3]
    r = d6[..., 3:]
    
    a1, a2 = r[..., :3], r[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack((b1, b2, b3), dim=-2)
    mat = torch.cat((R,t[...,None]),-1)
    dummy_tensor = torch.from_numpy(np.array([0,0,0,1])).to(mat.dtype).to(mat.device).unsqueeze(0).unsqueeze(0)
    mat = torch.cat((mat, dummy_tensor), dim=1).squeeze().transpose(1,0)
    return mat  # corresponds to row

def getWorld2View2(R, t):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] =R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    return Rt

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, camera_pose = None, render_only = False, pose_rep =None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # Add camera parameters to optimizer
    if not render_only:
        pc.set_camera(viewpoint_camera)
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    if pose_rep == '9D':
        current_pose = camera_pose
        projmatrix = viewpoint_camera.projection_matrix.cuda()
        proj_full_matrix = (current_pose.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)
    elif pose_rep == 'quaternion':
        current_pose = getWorld2View2(camera_pose[0],camera_pose[1]).transpose(0,1).cuda()
        projmatrix = viewpoint_camera.projection_matrix.cuda()
        proj_full_matrix = (current_pose.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)
    elif pose_rep == 'matrix':
        current_pose = camera_pose.cuda().transpose(0,1)
        projmatrix = viewpoint_camera.projection_matrix.cuda()
        proj_full_matrix = (current_pose.unsqueeze(0).bmm(projmatrix.unsqueeze(0))).squeeze(0)
    else:
        current_pose = viewpoint_camera.world_view_transform
        proj_full_matrix = viewpoint_camera.full_proj_transform
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        # Modified viewmatrix, projmatrix
        viewmatrix=current_pose if not render_only else viewpoint_camera.world_view_transform,
        projmatrix=proj_full_matrix if not render_only else viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        # GaussianModel에서 가져옴 -> Computational Graph에 Attach
        viewmatrix = current_pose if not render_only else viewpoint_camera.world_view_transform,
        projmatrix = proj_full_matrix if not render_only else viewpoint_camera.full_proj_transform,
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii
            }
