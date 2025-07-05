# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
from torch import nn
from pyro_slam.autograd.function import TrackingTensor, map_transform
from pyro_slam.utils.pysolvers import PCG
from pyro_slam.utils.ba import rotate_quat

from vggt.dependency.projection import project_3D_points_np, project_3D_points

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

import pypose as pp

# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def prepare_pyro(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    if max_reproj_error is not None:
        if type(points3d) is np.ndarray:
            projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
            projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        elif type(points3d) is torch.Tensor:
            projected_points_2d, projected_points_cam = project_3D_points(points3d, extrinsics, intrinsics)
            projected_diff = torch.linalg.norm(projected_points_2d - tracks, dim=-1)
        else:
            raise TypeError("points3d must be either a numpy array or a torch tensor")
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error
    if masks is not None and reproj_mask is not None:
        masks = masks & reproj_mask
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    frames_insufficient = masks.sum(1) < min_inlier_per_frame
    if frames_insufficient.sum() > 0:
        print(f"{frames_insufficient.sum()} / {masks.shape[0]} frames not enough inliers per frame, skip BA.")
    masks[masks.sum(1) < min_inlier_per_frame] = False

    # filter out points with too few observations
    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    masks[:, ~valid_mask] = False

    vggt_reproj_error = (projected_points_2d - tracks)[masks]
    observations = tracks[masks]
    observations = torch.tensor(observations, dtype=torch.float64, device='cuda')
    keyframe_indices, landmark_indices = masks.nonzero() if type(masks) is np.ndarray else masks.nonzero(as_tuple=True)
    assert len(keyframe_indices) == len(landmark_indices), "Keyframe and landmark indices must match in length"
    assert len(keyframe_indices) == len(observations), "Observations length must match keyframe and landmark indices length"
    keyframe_indices = torch.tensor(keyframe_indices, dtype=torch.int32, device='cuda')
    landmark_indices = torch.tensor(landmark_indices, dtype=torch.int32, device='cuda')
    unique_keyframe, camera_indices = torch.unique(keyframe_indices, sorted=True, return_inverse=True)
    unique_landmark, point_indices = torch.unique(landmark_indices, sorted=True, return_inverse=True)
    intrinsics_ = torch.tensor(intrinsics, dtype=torch.float64, device='cuda')
    f = intrinsics_.diagonal(dim1=-2, dim2=-1)[..., :-1]
    if camera_type == "SIMPLE_PINHOLE":
        f = f.mean(-1, keepdim=True)  # Use mean focal length for shared camera
    center = intrinsics_[..., :2, 2]
    cameras = pp.mat2SE3(torch.tensor(extrinsics, dtype=torch.float64, device='cuda'))
    if not shared_camera:
        cameras = torch.cat([cameras, f], dim=-1)
    cameras = cameras[unique_keyframe]
    points = torch.tensor(points3d, dtype=torch.float64, device='cuda')[unique_landmark]

    @map_transform
    def reproject_simple_pinhole(points, camera_params, center, shared_intr=None):
        
        points_proj = rotate_quat(points, pp.SE3(camera_params[..., :7]))
        points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)  # add dimension for broadcasting
        if shared_intr is not None:
            f = shared_intr
        else:
            f = camera_params[..., 7:]
        points_proj = points_proj * f + center
        return points_proj
    
    class ReprojNonBatched(nn.Module):
        def __init__(self, camera_params, points_3d, shared_intr=None):
            super().__init__()
            self.pose = nn.Parameter(TrackingTensor(camera_params))
            self.points_3d = nn.Parameter(TrackingTensor(points_3d))
            self.pose.trim_SE3_grad = True
            if shared_intr is not None:
                self.shared_intr = nn.Parameter(TrackingTensor(shared_intr))
                assert self.shared_intr.shape[0] == 1, "Shared intrinsics must be a single camera"
            else:
                self.shared_intr = None

        def forward(self, points_2d, camera_indices, point_indices, center):
            camera_params = self.pose
            points_3d = self.points_3d

            if self.shared_intr is not None:
                indices = torch.zeros_like(camera_indices)
                shared_intr = self.shared_intr[indices]
            else:
                shared_intr = None

            points_proj = reproject_simple_pinhole(points_3d[point_indices], camera_params[camera_indices], center[camera_indices], shared_intr)
            loss = points_proj - points_2d
            return loss
        
    from pyro_slam.optim import LM
    input = {'points_2d': observations,
             'camera_indices': camera_indices,
             'point_indices': point_indices,
             'center': center[unique_keyframe],}
    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    shared_intr = None
    if shared_camera:
        shared_intr = f.mean(0, keepdim=True)
    model = ReprojNonBatched(cameras, points, shared_intr).to('cuda')
    optimizer = LM(model, strategy=strategy, solver=PCG(), reject=10)
    scheduler = pp.optim.scheduler.StopOnPlateau(optimizer, steps=40, patience=3, decreasing=1e-3, verbose=False)
    scheduler.optimize(input=input)
    
    # Apply optimization results back to input tensors in place
    with torch.no_grad():
        # Extract optimized camera parameters
        optimized_cameras = model.pose.data  # Shape: [num_unique_keyframes, 8] (SE3 + focal)
        optimized_points = model.points_3d.data  # Shape: [num_unique_landmarks, 3]
        
        # Update extrinsics: extract SE3 part and convert back to 3x4 matrix
        optimized_se3 = optimized_cameras[:, :7]  # SE3 parameters
        optimized_extrinsics_tensor = pp.SE3(optimized_se3).matrix()  # Convert to 4x4 matrices
        optimized_extrinsics_3x4 = optimized_extrinsics_tensor[:, :3, :]  # Take 3x4 part
        
        # Map back to original extrinsics array using unique_keyframe indices
        if shared_camera:
            optimized_focal = model.shared_intr.data
        else:
            optimized_focal = optimized_cameras[:, 7:]  # Focal length parameters (fx, fy)
        if isinstance(extrinsics, np.ndarray):
            extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float64, device='cuda')
            extrinsics_tensor[unique_keyframe] = optimized_extrinsics_3x4
            extrinsics[:] = extrinsics_tensor.cpu().numpy()
            
            # Update intrinsics: extract focal lengths and update intrinsic matrices
            intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float64, device='cuda')
            # Update diagonal elements (fx, fy) with optimized focal lengths
            intrinsics_tensor[unique_keyframe, 0, 0] = optimized_focal[:, 0]  # fx
            intrinsics_tensor[unique_keyframe, 1, 1] = optimized_focal[:, -1]  # fy
            intrinsics[:] = intrinsics_tensor.cpu().numpy()
            
            # Update points3d: map back using unique_landmark indices
            points3d_tensor = torch.tensor(points3d, dtype=torch.float64, device='cuda')
            points3d_tensor[unique_landmark] = optimized_points
            points3d[:] = points3d_tensor.cpu().numpy()
        elif isinstance(extrinsics, torch.Tensor):
            extrinsics[unique_keyframe] = optimized_extrinsics_3x4.to(extrinsics.dtype)
            # Update intrinsics: extract focal lengths and update intrinsic matrices
            intrinsics[unique_keyframe, 0, 0] = optimized_focal[:, 0].to(intrinsics.dtype)  # fx
            intrinsics[unique_keyframe, 1, 1] = optimized_focal[:, -1].to(intrinsics.dtype)  # fy
            # Update points3d: map back using unique_landmark indices
            points3d[unique_landmark] = optimized_points.to(points3d.dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_ba", action="store_true", default=False, help="Use BA for reconstruction")
    ######### BA parameters #########
    parser.add_argument(
        "--max_reproj_error", type=float, default=8.0, help="Maximum reprojection error for reconstruction"
    )
    parser.add_argument(
        "--implementation", type=str, default="pycolmap", help="Implementation for reconstruction (pycolmap or pyro_slam)"
    )
    parser.add_argument("--shared_camera", action="store_true", default=False, help="Use shared camera for all images")
    parser.add_argument("--camera_type", type=str, default="SIMPLE_PINHOLE", help="Camera type for reconstruction")
    parser.add_argument("--vis_thresh", type=float, default=0.2, help="Visibility threshold for tracks")
    parser.add_argument("--query_frame_num", type=int, default=5, help="Number of frames to query")
    parser.add_argument("--max_query_pts", type=int, default=2048, help="Maximum number of query points")
    parser.add_argument(
        "--fine_tracking", action="store_true", default=True, help="Use fine tracking (slower but more accurate)"
    )
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def demo_fn(args):
    # Print configuration
    print("Arguments:", vars(args))
    assert args.implementation in ["pycolmap", "pyro_slam"], "Invalid implementation specified"

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
    print(f"Setting seed as: {args.seed}")

    # Set device and dtype
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Run VGGT for camera and depth estimation
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print(f"Model loaded")

    # Get image paths and preprocess them
    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    if args.use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with torch.cuda.amp.autocast(dtype=dtype):
            # Predicting Tracks
            # Using VGGSfM tracker instead of VGGT tracker for efficiency
            # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
            # Will be fixed in VGGT v2

            # You can also change the pred_tracks to tracks from any other methods
            # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
            pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                images,
                conf=depth_conf,
                points_3d=points_3d,
                masks=None,
                max_query_pts=args.max_query_pts,
                query_frame_num=args.query_frame_num,
                keypoint_extractor="aliked+sp",
                fine_tracking=args.fine_tracking,
            )

            torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        if args.implementation == "pyro_slam":
            # construct parameters
            prepare_pyro(
                points_3d,
                extrinsic,
                intrinsic,
                pred_tracks,
                image_size,
                masks=track_mask,
                max_reproj_error=args.max_reproj_error,
                shared_camera=shared_camera,
                camera_type=args.camera_type,
                points_rgb=points_rgb,
            )
        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        # Bundle Adjustment
        if not args.implementation == "pyro_slam":
            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = args.conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    print(f"Saving reconstruction to {args.scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(args.scene_dir, "sparse/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
