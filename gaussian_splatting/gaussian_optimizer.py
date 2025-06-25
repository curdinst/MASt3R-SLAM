import random
import time

import torch
from torch import nn
import torch.multiprocessing as mp
import numpy as np
import einops
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from gaussian_splatting.gaussian_renderer import render
# from utils.logging_utils import Log
# from utils.multiprocessing_utils import clone_obj
# from utils.pose_utils import update_pose
# from utils.slam_utils import get_loss_mapping
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from gaussian_splatting.utils.image_utils import mse, psnr
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.camera_utils import Camera 
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.geometry import constrain_points_to_ray, quat_mult
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.pose_utils import update_pose
from gaussian_splatting.utils.slam_utils import get_loss_tracking_rgb, get_loss_tracking_rgbd, get_loss_mapping_rgbd
from gaussian_splatting.utils.general_utils import slerp

from matplotlib import pyplot as plt
import pickle
from munch import munchify
import os
from datetime import datetime



class GaussianOptimizer:
    def __init__(self, config, dataset, device, learning_rate=0.01):
        """
        Initialize the GaussianOptimizer.

        Args:
            learning_rate (float): The learning rate for optimization.
        """
        self.learning_rate = learning_rate
        random.seed(config["gaussians"]["seed"])
        self.config = config
        self.gaussians = GaussianModel(sh_degree=0)
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = device
        self.dtype = torch.float32
        self.monocular = True
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoint_stack = {}
        self.current_window = []
        self.initialized = False
        self.keyframe_optimizers = None
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        self.tracking_itr_num = config["gaussians"]["tracking_itr_num"]
        
        if config["use_calib"]:
            K_frame = dataset.camera_intrinsics.K_frame
            fx = K_frame[0, 0]
            fy = K_frame[1, 1]
            cx = K_frame[0, 2]
            cy = K_frame[1, 2]
        elif "tum" in config["used_dataset"]:
            fx = config["gaussians"]["calib_tum"]["fx"]
            fy = config["gaussians"]["calib_tum"]["fy"]
            cx = config["gaussians"]["calib_tum"]["cx"]
            cy = config["gaussians"]["calib_tum"]["cy"]
        elif "replica" in config["used_dataset"]:
            fx = config["gaussians"]["calib_replica"]["fx"]
            fy = config["gaussians"]["calib_replica"]["fy"]
            cx = config["gaussians"]["calib_replica"]["cx"]
            cy = config["gaussians"]["calib_replica"]["cy"]
        H, W = dataset.get_img_shape()[0]
        # dataset_name = config["used_dataset"].split("/")[1]
        # W = config["gaussians"]["Calibration"][dataset_name]["width"]
        # H = config["gaussians"]["Calibration"][dataset_name]["height"]
        self.intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W":W, "H": H}
        print(f"fx {fx}, fy {fy}, cx {cx}, cy {cy}, W {W}, H {H}")
        self.projection_matrix = getProjectionMatrix2( znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1).to(device=device)
        self.fovx = focal2fov(fx, W)
        self.fovy = focal2fov(fy, H)
        self.pipeline_params = munchify(config["gaussians"]["pipeline_params"])
        opt_params = config["gaussians"]["map_optimisation_params"]
        self.opt_params = munchify(opt_params)
        self.init_lr = config["gaussians"]["init_lr"]
        self.gaussians.init_lr(self.init_lr)
        self.gaussians.training_setup(self.opt_params)
        self.valid_masks = {}
        self.window_size = config["gaussians"]["window_size"]
        self.c_conf_threshold = config["gaussians"]["c_conf_threshold"]
        self.keyframe_TFs = {}
        self.keyframes = []
        self.optimized_poses = {}
        self.N_optimized_kf_gaussians = {}
        self.averaged_masks = {}

    def optimize(self, keyframes: SharedKeyframes, iters, save_results=False, path=None):
        print(f"\033[92mrun Gaussian Optimizer, number of keyframes: {len(keyframes)}\033[0m")
        # if len(keyframes) > 2: return
        # del self.viewpoint_stack
        self.viewpoint_stack = {}
        self.keyframe_TFs = {}
        # del self.gaussians
        self.gaussians = GaussianModel(sh_degree=0)
        #TODO Keep gaussians, only update X_canon
        # self.valid_masks = {}
        num_keyframes = len(keyframes)
        for frame_idx in range(num_keyframes):
            keyframe = keyframes[frame_idx]
            idx = 0
            self.averaged_masks[frame_idx] = torch.ones((self.intrinsics["H"]*self.intrinsics["W"]), dtype=torch.bool, device=self.device)
            # print(f"self.averaged_masks[frame_idx] {self.averaged_masks[frame_idx].shape}")
            for idx, other_frame in enumerate(keyframe.corresponding_frames.tolist()):
                # print(f"other frame {other_frame} for keyframe {keyframe.frame_id}")
                if other_frame == -1: break
                valid_mask = ~keyframe.valid_match_i[idx,...]
                self.averaged_masks[other_frame] = self.averaged_masks[other_frame] & valid_mask
                # TODO: simplify above

        for frame_idx in range(num_keyframes):
            keyframe = keyframes[frame_idx]

            viewpoint = Camera(
                keyframe.frame_id,
                None,
                None,
                None,
                self.projection_matrix,
                self.intrinsics["fx"],
                self.intrinsics["fy"],
                self.intrinsics["cx"],
                self.intrinsics["cy"],
                self.fovx,
                self.fovy,
                self.intrinsics["H"],
                self.intrinsics["W"],
                device=self.device,
            )
            # print(f"keyframe {keyframe.frame_id} T_WC {keyframe.T_WC.data}")
            rot = R.from_quat(keyframe.T_WC.data[0,3:7].cpu()).as_matrix()
            viewpoint.R  = torch.from_numpy(rot).to(device=self.device).T
            viewpoint.T = -viewpoint.R.float() @ keyframe.T_WC.data[0,:3].float()
            # viewpoint.R = torch.from_numpy(rot).to(device=self.device)
            # viewpoint.T = keyframe.T_WC.data[0,:3].float()
            c_conf_threshold = self.c_conf_threshold
            valid = (
                        keyframe.get_average_conf().reshape(-1)
                        > c_conf_threshold
                    )
            self.valid_masks[frame_idx] = valid
            # print(f"frame {frame_idx} gaussians valid mask sum {keyframe.valid_match_i.sum()}")
            # if self.config["gaussians"]["use_matching_mask"] and keyframe.gaussian_mask.sum() > 0:
            #     valid_matching_mask = valid & keyframe.gaussian_mask
            # self.valid_masks[frame_idx] = valid
            # valid = torch.zeros_like(valid, dtype=torch.bool)
            # valid[int(288*512//2):] = True
            # depth = einops.rearrange(keyframe.X_canon[:, -1], "(h w) -> h w", h=self.intrinsics["H"], w=self.intrinsics["W"])
            # valid_image = einops.rearrange(valid, "(h w) -> h w", h=self.intrinsics["H"], w=self.intrinsics["W"])
            # depth[~valid_image] = 0.0
            # print(f"sum valid {(~valid_image).sum()}")
            # print(f"depth shape {depth.shape}, min {depth.min()}, max {depth.max()}")
            # viewpoint.depth = depth
            viewpoint.original_image = keyframe.img.clone().to(device=self.device)/2.0+0.5
            if num_keyframes > 1 and frame_idx == num_keyframes - 1 and self.config["gaussians"]["pre_pose_optimization"]:
                self.tracking(num_keyframes-1, viewpoint, tracking_itr_num=self.tracking_itr_num)
            self.viewpoint_stack[frame_idx] = viewpoint
            self.keyframe_TFs[frame_idx] = keyframe.T_WC
            # if i == len(keyframes):
            # print("add points to gaussians")

            # valid = np.ones_like(valid, dtype=bool)
            scales_new = (keyframe.T_WC.data[0,-1] * keyframe.scales)
            opacities_new = keyframe.opacities
            w_rotations = quat_mult(keyframe.T_WC.data, keyframe.rotations)
            if self.config["use_calib"]:
                X_ray = constrain_points_to_ray(keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K)
                w_means = keyframe.T_WC.act(X_ray[0,...] + keyframe.offsets)
            else:
                w_means = keyframe.T_WC.act(keyframe.X_canon + keyframe.offsets)
            sh = keyframe.SH
            if frame_idx > 0 and self.config["gaussians"]["average_correspondances"]:
                # correspondance_mask = matching_gaussians[-1]
                averaged_mask = self.averaged_masks[frame_idx]
                # print(f"other frames: {keyframe.corresponding_frames.tolist()}")
                for idx, other_frame in enumerate(keyframe.corresponding_frames.tolist()):
                    if other_frame == -1: break
                    # if other_frame != frame_idx-1: continue #TODO remove---------------------------------------
                    idx_j2i = keyframe.idx_j2i[idx]
                    valid_correspondance = keyframe.valid_match_i[idx]
                    old_keyframe = keyframes[other_frame]
                    # print(f"valid_masks[frame_idx] {self.valid_masks[frame_idx].shape}")
                    # print(f"valid_masks[frame_idx] {self.valid_masks[frame_idx]}")
                    # print(f"keyframe.valid_match_i[frame_idx] {keyframe.valid_match_i.shape}")
                    # print(f"keyframe.valid_match_i[frame_idx] {keyframe.valid_match_i[idx]}")
                    # print(f"idx {idx}")
                    to_average = valid & keyframe.valid_match_i[idx,...]
                    # print(f"to_avg {to_average} to_avg.sum {to_average.sum()}")
                    old_kf_scales = (old_keyframe.T_WC.data[0,-1] * old_keyframe.scales)
                    old_kf_opacities_new = old_keyframe.opacities
                    old_kf_w_rotations = quat_mult(old_keyframe.T_WC.data, old_keyframe.rotations)
                    if self.config["use_calib"]:
                        X_ray = constrain_points_to_ray(old_keyframe.img_shape.flatten()[:2], old_keyframe.X_canon[None], old_keyframe.K)
                        old_kf_w_means = old_keyframe.T_WC.act(X_ray[0, ...] + old_keyframe.offsets)
                    else:
                        old_kf_w_means = old_keyframe.T_WC.act(old_keyframe.X_canon + old_keyframe.offsets)
                    old_kf_sh = old_keyframe.SH
                    idx_j2i = idx_j2i[to_average]
                    # a = torch.zeros((self.intrinsics["H"] * self.intrinsics["W"]), dtype=torch.bool, device=self.device)
                    # a[idx_j2i] = True
                    # idx_j2i = a
                    # print(f"idx_j2i shape {idx_j2i.shape},\n to_average shape {to_average.shape}")
                    # print("old means shape: ", old_kf_w_means.shape)
                    # idx_j2i = to_average[idx_j2i]
                    old_kf_mask = to_average
                    gaussians_old_kf = (
                        old_kf_w_means[old_kf_mask],
                        old_kf_sh[old_kf_mask],
                        old_kf_opacities_new[old_kf_mask],
                        old_kf_scales[old_kf_mask],
                        old_kf_w_rotations[old_kf_mask]
                    )
                    mask_now = idx_j2i
                    gaussians_now = (
                        w_means[mask_now],
                        sh[mask_now],
                        opacities_new[mask_now],
                        scales_new[mask_now],
                        w_rotations[mask_now]
                    )
                    gaussians_avg = self.mean_gaussians(gaussians_now, gaussians_old_kf)
                    (
                        w_means[mask_now],
                        sh[mask_now],
                        opacities_new[mask_now],
                        scales_new[mask_now],
                        w_rotations[mask_now]
                    ) = gaussians_avg

            l1_mask = torch.ones_like(valid, dtype=torch.bool, device=self.device)
            if self.config["gaussians"]["l1_mask"] and frame_idx > 0:
                print(f"get l1 mask for frame {frame_idx}")
                render_pkg = render(self.viewpoint_stack[frame_idx], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                # Save the rendered image for debugging or visualization
                # ssim_loss_val = ssim(image, self.viewpoint_stack[frame_idx].original_image)
                # l1_loss_val = l1_loss(image, self.viewpoint_stack[frame_idx].original_image)
                l1_threshold = self.config["gaussians"]["l1_threshold"]
                l1_loss_img = torch.abs(image - self.viewpoint_stack[frame_idx].original_image).mean(dim=0).reshape(-1)
                l1_mask = (l1_loss_img > l1_threshold)
                # Save l1_mask as an image for debugging or visualization

                # self.valid_masks[frame_idx] = l1_mask
                # print(f"l1_loss_mask shape reshaped {l1_loss_mask.shape}")
                # print(f"l1_mask shape {l1_mask.shape}, l1_mask sum {l1_mask.sum()}")
            # elif (frame_idx == 0 and frame_idx not in self.valid_masks.keys()) or not self.config["gaussians"]["l1_mask"]:
            #     self.valid_masks[frame_idx] = valid

            # print(f"valid mask keys {self.valid_masks.keys()}")
            valid = valid & self.averaged_masks[frame_idx] & l1_mask
            self.valid_masks[frame_idx] = valid
            print(f"adding {valid.sum():,} points to gaussians")
            self.gaussians.add_points(
                new_xyz=w_means[valid],
                new_features_dc=sh[valid],
                new_opacities=opacities_new[valid],
                new_scales=scales_new[valid],
                new_rotations=w_rotations[valid]
            )
            # self.keyframes.append(keyframe.frame_id)
            # self.gaussians.load_ply("/home/curdinst/repos/MASt3R-SLAM/logs/rgbd_dataset_freiburg1_desk_2025-04-17_09-48-43_wa.ply")
            # print(f"num_gaussians: {self.gaussians._xyz.shape}")
            # break
        if self.config["gaussians"]["voxel_reduction"]:
            gaussians = self.reduce_gaussians_with_voxels(
                positions=self.gaussians._xyz,
                scales=self.gaussians._scaling,
                rotations=self.gaussians._rotation,
                sh_coeffs=self.gaussians._features_dc,
                opacities=self.gaussians._opacity,
                voxel_size=self.config["gaussians"]["voxel_size"]
            )
            self.gaussians._xyz, self.gaussians._scaling, self.gaussians._rotation, self.gaussians._features_dc, self.gaussians._opacity = gaussians
            self.gaussians._features_rest = nn.Parameter(torch.zeros((self.gaussians._features_dc.shape[0],0,3), dtype=self.dtype, device=self.device))

        # render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        # image = render_pkg["render"]
        # print(f"image render shape {image.shape}")
        # for frame_idx in range(num_keyframes):
        #     self.tracking(frame_idx, self.viewpoint_stack[frame_idx], tracking_itr_num=self.tracking_itr_num)
        print(f"num gaussians: {self.gaussians._xyz.shape[0]:,}")
        self.gaussians.init_lr(self.init_lr)
        self.gaussians.training_setup(self.opt_params)
        #         break
        add_random_frames = False
        if self.window_size == 1 and not save_results:
            optimisation_window = [num_keyframes - 1]
        elif num_keyframes > 2 and not save_results:
            add_random_frames = True   
        else:
            optimisation_window = list(range(num_keyframes))
        # optimisation_window = list(range(num_keyframes))
        self.rendering_vals = {}
        
        for i in range(iters):
            self.iteration_count += 1
            loss_mapping = 0
            time_now = time.time()
            if add_random_frames:
                optimisation_window = [num_keyframes - 2, num_keyframes - 1]
                rest_view_idxs = list(range(num_keyframes - 2))
                random.shuffle(rest_view_idxs)
                optimisation_window += rest_view_idxs[:self.window_size-2]
            if not save_results:
                for frame_idx in optimisation_window:
                    if frame_idx in self.N_optimized_kf_gaussians.keys() and iters > 0:
                        self.N_optimized_kf_gaussians[frame_idx] += 1
                    elif iters == 0:
                        self.N_optimized_kf_gaussians[frame_idx] = 0
                    else:
                        self.N_optimized_kf_gaussians[frame_idx] = 1
            # for frame_index in range(num_keyframes):
            for frame_index in optimisation_window:
            # for frame_index in range(2):
                render_pkg = render(self.viewpoint_stack[frame_index], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                # n_touched_acm.append(render_pkg["n_touched"])
                # image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                # print(f"exposure_a {viewpoint.exposure_a}, exposure_b {viewpoint.exposure_b}")
                # ssim_loss_val = torch.tensor(-1)
                l1_loss_val = l1_loss(image, self.viewpoint_stack[frame_index].original_image)
                # l1_loss_val = get_loss_mapping_rgbd(self.config, image, render_pkg["depth"], self.viewpoint_stack[frame_index], initialization=False)
                # l1_loss_val = get_loss_mapping_rgbd(self.config, image, render_pkg["depth"], self.viewpoint_stack[frame_index], initialization=False)
                # loss_mapping = l1_loss_val * 0.75 + 0.25 * (1-ssim_loss_val)
                loss_mapping += l1_loss_val
                if i == 0 or i == iters - 1:
                    ssim_loss_val = ssim(image, self.viewpoint_stack[frame_index].original_image)
                    psnr_val = psnr(image.unsqueeze(0), self.viewpoint_stack[frame_index].original_image.unsqueeze(0))
                    print(f"frame_index {frame_index} iteration {i} psnr: {psnr_val.item()} SSIM {round(ssim_loss_val.item(), 8)} L1 {round(l1_loss_val.item(), 8)}")
                # l1_loss_mask = torch.abs(image - self.viewpoint_stack[frame_index].original_image).mean(dim=0)
                # print(f"l1_loss_mask shape {l1_loss_mask.shape}")

                # print("image", image.shape)
                # print(f"render results: SSIM {round(ssim_loss_val.item(), 3)} L1 {round(l1_loss_val.item(), 3)}")

                save_plot = save_results
                if save_results:
                    self.rendering_vals[f"ssim_{frame_index}"] = ssim_loss_val.item()
                    self.rendering_vals[f"l1_{frame_index}"] = l1_loss_val.item()
                    self.rendering_vals[f"psnr_{frame_index}"] = psnr_val.item()
                if save_plot:
                    image_rearranged = einops.rearrange(image.cpu().detach().numpy(), "c h w -> h w c")
                    plt.figure()
                    plt.title(f"frame_index {frame_index} iteration {i} PSNR {round(psnr_val.item(), 3)}")
                    plt.axis("off")
                    # plt.subplot(1, 2, 1)
                    a,b = np.min(image_rearranged), np.max(image_rearranged)
                    plt.imshow((image_rearranged - a)/(b-a))
                    # plt.subplot(1, 2, 2)
                    # gt_img_rearranged = einops.rearrange(self.viewpoint_stack[frame_index].original_image.cpu().detach().numpy(), "c h w -> h w c")
                    # a,b = np.min(gt_img_rearranged), np.max(gt_img_rearranged)
                    # plt.imshow((gt_img_rearranged- a)/(b-a) )
                    plt.savefig(path / f"render_{frame_index}.png")
                    plt.close()
                
            if not save_results:
                loss_mapping.backward()
                with torch.no_grad():
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    self.gaussians.update_learning_rate(i)
                    loss_mapping = 0
            iteration_time = time.time() - time_now
            if i%10 == 0:
                print(f"iteration {i} took {iteration_time:.4f} seconds")
        if save_results:
            self.draw_cameras()
        # print(f"num gaussians 3333: {self.gaussians._xyz.shape[0]}")

        # # Overwrite
        # idx = 0
        # for frame_idx in range(num_keyframes):
        #     valid = self.valid_masks[frame_idx]
        #     num_valid = valid.sum()
        #     print(f"num_valid {num_valid} for frame {frame_idx}")
        #     # print(f"num gaussians {self.gaussians._xyz.shape}, num_valid {num_valid}")
        #     # print(f"num_valid: {num_valid} a {self.gaussians._xyz[idx:idx+num_valid].shape}, b {keyframes[frame_idx].X_canon[valid].shape}")
        #     # idx_0, idx_1 = gauss_indices[frame_idx], gauss_indices[frame_idx + 1]
            
        #     keyframe = keyframes[frame_idx]
        #     if num_keyframes > 1 and frame_idx == num_keyframes -1 and self.config["gaussians"]["pre_pose_optimization"]:
        #         R_CW = self.viewpoint_stack[frame_idx].R
        #         T_CW = self.viewpoint_stack[frame_idx].T
        #         T_WC = - R_CW.T @ T_CW
        #         quat_WC = torch.from_numpy(R.from_matrix(R_CW.T.cpu().numpy()).as_quat()).to(device=self.device)
        #         print(f"old quat {keyframe.T_WC.data[0,3:7]}, new quat {quat_WC}")
        #         print(f"old T {keyframe.T_WC.data[0,:3]}, new T {self.viewpoint_stack[frame_idx].T}")
        #         keyframe.T_WC.data[0,3:7] = quat_WC
        #         keyframe.T_WC.data[0,:3] = T_WC
        #     T_CW = keyframe.T_WC.inv()
        #     # print(f"T_WC {keyframe.T_WC.data}")
        #     # print(f"T_CW {T_CW.data}")
        #     scales_w = torch.exp(self.gaussians._scaling[idx:idx+num_valid]) * T_CW.data[0,-1]
        #     rotations_w = self.gaussians._rotation[idx:idx+num_valid]
        #     rotations_kf = quat_mult(T_CW.data, rotations_w)
        #     means_w = T_CW.act(self.gaussians._xyz[idx:idx+num_valid])

        #     # print("scaling:", self.gaussians._scaling[idx:idx+num_valid])
        #     # print("scales kf :", keyframes[frame_idx].scales)
        #     keyframe.update_gaussians(
        #         valid_mask=valid,
        #         scale=scales_w,
        #         rotation=rotations_kf,
        #         SH=einops.rearrange(self.gaussians._features_dc[idx:idx+num_valid], "wh d c -> wh c d"),
        #         opacity=self.gaussians._opacity[idx:idx+num_valid],
        #         mean=means_w,
        #     )
        #     keyframes[frame_idx] = keyframe
        #     idx += num_valid
        print(f"num gaussians: {self.gaussians._xyz.shape[0]:,}"+f", idx: {idx}")
        print(f"updated gaussians of {num_keyframes} keyframes")
        if save_results:
            for kf_idx in range(num_keyframes):
                if self.config["gaussians"]["post_pose_optimization"]:
                    self.tracking(kf_idx, self.viewpoint_stack[kf_idx], tracking_itr_num=self.tracking_itr_num)
                T_CW_opt = torch.eye(4, device=self.device)
                T_CW_opt[:3, :3] = self.viewpoint_stack[kf_idx].R
                T_CW_opt[:3, 3] = self.viewpoint_stack[kf_idx].T
                self.optimized_poses[kf_idx] = T_CW_opt.clone()
                    
        # if num_keyframes == 11:
        #     self.gaussians.save_ply(f"/home/curdinst/repos/MASt3R-SLAM/logs/online_opt_{iters}_it.ply")

        #         del render_pkg
        #         break
        #     break
                # self._save_checkpoint()
        return
    
    def tracking(self, cur_frame_idx, viewpoint, tracking_itr_num=100):
        # print(f"Initial pose for frame {cur_frame_idx}: T: {viewpoint.T}, R: {viewpoint.R}")
        T0 = viewpoint.T
        viewpoint.compute_grad_mask(self.config)
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["gaussians"]["tracking_lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["gaussians"]["tracking_lr"]["cam_trans_delta"],
                "name": "trans_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_a],
                "lr": 0.01,
                "name": "exposure_a_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.exposure_b],
                "lr": 0.01,
                "name": "exposure_b_{}".format(viewpoint.uid),
            }
        )

        pose_optimizer = torch.optim.Adam(opt_params)
        for tracking_itr in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking_rgbd(
                self.config, image, depth, opacity, viewpoint
            )
            if tracking_itr == 0 or tracking_itr == tracking_itr_num-1: print(f"tracking iteration {tracking_itr} loss_tracking {round(loss_tracking.item(), 5)}")
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                print(f"tracking converged at iteration {tracking_itr} with loss_tracking {round(loss_tracking.item(), 5)}")
                break
        # print(f"optimized pose for frame {cur_frame_idx}: T: {viewpoint.T}, R: {viewpoint.R}")
        print(f"Pose update of frame {cur_frame_idx}: T: {viewpoint.R.T @ (T0 - viewpoint.T)}")
        # self.median_depth = get_median_depth(depth, opacity)
        return render_pkg


    
    def reduce_gaussians_with_voxels(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        sh_coeffs: torch.Tensor,
        opacities: torch.Tensor,
        voxel_size: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reduces the number of 3D Gaussians using a voxel grid.

        Args:
            positions: Tensor of shape (N, 3) representing Gaussian positions.
            scales: Tensor of shape (N, 3) representing Gaussian scales.
            rotations: Tensor of shape (N, 4) representing Gaussian rotations as quaternions.
            sh_coeffs: Tensor of shape (N, C, D) representing Spherical Harmonics coefficients.
            opacities: Tensor of shape (N, 1) representing Gaussian opacities.
            voxel_size: The size of each voxel.

        Returns:
            A tuple containing the reduced positions, scales, rotations, sh_coeffs, and opacities.
        """
        # Get the device from the input tensors
        device = positions.device

        # Determine the scene bounds
        min_bound = torch.min(positions, dim=0)[0]
        max_bound = torch.max(positions, dim=0)[0]

        # Assign each Gaussian to a voxel
        voxel_indices = torch.floor((positions - min_bound) / voxel_size).long()

        # Create a unique integer ID for each voxel index for efficient grouping
        unique_voxel_ids, inverse_indices = torch.unique(voxel_indices, dim=0, return_inverse=True)
        print(f"positions shape: {positions.shape}, unique_voxel_ids shape: {unique_voxel_ids.shape}, inverse_indices shape: {inverse_indices.shape}")
        # Initialize tensors for the new, reduced Gaussians
        num_reduced_gaussians = unique_voxel_ids.shape[0]
        new_positions = torch.zeros((num_reduced_gaussians, 3), dtype=torch.float32, device=device)
        new_scales = torch.zeros((num_reduced_gaussians, 3), dtype=torch.float32, device=device)
        new_rotations = torch.zeros((num_reduced_gaussians, 4), dtype=torch.float32, device=device)
        new_sh_coeffs = torch.zeros((num_reduced_gaussians, sh_coeffs.shape[1], sh_coeffs.shape[2]), dtype=torch.float32, device=device)
        new_opacities = torch.zeros((num_reduced_gaussians, 1), dtype=torch.float32, device=device)

        # Use scatter_add_ to sum properties for each unique voxel
        new_positions.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), positions)
        new_scales.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), scales)
        new_rotations.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 4), rotations)
        new_sh_coeffs.scatter_add_(0, inverse_indices.unsqueeze(1).unsqueeze(2).expand(-1, sh_coeffs.shape[1], sh_coeffs.shape[2]), sh_coeffs)
        new_opacities.scatter_add_(0, inverse_indices.unsqueeze(1), opacities)

        # Count the number of Gaussians in each voxel
        voxel_counts = torch.zeros(num_reduced_gaussians, dtype=torch.long, device=device)
        voxel_counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.long))

        # Average the properties
        new_positions /= voxel_counts.unsqueeze(1)
        new_scales /= voxel_counts.unsqueeze(1)
        new_rotations /= voxel_counts.unsqueeze(1)
        # Normalize the averaged quaternions
        new_rotations = torch.nn.functional.normalize(new_rotations, p=2, dim=1)
        new_sh_coeffs /= voxel_counts.unsqueeze(1).unsqueeze(2)
        new_opacities /= voxel_counts.unsqueeze(1)


        return nn.Parameter(new_positions), nn.Parameter(new_scales), nn.Parameter(new_rotations), nn.Parameter(new_sh_coeffs), nn.Parameter(new_opacities)

    def mean_gaussians(self, gaussians_1, gaussians_2):
        (means_1, features_dc_1, opacities_1, scales_1, rotations_1) = gaussians_1
        (means_2, features_dc_2, opacities_2, scales_2, rotations_2) = gaussians_2
        dists = torch.sqrt(((means_1 - means_2)**2).sum(dim=1))
        mean_dists = torch.mean(dists)
        print(f"mean diffs: {mean_dists}, max: {torch.max(dists)}, min: {torch.min(dists)}")
        inlier_mask = dists < mean_dists
        print(f"inlier_mask shape {inlier_mask.shape}, inlier_mask sum {inlier_mask.sum()}")
        # inlier_mask = ~outliers_mask
        
        means_1[inlier_mask] = (means_1[inlier_mask] + means_2[inlier_mask]) / 2.0
        features_dc_1[inlier_mask] = (features_dc_1[inlier_mask] + features_dc_2[inlier_mask]) / 2.0
        opacities_1[inlier_mask] = (opacities_1[inlier_mask] + opacities_2[inlier_mask]) / 2.0
        # scales_1[inlier_mask] = torch.sqrt((scales_1[inlier_mask]**2 + scales_2[inlier_mask]**2))
        scales_1[inlier_mask] = (scales_1[inlier_mask] + scales_2[inlier_mask]) / 2.0
        rotations_1[inlier_mask] = slerp(rotations_1[inlier_mask], rotations_2[inlier_mask], 0.5)

        # take_1 = ~inlier_mask & valid_mask_1 
        # means_1[take_1] = means_1[take_1]
        # features_dc_1[take_1] = features_dc_1[take_1]
        # opacities_1[take_1] = opacities_1[take_1]
        # scales_1[take_1] = scales_1[take_1]
        # rotations_1[take_1] = rotations_1[take_1]

        # take_2 = ~inlier_mask & valid_mask_2 & ~valid_mask_1 # avoid double counting
        # means_1[take_2] = means_2[take_2]
        # features_dc_1[take_2] = features_dc_2[take_2]
        # opacities_1[take_2] = opacities_2[take_2]
        # scales_1[take_2] = scales_2[take_2]
        # rotations_1[take_2] = rotations_2[take_2]

        return (means_1, features_dc_1, opacities_1, scales_1, rotations_1)


    def draw_cameras(self):
        # Add red gaussians forming a camera frustum for each camera
        frustum_color = torch.tensor([50.0, 0.0, 0.0], device=self.device)  # Red color
        frustum_opacity = torch.tensor([1.0], device=self.device)  # Fully opaque
        frustum_scale = torch.tensor([0.002], device=self.device)  # Small scale for visualization

        # Define frustum vertices in camera space
        # tan_fovx = torch.tan(torch.deg2rad(torch.tensor(self.fovx / 2)))
        # tan_fovy = torch.tan(torch.deg2rad(torch.tensor(self.fovy / 2)))
        rotation = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # Identity rotation
        dist = 0.15
        left = dist * 0.5 # tan_fovx * 30
        top = dist * 0.5 # tan_fovy * 30
        # print(f"dist {dist}, left {left}, top {top}")
        origin = torch.tensor([0, 0, 0], dtype=torch.double, device=self.device)
        c1 = torch.tensor([left, top, dist], dtype=torch.double, device=self.device)
        c2 = torch.tensor([left, -top, dist], dtype=torch.double, device=self.device)
        c3 = torch.tensor([-left, -top, dist], dtype=torch.double, device=self.device)
        c4 = torch.tensor([-left, top, dist], dtype=torch.double, device=self.device)
        steps = 40
        t = torch.linspace(0, 1, steps, dtype=torch.double, device=self.device)
        frustum_vertices = torch.cat((
            (1 - t).unsqueeze(1) * origin + t.unsqueeze(1) * c1,
            (1 - t).unsqueeze(1) * origin + t.unsqueeze(1) * c2,
            (1 - t).unsqueeze(1) * origin + t.unsqueeze(1) * c3,
            (1 - t).unsqueeze(1) * origin + t.unsqueeze(1) * c4,
            (1 - t).unsqueeze(1) * c1 + t.unsqueeze(1) * c2,
            (1 - t).unsqueeze(1) * c2 + t.unsqueeze(1) * c3,
            (1 - t).unsqueeze(1) * c3 + t.unsqueeze(1) * c4,
            (1 - t).unsqueeze(1) * c4 + t.unsqueeze(1) * c1
        ))

        for (key, viewpoint) in self.viewpoint_stack.items():
            # Transform frustum vertices to world space
            frustum_vertices_world = self.keyframe_TFs[key].act(frustum_vertices.type(torch.float))

            # Add frustum vertices as gaussians
            self.gaussians.add_points(
                new_xyz=frustum_vertices_world,
                new_features_dc=frustum_color.repeat(frustum_vertices_world.shape[0], 1)[..., None],
                new_opacities=frustum_opacity.repeat(frustum_vertices_world.shape[0])[..., None],
                new_scales=frustum_scale.repeat(frustum_vertices_world.shape[0], 3),
                new_rotations=rotation.repeat(frustum_vertices_world.shape[0], 1)
            )

    def save_results(self, path, keyframes):
        self.optimize(keyframes=keyframes, iters=1, save_results=True, path=path)
        total_ssim, total_l1, total_psnr, num = 0, 0, 0, 0
        for (key, val) in self.rendering_vals.items():
            if "ssim" in key:
                total_ssim += val
                num += 1
            if "l1" in key:
                total_l1 += val
            if "psnr" in key:
                total_psnr += val
        self.rendering_vals["psnr mean"] = total_psnr / num
        self.rendering_vals["ssim mean"] = total_ssim / num
        self.rendering_vals["l1 mean"] = total_l1 / num

        # Write self.rendering_vals to a text file
        output_folder = path
        rendering_results_file = os.path.join(output_folder, "rendering_results.txt")
        with open(rendering_results_file, "w") as f:
            for key, value in self.rendering_vals.items():
                if "mean" in key:
                    f.write("\n")
                f.write(f"{key}: {value}\n")
            f.write(f"\n# of Map optimisations per keyframe:\n keyframe: # Map optimisations\n")
            for key, value in self.N_optimized_kf_gaussians.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n# of Gaussians: {self.gaussians._xyz.shape[0]}\n")
            
        # Create a folder with the current datetime
        optimized_poses_file = os.path.join(output_folder, "optimized_poses.pkl")
        with open(optimized_poses_file, "wb") as f:
            pickle.dump(self.optimized_poses, f)
        gaussinas_file = os.path.join(output_folder, "gaussians.ply")
        self.gaussians.save_ply(gaussinas_file)
        return