import random
import time

import torch
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
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.camera_utils import Camera 
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.geometry import constrain_points_to_ray, quat_mult
from gaussian_splatting.utils.graphics_utils import focal2fov
from gaussian_splatting.utils.pose_utils import update_pose
from gaussian_splatting.utils.slam_utils import get_loss_tracking_rgb, get_median_depth

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
        self.initialized = not self.monocular
        self.keyframe_optimizers = None
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        self.tracking_itr_num = config["gaussians"]["tracking_itr_num"]
        
        K_frame = dataset.camera_intrinsics.K_frame
        fx = K_frame[0, 0]
        fy = K_frame[1, 1]
        cx = K_frame[0, 2]
        cy = K_frame[1, 2]
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
        self.optimized_poses = {}
        self.N_optimized_kf_gaussians = {}

    def optimize(self, keyframes: SharedKeyframes, iters, save_results=False, path=None):
        print(f"run Gaussian Optimizer, number of keyframes: {len(keyframes)}")
        # if len(keyframes) > 2: return
        # del self.viewpoint_stack
        self.viewpoint_stack = {}
        self.keyframe_TFs = {}
        # del self.gaussians
        self.gaussians = GaussianModel(sh_degree=0)
        self.valid_masks = {}
        num_keyframes = len(keyframes)
        for idx in range(num_keyframes):
        # for idx in range(2):
            keyframe = keyframes[idx]
            # print(f"viewpoint_stack.keys() {self.viewpoint_stack.keys()}")
            # if idx in self.viewpoint_stack.keys(): continue
            print(f"Adding viewpoint for keyframe {keyframe.frame_id}")
            # gt_color, gt_depth, gt_pose = dataset[keyframe.frame_id]
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
            viewpoint.original_image = keyframe.img.clone().to(device=self.device)/2.0+0.5
            # print(f"imgshape {keyframe.img.shape}")
            # print(f"viewpoint.image_width {viewpoint.image_width}")
            # print(f"viewpoint.image_height {viewpoint.image_height}")
            # viewpoint.image_width = self.intrinsics.width
            # viewpoint.image_height = self.intrinsics.height
            self.viewpoint_stack[idx] = viewpoint
            self.keyframe_TFs[idx] = keyframe.T_WC
            # if i == len(keyframes):
            # print("add points to gaussians")
            c_conf_threshold = self.c_conf_threshold
            valid = (
                        keyframe.get_average_conf().reshape(-1)
                        > c_conf_threshold
                    )
            self.valid_masks[idx] = valid
            # valid = np.ones_like(valid, dtype=bool)
            scales_new = (keyframe.T_WC.data[0,-1] * keyframe.scales)
            opacities_new = keyframe.opacities
            w_rotations = quat_mult(keyframe.T_WC.data, keyframe.rotations)
            w_means = keyframe.T_WC.act(keyframe.X_canon + keyframe.offsets)


            if False and idx > 1:
                render_pkg = render(self.viewpoint_stack[idx], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                # ssim_loss_val = ssim(image, self.viewpoint_stack[idx].original_image)
                # l1_loss_val = l1_loss(image, self.viewpoint_stack[idx].original_image)
                l1_threshold = 0.5
                l1_loss_img = torch.abs(image - self.viewpoint_stack[idx].original_image).mean(dim=0).reshape(-1)
                l1_mask = valid * (l1_loss_img > l1_threshold)
                # print(f"l1_loss_mask shape reshaped {l1_loss_mask.shape}")
                print(f"l1_mask shape {l1_mask.shape}, l1_mask sum {l1_mask.sum()}")
                self.gaussians.add_points(
                    new_xyz=w_means[l1_mask],
                    new_features_dc=keyframe.SH[l1_mask],
                    new_opacities=opacities_new[l1_mask],
                    new_scales=scales_new[l1_mask],
                    new_rotations=w_rotations[l1_mask]
                )
            else:
                print(f"adding {valid.sum()} points to gaussians")
                self.gaussians.add_points(
                    new_xyz=w_means[valid],
                    new_features_dc=keyframe.SH[valid],
                    new_opacities=opacities_new[valid],
                    new_scales=scales_new[valid],
                    new_rotations=w_rotations[valid]
                )
            # self.gaussians.load_ply("/home/curdinst/repos/MASt3R-SLAM/logs/rgbd_dataset_freiburg1_desk_2025-04-17_09-48-43_wa.ply")
            print(f"num_gaussians: {self.gaussians._xyz.shape}")
            # break
        
        # render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        # image = render_pkg["render"]
        # print(f"image render shape {image.shape}")
        self.gaussians.init_lr(self.init_lr)
        self.gaussians.training_setup(self.opt_params)
        #         break
        if self.window_size == 1 and not save_results:
            optimisation_window = [num_keyframes - 1]
        elif num_keyframes > 2 and not save_results:
            optimisation_window = [num_keyframes - 2, num_keyframes - 1]
            rest_view_idxs = list(range(num_keyframes - 2))
            random.shuffle(rest_view_idxs)
            optimisation_window += rest_view_idxs[:self.window_size-2]
        else:
            optimisation_window = list(range(num_keyframes))
        # optimisation_window = list(range(num_keyframes))
        self.rendering_vals = {}
        print(f"optimisation_window {optimisation_window}")
        if not save_results:
            for frame_idx in optimisation_window:
                if frame_idx in self.N_optimized_kf_gaussians.keys() and iters > 0:
                    self.N_optimized_kf_gaussians[frame_idx] += 1
                elif iters == 0:
                    self.N_optimized_kf_gaussians[frame_idx] = 0
                else:
                    self.N_optimized_kf_gaussians[frame_idx] = 1
        for i in range(iters):
            self.iteration_count += 1
            loss_mapping = 0
            # for frame_index in range(num_keyframes):
            for frame_index in optimisation_window:
            # for frame_index in range(2):
                render_pkg = render(self.viewpoint_stack[frame_index], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
                # print(f"exposure_a {viewpoint.exposure_a}, exposure_b {viewpoint.exposure_b}")
                ssim_loss_val = ssim(image, self.viewpoint_stack[frame_index].original_image)
                # ssim_loss_val = 
                l1_loss_val = l1_loss(image, self.viewpoint_stack[frame_index].original_image)
                # loss_mapping = l1_loss_val * 0.75 + 0.25 * (1-ssim_loss_val)
                loss_mapping = l1_loss_val
                if i == 0 or i == iters - 1:
                    print(f"frame_index {frame_index} iteration {i} SSIM {round(ssim_loss_val.item(), 8)} L1 {round(l1_loss_val.item(), 8)}")
                # l1_loss_mask = torch.abs(image - self.viewpoint_stack[frame_index].original_image).mean(dim=0)
                # print(f"l1_loss_mask shape {l1_loss_mask.shape}")

                # print("image", image.shape)
                # print(f"render results: SSIM {round(ssim_loss_val.item(), 3)} L1 {round(l1_loss_val.item(), 3)}")

                save_plot = save_results
                self.rendering_vals[f"ssim_{frame_index}"] = ssim_loss_val.item()
                self.rendering_vals[f"l1_{frame_index}"] = l1_loss_val.item()
                if save_plot:
                    image_rearranged = einops.rearrange(image.cpu().detach().numpy(), "c h w -> h w c")
                    plt.figure()
                    plt.title(f"frame_index {frame_index} iteration {i} SSIM {round(ssim_loss_val.item(), 3)} L1 {round(l1_loss_val.item(), 3)}")
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
                        self.gaussians.update_learning_rate(idx)
                        loss_mapping = 0

        if save_results:
            self.draw_cameras()

        # Overwrite
        idx = 0
        # for frame_idx in range(num_keyframes):
        num_valid_list = [valid.sum().item() for valid in self.valid_masks.values()]
        gauss_indices = [sum(num_valid_list[:i]) for i in range(len(num_valid_list)+1)]
        print(f"num_valid_list {num_valid_list}, \ngauss_indices {gauss_indices}")
        for frame_idx in range(num_keyframes):
            valid = self.valid_masks[frame_idx]
            num_valid = valid.sum()
            # print(f"num gaussians {self.gaussians._xyz.shape}, num_valid {num_valid}")
            # print(f"num_valid: {num_valid} a {self.gaussians._xyz[idx:idx+num_valid].shape}, b {keyframes[frame_idx].X_canon[valid].shape}")
            idx_0, idx_1 = gauss_indices[frame_idx], gauss_indices[frame_idx + 1]
            
            keyframe = keyframes[frame_idx]
            T_CW = keyframe.T_WC.inv()
            # print(f"T_WC {keyframe.T_WC.data}")
            # print(f"T_CW {T_CW.data}")
            scales_w = torch.exp(self.gaussians._scaling[idx:idx+num_valid]) * T_CW.data[0,-1]
            rotations_w = self.gaussians._rotation[idx:idx+num_valid]
            rotations_kf = quat_mult(T_CW.data, rotations_w)
            means_w = T_CW.act(self.gaussians._xyz[idx:idx+num_valid])

            # print("scaling:", self.gaussians._scaling[idx:idx+num_valid])
            # print("scales kf :", keyframes[frame_idx].scales)
            keyframe.update_gaussians(
                valid_mask=valid,
                scale=scales_w,
                rotation=rotations_kf,
                SH=einops.rearrange(self.gaussians._features_dc[idx:idx+num_valid], "wh d c -> wh c d"),
                opacity=self.gaussians._opacity[idx:idx+num_valid],
                mean=means_w,
            )
            # keyframe.update_gaussians(
            #     valid_mask=valid,
            #     scale=self.gaussians._scaling[idx_0:idx_1],
            #     rotation=self.gaussians._rotation[idx_0:idx_1],
            #     SH=einops.rearrange(self.gaussians._features_dc[idx_0:idx_1], "wh d c -> wh c d"),
            #     opacity=self.gaussians._opacity[idx_0:idx_1],
            #     mean=self.gaussians._xyz[idx_0:idx_1],
            # )
            keyframes[frame_idx] = keyframe
        
            # X_canon = keyframes[frame_idx].X_canon[valid].clone()
            # keyframes.update_gaussians(
            #     frame_idx=frame_idx,
            #     valid_mask=valid,
            #     offset=self.gaussians._xyz[idx:idx+num_valid]  - X_canon,
            #     scale=self.gaussians._scaling[idx:idx+num_valid],
            #     rotation=self.gaussians._rotation[idx:idx+num_valid],
            #     opacity=self.gaussians._opacity[idx:idx+num_valid],
            #     SH=einops.rearrange(self.gaussians._features_dc[idx:idx+num_valid], "wh d c -> wh c d"),
            # )
            # keyframes[frame_idx].offsets[valid] = self.gaussians._xyz[idx:idx+num_valid] - keyframes[frame_idx].X_canon[valid].clone()
            # keyframes[frame_idx].rotations[valid] = self.gaussians._rotation[idx:idx+num_valid].clone()
            # keyframes[frame_idx].SH[valid] = einops.rearrange(self.gaussians._features_dc[idx:idx+num_valid], "wh d c -> wh c d").clone()
            # keyframes[frame_idx].scales[valid] = self.gaussians._scaling[idx:idx+num_valid].clone()
            # keyframes[frame_idx].opacities[valid] = self.gaussians._opacity[idx:idx+num_valid].clone()
            idx += num_valid
        print(f"updated gaussians of {num_keyframes} keyframes")
        if save_results:
            for kf_idx in range(num_keyframes):
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
        print(f"Initial pose for frame {cur_frame_idx}: T: {viewpoint.T}, R: {viewpoint.R}")

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
            loss_tracking = get_loss_tracking_rgb(
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
        print(f"optimized pose for frame {cur_frame_idx}: T: {viewpoint.T}, R: {viewpoint.R}")
        # self.median_depth = get_median_depth(depth, opacity)
        return render_pkg



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
        total_ssim, total_l1, num = 0, 0, 0
        for (key, val) in self.rendering_vals.items():
            if "ssim" in key:
                total_ssim += val
                num += 1
            if "l1" in key:
                total_l1 += val
        self.rendering_vals["ssim mean"] = total_ssim / num
        self.rendering_vals["l1 mean"] = total_l1 / num

        # Write self.rendering_vals to a text file
        output_folder = path
        rendering_results_file = os.path.join(output_folder, "rendering_results.txt")
        with open(rendering_results_file, "w") as f:
            for key, value in self.rendering_vals.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n# of Map optimisations per keyframe:\n keyframe: # Map optimisations\n")
            for key, value in self.N_optimized_kf_gaussians.items():
                f.write(f"{key}: {value}\n")
        # Create a folder with the current datetime
        optimized_poses_file = os.path.join(output_folder, "optimized_poses.pkl")
        with open(optimized_poses_file, "wb") as f:
            pickle.dump(self.optimized_poses, f)
        gaussinas_file = os.path.join(output_folder, "gaussians.ply")
        self.gaussians.save_ply(gaussinas_file)
        return