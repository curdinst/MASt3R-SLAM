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

from matplotlib import pyplot as plt

from munch import munchify




class GaussianOptimizer:
    def __init__(self, config, device, learning_rate=0.01):
        """
        Initialize the GaussianOptimizer.

        Args:
            learning_rate (float): The learning rate for optimization.
        """
        self.learning_rate = learning_rate

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
        
        print(f"intrinsics: {config['gaussians']['Calibration']}")
        intrinsics = munchify(config["gaussians"]["Calibration"])
        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            W=intrinsics.width,
            H=intrinsics.height,
        ).transpose(0, 1)
        self.fovx = focal2fov(intrinsics.fx, intrinsics.width)
        self.fovy = focal2fov(intrinsics.fy, intrinsics.height)
        self.intrinsics = intrinsics
        self.projection_matrix = self.projection_matrix.to(device=device)
        self.pipeline_params = munchify(config["gaussians"]["pipeline_params"])

    def get_loss_tracking_rgb(config, image, opacity, viewpoint):
        image = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
        gt_image = viewpoint.original_image.cuda()
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = config["gaussians"]["rgb_boundary_threshold"]
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
        l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
        return l1.mean()
    
    def tracking(self, cur_frame_idx, viewpoint: Camera, config, tracking_itr_num=30):
        prev_idx = cur_frame_idx - 1
        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": config["gaussians"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": config["gaussians"]["lr"]["cam_trans_delta"],
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
        pipeline_params = munchify(config["pipeline_params"])

        for tracking_itr in range(tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = self.get_loss_tracking_rgb(
                config, image, opacity, viewpoint
            )
            print(f"Tracking loss: {loss_tracking.item()}")
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if converged:
                print("Converged")

                break

        # self.median_depth = get_median_depth(depth, opacity)
        plt.imshow(einops.rearrange(viewpoint.original_image.detach().cpu().numpy(), "c h w -> h w c")*0.5 + einops.rearrange(image.detach().cpu().numpy(), "c h w -> h w c")*0.5)
        plt.show()
        return render_pkg

    def optimize(self, dataset, keyframes: SharedKeyframes, iters):
        print(f"run Gaussian Optimizer, number of keyframes: {len(keyframes)}")
        if len(keyframes) > 2: return
        # del self.viewpoint_stack
        self.viewpoint_stack = {}
        # del self.gaussians
        self.gaussians = GaussianModel(sh_degree=0)
        for idx in range(len(keyframes)):
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
                self.intrinsics.fx,
                self.intrinsics.fy,
                self.intrinsics.cx,
                self.intrinsics.cy,
                self.fovx,
                self.fovy,
                self.intrinsics.height,
                self.intrinsics.width,
                device=self.device,
            )
            print(f"keyframe {keyframe.frame_id} T_WC {keyframe.T_WC.data}")
            rot = R.from_quat(keyframe.T_WC.data[0,3:7].cpu()).as_matrix()
            viewpoint.R  = torch.from_numpy(rot).to(device=self.device).T
            viewpoint.T = -viewpoint.R.float() @ keyframe.T_WC.data[0,:3].float()
            # viewpoint.R = torch.from_numpy(rot).to(device=self.device)
            # viewpoint.T = keyframe.T_WC.data[0,:3].float()
            viewpoint.original_image = keyframe.img
            # print(f"imgshape {keyframe.img.shape}")
            # print(f"viewpoint.image_width {viewpoint.image_width}")
            # print(f"viewpoint.image_height {viewpoint.image_height}")
            viewpoint.image_width = 512
            viewpoint.image_height = 384
            self.viewpoint_stack[idx] = viewpoint
            # if i == len(keyframes):
            print("add points to gaussians")
            c_conf_threshold = 1.5
            valid = (
                        keyframe.get_average_conf().reshape(-1)
                        > c_conf_threshold
                    )
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
                self.gaussians.add_points(
                    new_xyz=w_means[valid],
                    new_features_dc=keyframe.SH[valid],
                    new_opacities=opacities_new[valid],
                    new_scales=scales_new[valid],
                    new_rotations=w_rotations[valid]
                )
            # self.gaussians.load_ply("/home/curdinst/repos/MASt3R-SLAM/logs/rgbd_dataset_freiburg1_desk_2025-04-17_09-48-43_wa.ply")
            print(f"num_gaussians: {self.gaussians._xyz.shape}")
            break
        
        # render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        # image = render_pkg["render"]
        # print(f"image render shape {image.shape}")

        #         break
        for i in range(iters):
            self.iteration_count += 1
            for frame_index in range(len(keyframes)):
            # for frame_index in range(2):
                render_pkg = render(self.viewpoint_stack[frame_index], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                ssim_loss_val = ssim(image, self.viewpoint_stack[frame_index].original_image)
                l1_loss_val = l1_loss(image, self.viewpoint_stack[frame_index].original_image)

                l1_loss_mask = torch.abs(image - self.viewpoint_stack[frame_index].original_image).mean(dim=0)
                print(f"l1_loss_mask shape {l1_loss_mask.shape}")

                print("image", image.shape)
                print(f"render results: SSIM {round(ssim_loss_val.item(), 3)} L1 {round(l1_loss_val.item(), 3)}")
                image_rearranged = einops.rearrange(image.cpu().detach().numpy(), "c h w -> h w c")
                plt.figure()
                plt.title(f"frame_index {frame_index} iteration {i} SSIM {round(ssim_loss_val.item(), 3)} L1 {round(l1_loss_val.item(), 3)}")
                plt.axis("off")
                plt.subplot(1, 3, 1)
                a,b = np.min(image_rearranged), np.max(image_rearranged)
                plt.imshow((image_rearranged - a)/(b-a))
                plt.subplot(1, 3, 2)
                gt_img_rearranged = einops.rearrange(self.viewpoint_stack[frame_index].original_image.cpu().detach().numpy(), "c h w -> h w c")
                a,b = np.min(gt_img_rearranged), np.max(gt_img_rearranged)
                plt.imshow((gt_img_rearranged- a)/(b-a) )
                plt.subplot(1, 3, 3)
                plt.imshow(l1_loss_mask.cpu().detach().numpy())
                path = "/home/curdinst/repos/MASt3R-SLAM/logs/"
                
                plt.savefig(path + f"render_{frame_index}.png")
                plt.close()
                return
        #         del render_pkg
        #         break
        #     break
                # self._save_checkpoint()
        return