import random
import time

import torch
import torch.multiprocessing as mp
import einops
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
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

    def optimize(self, dataset, keyframes: SharedKeyframes, iters):
        print("run Gaussian Optimizer")
        
        for idx in range(len(keyframes)):

            keyframe = keyframes[idx]
            print(f"viewpoint_stack.keys() {self.viewpoint_stack.keys()}")
            if idx in self.viewpoint_stack.keys(): continue
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
            print("keyframe.T_WC", keyframe.T_WC.data)
            rot = R.from_quat(keyframe.T_WC.data[0,3:7].cpu()).as_matrix()
            viewpoint.R  = torch.from_numpy(rot).to(device=self.device).T
            viewpoint.T = -viewpoint.R.float() @ keyframe.T_WC.data[0,:3].float()
            viewpoint.original_image = keyframe.img
            print(f"imgshape {keyframe.img.shape}")
            print(f"viewpoint.image_width {viewpoint.image_width}")
            print(f"viewpoint.image_height {viewpoint.image_height}")
            viewpoint.image_width = 512
            viewpoint.image_height = 384
            self.viewpoint_stack[idx] = viewpoint
            # if i == len(keyframes):
            print("add points to gaussians")

            scales_new = (keyframe.T_WC.data[0,-1] * keyframe.scales)
            opacities_new = keyframe.opacities
            w_rotations = quat_mult(keyframe.T_WC.data, keyframe.rotations)
            w_means = keyframe.T_WC.act(keyframe.X_canon + keyframe.offsets)

            self.gaussians.add_points(
                new_xyz=w_means,
                new_features_dc=keyframe.SH,
                new_opacities=opacities_new,
                new_scales=scales_new,
                new_rotations=w_rotations
            )
            # self.gaussians.load_ply("/home/curdinst/repos/MASt3R-SLAM/logs/rgbd_dataset_freiburg1_desk_2025-04-17_09-48-43_wa.ply")
            print(f"num_gaussians: {self.gaussians._xyz.shape}")
        
        # render_pkg = render(viewpoint, self.gaussians, self.pipeline_params, self.background)
        # image = render_pkg["render"]
        # print(f"image render shape {image.shape}")

        #         break
        for i in range(iters):
            self.iteration_count += 1
            for frame_index in range(len(keyframes)):
                render_pkg = render(self.viewpoint_stack[frame_index], self.gaussians, self.pipeline_params, self.background)
                image = render_pkg["render"]
                print("image", image.shape)
                image_rearranged = einops.rearrange(image.cpu().detach().numpy(), "c h w -> h w c")
                plt.subplot(1, 2, 1)
                plt.imshow(image_rearranged)
                plt.subplot(1, 2, 2)
                gt_img_rearranged = einops.rearrange(self.viewpoint_stack[frame_index].original_image.cpu().detach().numpy(), "c h w -> h w c")
                plt.imshow(gt_img_rearranged)
                path = "/home/curdinst/repos/MASt3R-SLAM/logs/"
                
                plt.savefig(path + f"render_{frame_index}.png")
        #         del render_pkg
        #         break
        #     break
                # self._save_checkpoint()
        return