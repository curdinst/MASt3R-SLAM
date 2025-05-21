import torch
import torch.nn.functional as F
import yaml
import pathlib
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.evaluate import save_gaussian_new_ply, save_as_ply
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
import lietorch
from mast3r_slam.dataloader import Intrinsics, load_dataset
from mast3r_slam.frame import Frame
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.mast3r_utils import mast3r_asymmetric_inference
from mast3r_slam.evaluate import save_gaussian_new_ply, save_as_ply
import mast3r_slam.evaluate as eval
import mast3r_slam.utils.geometry as geometry
from matplotlib import pyplot as plt
import einops
import numpy as np
import time
from gaussian_splatting.gaussian_optimizer import GaussianOptimizer
from scipy.spatial.transform import Rotation

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
from gaussian_splatting.utils.slam_utils import get_loss_tracking_rgb, get_loss_tracking_rgbd, get_loss_mapping_rgbd
from gaussian_splatting.utils.image_utils import psnr
from munch import munchify
from torchvision.utils import save_image


grad_threshold = 0.1

# Load images
reduced = False
load_config("config/base.yaml")

device = "cuda:0"
model = load_mast3r(device=device, path="checkpoints/MASt3R_gaussians_v1.pth")
# model = load_mast3r(device=device)
model.share_memory()
print("loaded model")

# dataset_path = "datasets/tum/rgbd_dataset_freiburg1_room/"
dataset_path = "datasets/replica/office2/"
dataset = load_dataset(dataset_path)
H, W = dataset.get_img_shape()[0]
img_size = (H, W)
print("Image size:", img_size)

img1_idx, img2_idx = 0, 12
timestamp1, img1 = dataset[img1_idx]
timestamp2, img2 = dataset[img2_idx]

image1_save_path = "logs/image" + str(img1_idx) + ".pt"
image2_save_path = "logs/image" + str(img2_idx) + ".pt"

T_WC = lietorch.Sim3.Identity(1, device=device)

frame1 = create_frame(img1_idx, img1, img_size=dataset.img_size,  T_WC=T_WC)
frame2 = create_frame(img2_idx, img2, img_size=dataset.img_size, T_WC=T_WC)

# torch.save(frame1.img, image1_save_path)
# torch.save(frame2.img, image2_save_path)
def spatial_derivative(img):
    """
    Compute spatial derivatives (gradients) in x and y directions
    using Sobel filters.

    Args:
        img (torch.Tensor): Image tensor of shape (C, H, W)

    Returns:
        grad_x, grad_y: each of shape (C, H, W)
    """
    if img.ndim != 3:
        raise ValueError("Image must be 3D tensor (C, H, W)")

    C, H, W = img.shape

    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device)

    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2,  1]], dtype=torch.float32, device=device)

    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    grad_x = []
    grad_y = []

    for c in range(C):
        channel = img[c:c+1, :, :].unsqueeze(0)  # shape: (1, 1, H, W)
        gx = F.conv2d(channel, sobel_x, padding=1)
        gy = F.conv2d(channel, sobel_y, padding=1)
        grad_x.append(gx.squeeze(0))
        grad_y.append(gy.squeeze(0))

    grad_x = torch.cat(grad_x, dim=0)  # (C, H, W)
    grad_y = torch.cat(grad_y, dim=0)  # (C, H, W)

    return grad_x, grad_y

def get_mask(img):
    grad_x, grad_y = spatial_derivative(img)
    grad_x, grad_y = torch.abs(grad_x), torch.abs(grad_y)
    print(grad_x.shape, grad_y.shape)

    print("Gradient x max:", grad_x.max(), "min:", grad_x.min())
    print("Gradient y max:", grad_y.max(), "min:", grad_y.min())

    grad_x_max, grad_x_max_indices = torch.max(grad_x, dim=0)
    grad_y_max, grad_y_max_indices = torch.max(grad_y, dim=0)

    mask_x = grad_x_max < grad_threshold
    mask_y = grad_y_max < grad_threshold
    mask = mask_x & mask_y
    print("Mask shape:", mask.shape)
    print(f"mask.sum(): {mask.sum()}, mask.numel(): {mask.numel()}, mask.sum()/mask.numel(): {mask.sum()/mask.numel()}")

    H_mask, W_mask = mask.shape[0] // 2, mask.shape[1] // 2
    mask_downsampled = F.upsample(mask.float().unsqueeze(0).unsqueeze(0), size=(H_mask,W_mask), mode="bilinear")
    print("Downsampled mask shape:", mask_downsampled.shape)
    mask_downsampled = (mask_downsampled > 0.9)
    print("Downsampled mask sum:", mask_downsampled.sum(), "numel:", mask_downsampled.numel(), "ratio:", mask_downsampled.sum()/mask_downsampled.numel())
    mask_upsampled = F.interpolate(mask_downsampled.float(), size=(H, W), mode="nearest").squeeze(0).squeeze(0).bool()
    print("Upsampled mask shape:", mask_upsampled.shape)
    return mask_downsampled.squeeze(0).squeeze(0), mask_upsampled

print(f"frame1 img shape: {frame1.img.shape}")
image1 = einops.rearrange(frame1.img, "b c h w ->(b c) h w")
image1 = image1*0.5 + 0.5
print(f"image1 max: {image1.max()}, min: {image1.min()}")
# plt.imsave("logs/feat1.png", image1)
# dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
# pred1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
# pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
# pred2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
# X, C, D, Q, S, R, SH, O, M = mast3r_asymmetric_inference(model=model, frame_i=frame1, frame_j=frame2)

# frame2.img = einops.rearrange(frame2.img, "b c h w ->(b c) h w")
idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, gaussian_params = mast3r_match_asymmetric(model=model, frame_i=frame1, frame_j=frame2)
(Sii, Rii, SHii, Oii, Mii, Sji, Rji, SHji, Oji, Mji) = gaussian_params

save_dir = pathlib.Path("logs")
save_dir.mkdir(exist_ok=True, parents=True)
filename = "gaussians_" + str(img1_idx) + "_" + str(img2_idx) + ".ply"
recon_file = save_dir / filename
if recon_file.exists():
    recon_file.unlink()
print(Mii.shape)
fake_means_1 = torch.zeros((1, 3, H, W)).to(device)
covariances = geometry.build_covariance(Sii, Rii)
# einops.rearrange(Sii, "b h w c -> b (h w) c")
# einops.rearrange(Rii, "b h w c -> b (h w) c")
spherical_harmonics = einops.rearrange(SHii, "(h w) c d -> (c d) h w", h=H, w=W)
opacities = einops.rearrange(Oii, "(h w) c -> c h w", h=H, w=W)
means = einops.rearrange(Mii, "(h w) c -> c h w", h=H, w=W)
print(f"covariances shape: {covariances.shape}")
covariances = einops.rearrange(covariances, "(h w) x y -> h w x y", h=H, w=W)


mask_downsampled, mask_upsampled = get_mask(image1)
print("Mask shape:", mask_downsampled.shape)
print(mask_downsampled)

# exit()
num_fused = mask_downsampled.sum()
#upsample mask again to take gaussians that are not in the mask
indices = torch.nonzero(mask_downsampled, as_tuple=False)
print(f"Indices shape: {indices.shape}, num_fused: {num_fused}")

# Get u, v coordinates from indices
u = indices[:, 0] * 2
v = indices[:, 1] * 2
print(f"u shape: {u.shape}, v shape: {v.shape}")
print(f"max u {u.max()}, max v {v.max()}")

# Gather neighbor values
val0 = means[:, u, v]
val1 = means[:, u, v+1]
val2 = means[:, u+1, v]
val3 = means[:, u+1, v+1]
print(f"val0: {val0}")
fused_means = ((val0 + val1 + val2 + val3) / 4)
covariance_factor = 1.0
fused_opacities = (opacities[:, u, v] + opacities[:, u, v+1] + opacities[:, u+1, v] + opacities[:, u+1, v+1]) / 4.0
fused_covariances = (covariances[u, v, ...] + covariances[u, v+1, ...] + covariances[u+1, v, ...] + covariances[u+1, v+1, ...]) * covariance_factor
fused_sh = (spherical_harmonics[:, u, v] + spherical_harmonics[:, u, v+1] + spherical_harmonics[:, u+1, v] + spherical_harmonics[:, u+1, v+1]) / 4.0
print("Fused means shape:", fused_means.shape)

fused_means = einops.rearrange(fused_means, "c n -> n c")
fused_opacities = einops.rearrange(fused_opacities, "c n-> n c")
fused_sh = einops.rearrange(fused_sh, "c n -> n c")
print(f"fused_menas {fused_means}")

original_means = means[:, ~mask_upsampled]
original_opacities = opacities[:, ~mask_upsampled]
original_covariances = covariances[~mask_upsampled, ...]
original_sh = spherical_harmonics[:, ~mask_upsampled]

original_means = einops.rearrange(original_means, "c n -> n c")
original_opacities = einops.rearrange(original_opacities, "c n-> n c")
original_sh = einops.rearrange(original_sh, "c n -> n c")

reduced_means = torch.cat((fused_means, original_means), dim=0)
reduced_opacities = torch.cat((fused_opacities, original_opacities), dim=0)
reduced_sh = torch.cat((fused_sh, original_sh), dim=0)
reduced_covariances = torch.cat((fused_covariances, original_covariances), dim=0)
num_gaussians = reduced_means.shape[0]
reduced = True
# save_as_ply(pred1, pred1, recon_file)
reduced_name = f"gaussians_reduced_th_{grad_threshold}_covf_{covariance_factor}_n_{num_gaussians}" if reduced else f"gaussians_original"

results_path = pathlib.Path(f"/home/curdinst/repos/MASt3R-SLAM/logs/{reduced_name}/")
results_path.mkdir(exist_ok=True, parents=True)
gaussians_file = results_path / f"gaussians.ply"
save_ply = False
if not reduced and save_ply:
    save_gaussian_new_ply(
        save_path=gaussians_file,
        S=Sii.cpu().numpy(),
        R=Rii.cpu().numpy(),
        M=Mii.cpu().numpy(),
        SH=SHii.squeeze(-1).cpu().numpy(),
        O=Oii.cpu().numpy(),
    )
elif save_ply:
    save_gaussian_new_ply(
        save_path=gaussians_file,
        M=reduced_means.cpu().numpy(),
        SH=reduced_sh.squeeze(-1).cpu().numpy(),
        O=reduced_opacities.cpu().numpy(),
        covariance=reduced_covariances,
    )

print("predctions done")

def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()
        quaternion = torch.from_numpy(quaternion).to(device)

        return quaternion, scale
print(f"SHii shape: {SHii.shape}")
reduced_sh = einops.rearrange(reduced_sh, "n c -> n c 1")
print(f"reduced_sh shape: {reduced_sh.shape}")
reduced_rotations, reduced_scales = covariance_to_quaternion_and_scale(reduced_covariances)
gaussians = GaussianModel(sh_degree=0)
if not reduced:
    gaussians.add_points(
                        new_xyz=Mii,
                        new_features_dc=SHii,
                        new_opacities=Oii,
                        new_scales=Sii,
                        new_rotations=Rii,
                    )
else:
    gaussians.add_points(
                        new_xyz=reduced_means,
                        new_features_dc=reduced_sh,
                        new_opacities=reduced_opacities,
                        new_scales=reduced_scales,
                        new_rotations=reduced_rotations
                    )

K_frame = dataset.camera_intrinsics.K_frame
print(f"K_frame {K_frame}")
fx = K_frame[0, 0]
fy = K_frame[1, 1]
cx = K_frame[0, 2]
cy = K_frame[1, 2]
fovx = focal2fov(fx, W)
fovy = focal2fov(fy, H)
projection_matrix = getProjectionMatrix2( znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H).transpose(0, 1).to(device=device)
viewpoint1 = Camera(
                1,
                None,
                None,
                None,
                projection_matrix,
                fx,
                fy,
                cx,
                cy,
                fovx,
                fovy,
                H,
                W,
                device=device,
            )

print(f"viewpoint1.T {viewpoint1.T}")
print(f"viewpoint1.R {viewpoint1.R}")
background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
pipeline_params = munchify(config["gaussians"]["pipeline_params"])
render_pkg = render(viewpoint1, gaussians, pipeline_params, background)

rendered_img = render_pkg['render']
print(f"Rendered image shape: {rendered_img.shape}")
psnr_value = psnr(rendered_img.unsqueeze(0), image1.unsqueeze(0)).item()

print("psnr: ", psnr(rendered_img.unsqueeze(0)[...,10:-10,10:-10], image1.unsqueeze(0)[...,10:-10,10:-10]))
print("ssim: ", ssim(rendered_img, image1))
print(f"image1 max {image1.max()}, min {image1.min()}")
print(f"rendered_img max {rendered_img.max()}, min {rendered_img.min()}")
# Save the rendered image as a PNG file
# save_image(rendered_img, results_path / "render.png")
# save_image(image1, results_path / "gt_image.png")
print("Rendered image saved as logs/rendered_image.png")
# print(image1)
# print(rendered_img)

# image1 = image1[...,10:-10,10:-10]
gt_img_rearranged = einops.rearrange(image1.cpu().detach().numpy(), "c h w -> h w c")
a,b = np.min(gt_img_rearranged), np.max(gt_img_rearranged)
plt.figure()
plt.imshow((gt_img_rearranged- a)/(b-a))
plt.title("Ground Truth Image")
plt.savefig(results_path / f"gt_image1.png")
plt.close()
plt.figure()
# rendered_img = rendered_img[...,10:-10,10:-10]
plt.imshow(rendered_img.cpu().detach().numpy().transpose(1,2,0))
plt.title(f"Rendered Image - PSNR: {psnr_value:.2f}")
plt.savefig(results_path / f"rendered_image1.png")
plt.close()

plt.figure()
show_mask = rendered_img.clone()
show_mask[:, ~mask_upsampled] = torch.zeros((3, H*W-mask_upsampled.sum())).type(torch.float32).to(device=device)
plt.imshow(show_mask.cpu().detach().numpy().transpose(1,2,0))
plt.title("Mask")
plt.savefig(results_path / f"mask1.png")
plt.close()

print("saved")