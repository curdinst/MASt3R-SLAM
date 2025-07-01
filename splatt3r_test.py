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


reduced = True
grad_threshold = 0.2
depth_grad_threshold = 0.2
scale_factor = 1.0
save_ply = True


# Load images
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

img1_idx, img2_idx = 0, 24
timestamp1, img1 = dataset[img1_idx]
timestamp2, img2 = dataset[img2_idx]

image1_save_path = "logs/image" + str(img1_idx) + ".pt"
image2_save_path = "logs/image" + str(img2_idx) + ".pt"

T_WC = lietorch.Sim3.Identity(1, device=device)

img_size = dataset.img_size
# img_size = 256
# H, W = 144, 256
print(f"img_size: {img_size}, H: {H}, W: {W}")
frame1 = create_frame(img1_idx, img1, img_size=img_size,  T_WC=T_WC)
frame2 = create_frame(img2_idx, img2, img_size=img_size, T_WC=T_WC)

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

def get_mask(img, depth_img):
    grad_x, grad_y = spatial_derivative(img)
    grad_x, grad_y = torch.abs(grad_x), torch.abs(grad_y)
    print(grad_x.shape, grad_y.shape)
    depth_grad_x, depth_grad_y = spatial_derivative(depth_img.unsqueeze(0))
    depth_grad_x, depth_grad_y = torch.abs(depth_grad_x), torch.abs(depth_grad_y)
    print(f"depth_grad_x shape: {depth_grad_x.shape}, depth_grad_y shape: {depth_grad_y.shape}")
    print(f"max depth_grad_x: {depth_grad_x.max()}, min depth_grad_x: {depth_grad_x.min()}")
    print(f"max depth_grad_y: {depth_grad_y.max()}, min depth_grad_y: {depth_grad_y.min()}")

    print("Gradient x max:", grad_x.max(), "min:", grad_x.min())
    print("Gradient y max:", grad_y.max(), "min:", grad_y.min())

    grad_x_max, grad_x_max_indices = torch.max(grad_x, dim=0)
    grad_y_max, grad_y_max_indices = torch.max(grad_y, dim=0)

    mask_x = grad_x_max < grad_threshold
    mask_y = grad_y_max < grad_threshold
    depth_mask_x = depth_grad_x.squeeze(0) < depth_grad_threshold
    depth_mask_y = depth_grad_y.squeeze(0) < depth_grad_threshold
    print(f"mask_x shape: {mask_x.shape}, mask_y shape: {mask_y.shape}")
    print(f"depth_mask_x shape: {depth_mask_x.shape}, depth_mask_y shape: {depth_mask_y.shape}")
    mask = mask_x & mask_y & depth_mask_x & depth_mask_y
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

def average_quaternions(quaternions):
    """
    Average a batch of unit quaternions using Markley's method.
    Args:
        quaternions: Tensor of shape (..., N, 4), where N is the number of quaternions.
    Returns:
        avg_quaternion: Tensor of shape (..., 4)
    """
    # Ensure input is normalized
    print(quaternions)
    quaternions = F.normalize(quaternions, dim=-1)
    print(quaternions)
    # Compute symmetric accumulator matrix A = sum(q_i * q_i^T)
    A = torch.einsum("...ni,...nj->...ij", quaternions, quaternions)

    # Compute eigenvectors and eigenvalues of A
    eigvals, eigvecs = torch.linalg.eigh(A)

    # Eigenvector with largest eigenvalue is the average quaternion
    avg_quaternion = eigvecs[..., -1]  # (..., 4)
    return avg_quaternion

def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        # U, S, V = torch.linalg.svd(covariance)
        S, U = torch.linalg.eig(covariance)
        S = S.real
        U = U.real
        print(f"S: \n{S}, U: \n{U}")
        # Take the real part of S and U
        rotation = U
        identity_check = torch.allclose(
        rotation.transpose(-1, -2) @ rotation, 
        torch.eye(3, device=rotation.device, dtype=rotation.dtype).expand_as(rotation),
        atol=1e-6
        )
        determinant_check = torch.allclose(
            torch.linalg.det(rotation), 
            torch.ones(rotation.shape[:-2], device=rotation.device, dtype=rotation.dtype),
            atol=1e-6
        )
        print(f"The rotation matrix is not orthonormal, identity_check: {identity_check}, determinant_check: {determinant_check}, det: {torch.linalg.det(rotation)}.")
        # if not (identity_check and determinant_check):
        #     raise ValueError(f"The rotation matrix is not orthonormal, identity_check: {identity_check}, determinant_check: {determinant_check}, det: {torch.linalg.det(rotation)}.")
        # else:
        #     print("The rotation matrix is orthonormal.")

        # Swap eigenvalue positions and adjust U to ensure determinant of 1
        negative_determinants = torch.linalg.det(U) < 0
        U[negative_determinants,..., -1] = -U[negative_determinants,..., -1]
        # print(F"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)

        # The rotation matrix is U*Vt
        # rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix = U
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # print(f"covariance: {covariance[0,...]}")
        # print(f"RSStRt: {torch.bmm(torch.bmm(rotation_matrix,scale.diag_embed()),scale.diag_embed().transpose(-2, -1))}")

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()
        quaternion = torch.from_numpy(quaternion).to(device)

        return quaternion, scale

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
# print(f"covariances shape: {covariances.shape}")
covariances = einops.rearrange(covariances, "(h w) x y -> h w x y", h=H, w=W)
scales = einops.rearrange(Sii, "(h w) c -> c h w", h=H, w=W)
rotations = einops.rearrange(Rii, "(h w) c -> c h w", h=H, w=W)

depths = Mii.norm(dim=1)
depth_img = einops.rearrange(depths, "(h w) -> h w", h=H, w=W)
print(f"depth_img shape: {depth_img.shape}, min: {depth_img.min()}, max: {depth_img.max()}")
# Save depth image as a PNG file
depth_img_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
plt.figure()
plt.imshow(depth_img_normalized.cpu().detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title("Depth Image")
plt.savefig("logs/depth_image.png")
plt.close()
mask_downsampled, mask_upsampled = get_mask(image1, depth_img)
print("Mask shape:", mask_downsampled.shape)
print(mask_downsampled)

num_fused = mask_downsampled.sum()
#upsample mask again to take gaussians that are not in the mask
indices = torch.nonzero(mask_downsampled, as_tuple=False)
print(f"Indices shape: {indices.shape}, num_fused: {num_fused}")

# Get u, v coordinates from indices
u = indices[:, 0] * 2
v = indices[:, 1] * 2
print(f"u shape: {u.shape}, v shape: {v.shape}")
# print(f"max u {u.max()}, max v {v.max()}")

fused_means = (means[:, u, v] + means[:, u, v+1] + means[:, u+1, v] + means[:, u+1, v+1]) / 4.0
mean_offset1 = means[:, u, v] - fused_means
mean_offset2 = means[:, u, v+1] - fused_means
mean_offset3 = means[:, u+1, v] - fused_means
mean_offset4 = means[:, u+1, v+1] - fused_means

mean_offsets = torch.mean(torch.stack((mean_offset1, mean_offset2, mean_offset3, mean_offset4), dim=0), dim=0)
print(f"mean_offsets shape: {mean_offsets.shape}")
# offset_threshold = 0.05
# too_large_offset = (mean_offset1.norm(dim=0) > offset_threshold) | (mean_offset2.norm(dim=0) > offset_threshold) | (mean_offset3.norm(dim=0) > offset_threshold) | (mean_offset4.norm(dim=0) > offset_threshold)
# print(f"too_large_offset shape: {too_large_offset.shape}, sum: {too_large_offset.sum()}")
# valid_offset = ~too_large_offset
# print(f"mask_upsampled.shape: {mask_upsampled.shape}, valid_offset shape: {valid_offset.shape}, valid_offset sum: {valid_offset.sum()}")
# mask_upsampled[u[too_large_offset], v[too_large_offset]] = False
# mask_upsampled[u[too_large_offset]+1, v[too_large_offset]] = False
# mask_upsampled[u[too_large_offset], v[too_large_offset]+1] = False
# mask_upsampled[u[too_large_offset]+1, v[too_large_offset]+1] = False

# mean_offset1 = mean_offset1[:, valid_offset]
# mean_offset2 = mean_offset2[:, valid_offset]
# mean_offset3 = mean_offset3[:, valid_offset]
# mean_offset4 = mean_offset4[:, valid_offset]
# u = u[valid_offset]
# v = v[valid_offset]
fused_means = (means[:, u, v] + means[:, u, v+1] + means[:, u+1, v] + means[:, u+1, v+1]) / 4.0

print(f"mean_offset1 shape: {mean_offset1.shape}")
print(f"mean_offset1 max: {mean_offset1.max()}, min: {mean_offset1.min()}, mean: {mean_offset1.mean(axis=1)}") 
print(F"mean_offset2 max: {mean_offset2.max()}, min: {mean_offset2.min()}, mean: {mean_offset2.mean(axis=1)}") 
print(f"mean_offset3 max: {mean_offset3.max()}, min: {mean_offset3.min()}, mean: {mean_offset3.mean(axis=1)}") 
print(f"mean_offset4 max: {mean_offset4.max()}, min: {mean_offset4.min()}, mean: {mean_offset4.mean(axis=1)}")

matrix1 = torch.einsum('ji,ki->ijk', mean_offset1, mean_offset1)
matrix2 = torch.einsum('ji,ki->ijk', mean_offset2, mean_offset2)
matrix3 = torch.einsum('ji,ki->ijk', mean_offset3, mean_offset3)
matrix4 = torch.einsum('ji,ki->ijk', mean_offset4, mean_offset4)
mean_matrix = torch.mean(torch.stack((matrix1, matrix2, matrix3, matrix4), dim=0), dim=0)
print(f"meanoffset1: {mean_offset1[:, 0]}")
print(f"matrix1: {matrix1[0, ...]}")
print(f"matrix1 shape: {matrix1.shape}")
print(f"mean_offset1 mean {mean_offset1.mean(axis=1)}, max {mean_offset1.max()}, min {mean_offset1.min()}")
fused_covariances = (0.25 * covariances[u, v, ...] + matrix1
                    + 0.25 * covariances[u, v+1, ...] + matrix2
                    + 0.25 * covariances[u+1, v, ...] + matrix3
                    + 0.25 * covariances[u+1, v+1, ...] + matrix4)
# fused_covariances = (0.25 * covariances[u, v, ...]
#                     + 0.25 * covariances[u, v+1, ...]
#                     + 0.25 * covariances[u+1, v, ...]
#                     + 0.25 * covariances[u+1, v+1, ...]) + mean_matrix*4
fused_opacities = (opacities[:, u, v] + opacities[:, u, v+1] + opacities[:, u+1, v] + opacities[:, u+1, v+1]) / 4.0
fused_sh = (spherical_harmonics[:, u, v] + spherical_harmonics[:, u, v+1] + spherical_harmonics[:, u+1, v] + spherical_harmonics[:, u+1, v+1]) / 4.0
fused_scales = (scales[:, u, v] + scales[:, u, v+1] + scales[:, u+1, v] + scales[:, u+1, v+1]) * scale_factor
quat1, quat2, quat3, quat4 = rotations[:, u, v], rotations[:, u, v+1], rotations[:, u+1, v], rotations[:, u+1, v+1]
quats = torch.stack((quat1, quat2, quat3, quat4), dim=1).permute(2, 0, 1)
print(f"quats shape: {quats.shape}")
print(f"quats: {quats}")
print(f"scales: {scales[:, u, v].max()}, {scales[:, u, v].min()}")
fused_rotations = average_quaternions(quats)
# fused_rotations = (quat1 + quat2 + quat3 + quat4) / 4.0

# fused_means = means[:, u, v]
# fused_opacities = opacities[:, u, v]
# fused_sh = spherical_harmonics[:, u, v]
# fused_scales = scales[:, u, v]
fused_rotations= rotations[:, u, v]
fused_rotations = einops.rearrange(fused_rotations, "c n -> n c")
print(f"fused_rotations shape: {fused_rotations.shape}")

print("Fused means shape:", fused_means.shape)

fused_means = einops.rearrange(fused_means, "c n -> n c")
fused_opacities = einops.rearrange(fused_opacities, "c n-> n c")
fused_sh = einops.rearrange(fused_sh, "c n -> n c")
# print(f"fused_menas {fused_means}")
fused_scales = einops.rearrange(fused_scales, "c n -> n c")
print(f"fused_scales shape: {fused_scales.shape}")

# fused_sh[:, 0] += 5.0

original_means = means[:, ~mask_upsampled]
original_opacities = opacities[:, ~mask_upsampled]
original_covariances = covariances[~mask_upsampled, ...]
original_sh = spherical_harmonics[:, ~mask_upsampled]
original_scales = scales[:, ~mask_upsampled]
original_rotations = rotations[:, ~mask_upsampled]

# original_means = einops.rearrange(means, "xyz h w -> xyz (h w)")
# original_opacities = einops.rearrange(opacities, "o h w -> o (h w)")
# original_covariances = einops.rearrange(covariances, "h w x y -> (h w) x y")
# original_sh = einops.rearrange(spherical_harmonics, "c h w -> c (h w)")

original_means = einops.rearrange(original_means, "c n -> n c")
original_opacities = einops.rearrange(original_opacities, "c n-> n c")
original_sh = einops.rearrange(original_sh, "c n -> n c")
original_scales = einops.rearrange(original_scales, "c n -> n c")
original_rotations = einops.rearrange(original_rotations, "c n -> n c")

reduced_means = torch.cat((fused_means, original_means), dim=0)
reduced_opacities = torch.cat((fused_opacities, original_opacities), dim=0)
reduced_sh = torch.cat((fused_sh, original_sh), dim=0)
reduced_covariances = torch.cat((fused_covariances, original_covariances), dim=0)
reduced_scales = torch.cat((fused_scales, original_scales), dim=0)
reduced_rotations = torch.cat((fused_rotations, original_rotations), dim=0)
num_gaussians = reduced_means.shape[0]
num_gaussians_original = Xii.shape[0]
# save_as_ply(pred1, pred1, recon_file)
print(f"SHii shape: {SHii.shape}")
reduced_sh = einops.rearrange(reduced_sh, "n c -> n c 1")
print(f"reduced_sh shape: {reduced_sh.shape}")
print(f"reduced_covariances shape: {reduced_covariances.shape}")
reduced_rotations, reduced_scales = covariance_to_quaternion_and_scale(reduced_covariances)

print(f"reduced_scales mean: {reduced_scales.mean()}, reduced_scales max: {reduced_scales.max()}, reduced_scales min: {reduced_scales.min()}")
# cov_test = geometry.build_covariance(Sii, Rii)
# Rii, Sii = covariance_to_quaternion_and_scale(cov_test)

if not reduced: 
    num_gaussians_original = Sii.shape[0]
    num_gaussians = Sii.shape[0]

input_imgs = f"_frame_{img1_idx}_{img2_idx}"
reduced_name = f"gaussians_reduced_th_{grad_threshold}_covf_{scale_factor}_n_{num_gaussians}" if reduced else f"gaussians_original_n_{num_gaussians_original}"
reduced_name += input_imgs
results_path = pathlib.Path(f"/home/curdinst/repos/MASt3R-SLAM/logs/{reduced_name}/")
results_path.mkdir(exist_ok=True, parents=True)
gaussians_file = results_path / f"gaussians.ply"
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
    # save_gaussian_new_ply(
    #     save_path=gaussians_file,
    #     M=reduced_means.cpu().numpy(),
    #     SH=reduced_sh.squeeze(-1).cpu().numpy(),
    #     O=reduced_opacities.cpu().numpy(),
    #     covariance=reduced_covariances,
    # )
    save_gaussian_new_ply(
        save_path=gaussians_file,
        M=reduced_means.cpu().numpy(),
        SH=reduced_sh.squeeze(-1).cpu().numpy(),
        O=reduced_opacities.cpu().numpy(),
        S=reduced_scales.cpu().numpy(),
        R=reduced_rotations.cpu().numpy(),
    )

print("predctions done")



# example_rot = torch.tensor([[0.3152946438, 0.5, 0.7, 0.1]], device=device)
# example_scale = torch.tensor([[1.0, 2.0, 3.0]], device=device)
# example_cov = geometry.build_covariance(example_scale, example_rot)
# ret_rot, ret_scale = covariance_to_quaternion_and_scale(example_cov)
# # example_cov = geometry.build_covariance(ret_scale.float(), ret_rot.float())
# # ret_rot, ret_scale = covariance_to_quaternion_and_scale(example_cov)
# # ret_rot, ret_scale = geometry.inverse_build_covariance_torch(example_cov)
# print(f"ret_rot: {ret_rot}")
# print(f"ret_scale: {ret_scale}")
# exit()


# reduced_rotations, reduced_scales = Rii, Sii
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
downsampling_factor = 0.5 if W == 256 else 1.0
downsampling_factor = 1.0
fx = K_frame[0, 0] * downsampling_factor
fy = K_frame[1, 1] * downsampling_factor
cx = K_frame[0, 2] * downsampling_factor
cy = K_frame[1, 2] * downsampling_factor
print(f"fx: {fx}, fy: {fy}, cx: {cx}, cy: {cy}")
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

print(f"Saved results in {results_path}")