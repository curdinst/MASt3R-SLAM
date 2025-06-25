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
    resize_img,
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
from mast3r_slam.utils import sh_utils


reduced = False
grad_threshold = 0.2
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

@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2
print(f"frame1 img shape: {frame1.img.shape}")
image1 = einops.rearrange(frame1.img, "b c h w ->(b c) h w")
image1 = image1*0.5 + 0.5
gt_img_rearranged = einops.rearrange(image1.cpu().detach().numpy(), "c h w -> h w c")
a,b = np.min(gt_img_rearranged), np.max(gt_img_rearranged)
plt.figure()
plt.imshow((gt_img_rearranged- a)/(b-a))
plt.title("Ground Truth Image")
plt.savefig(f"logs/gt_image1.png")
plt.close()

print(f"image1 max: {image1.max()}, min: {image1.min()}")
# plt.imsave("logs/feat1.png", image1)
# dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
# pred1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
# pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
# pred2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
# X, C, D, Q, S, R, SH, O, M = mast3r_asymmetric_inference(model=model, frame_i=frame1, frame_j=frame2)
frame_i = frame1
frame_j = frame2

frame_i.feat, frame_i.pos, _ = model._encode_image(
    frame_i.img, frame_i.img_true_shape
)
frame_j.feat, frame_j.pos, _ = model._encode_image(
    frame_j.img, frame_j.img_true_shape
)

feat1, feat2 = frame_i.feat, frame_j.feat

print(f"feat1 shape: {feat1.shape}, feat2 shape: {feat2.shape}")
pos1, pos2 = frame_i.pos, frame_j.pos
print(f"pos1 shape: {pos1.shape}, pos2 shape: {pos2.shape}")
print(pos1)
shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

feat1_2d = einops.rearrange(feat1, "b (h w) d -> b h w d", h=H//16, w=W//16)
feat2_2d = einops.rearrange(feat2, "b (h w) d -> b h w d", h=H//16, w=W//16)

dim_w = W//16
feat1_a = feat1_2d[:,::2,::2,:]
feat1_b = feat1_2d[:,1::2,::2,:]
feat1_c = feat1_2d[:,::2,1::2,:]
feat1_d = feat1_2d[:,1::2,1::2,:]

feat2_a = feat2_2d[:,::2,::2,:]
feat2_b = feat2_2d[:,1::2,::2,:]
feat2_c = feat2_2d[:,::2,1::2,:]
feat2_d = feat2_2d[:,1::2,1::2,:]

feat1_mean = (feat1_a + feat1_b + feat1_c + feat1_d) / 4.0
feat2_mean = (feat2_a + feat2_b + feat2_c + feat2_d) / 4.0
feat1_downsampled_1d = einops.rearrange(feat1_mean, "b h w d -> b (h w) d")
feat2_downsampled_1d = einops.rearrange(feat2_mean, "b h w d -> b (h w) d")
pos_downsampled = torch.stack(torch.meshgrid(torch.arange(9, device=device), torch.arange(16, device=device), indexing='ij'), dim=-1).reshape(-1, 2).unsqueeze(0)
print(f"feat_a shape: {feat1_a.shape}")
print(f"pos downsampled: {pos_downsampled}")
print(f"pos downsampled shape: {pos_downsampled.shape}")
print(f"shape1: {shape1}, shape2: {shape2}")
shape_downsampled = torch.tensor([[H//2, W//2]]).to(device=device)
print(f"shape_downsampled: {shape_downsampled}")

res11, res21 = decoder(model, feat1_downsampled_1d, feat2_downsampled_1d, pos_downsampled, pos_downsampled, shape_downsampled, shape_downsampled)
res = [res11, res21]
X, C, D, Q, S, R, SH, O, M  = zip(
    *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0], r["scales"][0], r["rotations"][0], r["sh"][0], r["opacities"][0], r["means"][0]) for r in res]
)
X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
S, R, SH, O, M = torch.stack(S), torch.stack(R), torch.stack(SH), torch.stack(O), torch.stack(M)
b, h, w = X.shape[:-1]
# 2 outputs per inference
b = b // 2

Xii, Xji = X[:b], X[b:]
Cii, Cji = C[:b], C[b:]
Dii, Dji = D[:b], D[b:]
Qii, Qji = Q[:b], Q[b:]

# How rest of system expects it
Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

Sii, Sji = einops.rearrange(S, "b h w c -> b (h w) c")
Rii, Rji = einops.rearrange(R, "b h w c -> b (h w) c")
SHii, SHji = einops.rearrange(SH, "b h w c d -> b (h w) c d")
Oii, Oji = einops.rearrange(O, "b h w c -> b (h w) c")
Mii, Mji = einops.rearrange(M, "b h w c -> b (h w) c")

# add frame colors to sh colors
new_sh1 = torch.zeros_like(SHii)
new_sh2 = torch.zeros_like(SHji)
frame_i_img = einops.rearrange(frame_i.img, "b c h w -> (b h) w c").cpu().detach().numpy()
frame_j_img = einops.rearrange(frame_j.img, "b c h w -> (b h) w c").cpu().detach().numpy()
# img1_downsamped = resize_img(frame_i_img, W//2)["img"].to(device=device)
img1_downsampled = F.avg_pool2d(image1.unsqueeze(0), kernel_size=2, stride=2).squeeze(0)

img2_downsamped = resize_img(frame_j_img, W//2)["img"].to(device=device)
new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(img1_downsampled, '(b c) h w -> b (h w) c', b=1))
new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(img2_downsamped/2.0+0.5, 'b c h w -> b (h w) c'))
SHii = SHii + new_sh1
SHji = SHji + new_sh2
H, W = H // 2, W // 2
# image1_rearranged = einops.rearrange(frame_i.img, "b c h w -> (b h) w c")
# image1 = resize_img(image1_rearranged.cpu().detach().numpy(), W)["img"].squeeze(0).to(device=device)
# image1 = einops.rearrange(img1_downsamped/2.0+0.5, 'b c h w -> c (b h) w')
image1 = img1_downsampled
print(f"image1 shape: {image1.shape}, image1 max: {image1.max()}, min: {image1.min()}")
# gt_img_rearranged = einops.rearrange(image1.cpu().detach().numpy(), "c h w -> h w c")
# a,b = np.min(gt_img_rearranged), np.max(gt_img_rearranged)
# plt.figure()
# plt.imshow((gt_img_rearranged- a)/(b-a))
# plt.title("Ground Truth Image")
# plt.savefig(f"logs/gt_image1.png")
# plt.close()
# idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, gaussian_params = mast3r_match_asymmetric(model=model, frame_i=frame1, frame_j=frame2)
# (Sii, Rii, SHii, Oii, Mii, Sji, Rji, SHji, Oji, Mji) = gaussian_params

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

mask_downsampled, mask_upsampled = get_mask(image1)
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

# mean1, mean2, mean3, mean4 = means[:, u, v], means[:, u, v+1], means[:, u+1, v], means[:, u+1, v+1]
# # Fit a plane over the four mean points
# # The plane equation is ax + by + cz + d = 0
# # We solve for [a, b, c, d] using the four points
# # Stack the four mean points into a matrix
# # points = torch.stack((mean1, mean2, mean3, mean4), dim=1)  # Shape: (3, 4)
# # # Add a row of ones for the homogeneous coordinates
# # points_h = torch.cat((points, torch.ones(1, 4, device=device)), dim=0)  # Shape: (4, 4)
# # # Perform SVD to find the null space of the matrix
# # _, _, V = torch.linalg.svd(points_h.T)
# # plane_coeffs = V[-1]  # The last row of V corresponds to the null space
# # # Normalize the plane coefficients
# # plane_coeffs /= torch.norm(plane_coeffs[:3])
# # # Extract the plane parameters
# # a, b, c, d = plane_coeffs

# # print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
# fused_means = (means[:, u, v] + means[:, u, v+1] + means[:, u+1, v] + means[:, u+1, v+1]) / 4.0
# fused_opacities = (opacities[:, u, v] + opacities[:, u, v+1] + opacities[:, u+1, v] + opacities[:, u+1, v+1]) / 4.0
# fused_covariances = (covariances[u, v, ...] + covariances[u, v+1, ...] + covariances[u+1, v, ...] + covariances[u+1, v+1, ...]) * scale_factor
# print(fused_covariances)
# fused_sh = (spherical_harmonics[:, u, v] + spherical_harmonics[:, u, v+1] + spherical_harmonics[:, u+1, v] + spherical_harmonics[:, u+1, v+1]) / 4.0
# fused_scales = (scales[:, u, v] + scales[:, u, v+1] + scales[:, u+1, v] + scales[:, u+1, v+1]) * scale_factor
# quat1, quat2, quat3, quat4 = rotations[:, u, v], rotations[:, u, v+1], rotations[:, u+1, v], rotations[:, u+1, v+1]
# quats = torch.stack((quat1, quat2, quat3, quat4), dim=1).permute(2, 0, 1)
# print(f"quats shape: {quats.shape}")
# print(f"quats: {quats}")
# print(f"scales: {scales[:, u, v].max()}, {scales[:, u, v].min()}")
# fused_rotations = average_quaternions(quats)
# # fused_rotations = (quat1 + quat2 + quat3 + quat4) / 4.0

# # fused_means = means[:, u, v]
# # fused_opacities = opacities[:, u, v]
# # fused_sh = spherical_harmonics[:, u, v]
# # fused_scales = scales[:, u, v]
# fused_rotations= rotations[:, u, v]
# fused_rotations = einops.rearrange(fused_rotations, "c n -> n c")
# print(f"fused_rotations shape: {fused_rotations.shape}")

# print("Fused means shape:", fused_means.shape)

# fused_means = einops.rearrange(fused_means, "c n -> n c")
# fused_opacities = einops.rearrange(fused_opacities, "c n-> n c")
# fused_sh = einops.rearrange(fused_sh, "c n -> n c")
# # print(f"fused_menas {fused_means}")
# fused_scales = einops.rearrange(fused_scales, "c n -> n c")
# print(f"fused_scales shape: {fused_scales.shape}")

# # fused_sh[:, 0] += 5.0

# original_means = means[:, ~mask_upsampled]
# original_opacities = opacities[:, ~mask_upsampled]
# original_covariances = covariances[~mask_upsampled, ...]
# original_sh = spherical_harmonics[:, ~mask_upsampled]
# original_scales = scales[:, ~mask_upsampled]
# original_rotations = rotations[:, ~mask_upsampled]

# # original_means = einops.rearrange(means, "xyz h w -> xyz (h w)")
# # original_opacities = einops.rearrange(opacities, "o h w -> o (h w)")
# # original_covariances = einops.rearrange(covariances, "h w x y -> (h w) x y")
# # original_sh = einops.rearrange(spherical_harmonics, "c h w -> c (h w)")

# original_means = einops.rearrange(original_means, "c n -> n c")
# original_opacities = einops.rearrange(original_opacities, "c n-> n c")
# original_sh = einops.rearrange(original_sh, "c n -> n c")
# original_scales = einops.rearrange(original_scales, "c n -> n c")
# original_rotations = einops.rearrange(original_rotations, "c n -> n c")

# reduced_means = torch.cat((fused_means, original_means), dim=0)
# reduced_opacities = torch.cat((fused_opacities, original_opacities), dim=0)
# reduced_sh = torch.cat((fused_sh, original_sh), dim=0)
# reduced_covariances = torch.cat((fused_covariances, original_covariances), dim=0)
# reduced_scales = torch.cat((fused_scales, original_scales), dim=0)
# reduced_rotations = torch.cat((fused_rotations, original_rotations), dim=0)
# num_gaussians = reduced_means.shape[0]
# num_gaussians_original = Xii.shape[0]
# # save_as_ply(pred1, pred1, recon_file)
# print(f"SHii shape: {SHii.shape}")
# reduced_sh = einops.rearrange(reduced_sh, "n c -> n c 1")
# print(f"reduced_sh shape: {reduced_sh.shape}")
# print(f"reduced_covariances shape: {reduced_covariances.shape}")
# reduced_rotations, reduced_scales = covariance_to_quaternion_and_scale(reduced_covariances)

# # cov_test = geometry.build_covariance(Sii, Rii)
# # Rii, Sii = covariance_to_quaternion_and_scale(cov_test)

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