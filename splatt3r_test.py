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



# Load images

load_config("config/base.yaml")

device = "cuda"
# model = load_mast3r(device=device, path="checkpoints/MASt3R_gaussians_v1.pth")
# # model = load_mast3r(device=device)
# model.share_memory()
# print("loaded model")

dataset_path = "datasets/tum/rgbd_dataset_freiburg1_room/"
dataset_path = "datasets/replica/office2/"
dataset = load_dataset(dataset_path)
H, W = dataset.get_img_shape()[0]
img_size = (H, W)
print("Image size:", img_size)

img1_idx, img2_idx = 0, 30
timestamp1, img1 = dataset[img1_idx]
timestamp2, img2 = dataset[img2_idx]

image1_save_path = "logs/image" + str(img1_idx) + ".pt"
image2_save_path = "logs/image" + str(img2_idx) + ".pt"


frame1 = create_frame(img1_idx, img1, T_WC=None)
frame2 = create_frame(img2_idx, img2, T_WC=None)

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
    grad_threshold = 0.1

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
    mask_upsampled = F.interpolate(mask_downsampled.float(), size=(H, W), mode="nearest").squeeze(0).squeeze(0)
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
# # X, C, D, Q, S, R, SH, O, M, res11, res21 = mast3r_asymmetric_inference(model=model, frame_i=frame1, frame_j=frame2)

save_dir = pathlib.Path("logs")
save_dir.mkdir(exist_ok=True, parents=True)
filename = "gaussians_" + str(img1_idx) + "_" + str(img2_idx) + ".ply"
recon_file = save_dir / filename
if recon_file.exists():
    recon_file.unlink()

fake_means_1 = torch.zeros((1, 3, H, W)).to(device)

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
val0 = fake_means_1[0, :, u, v]
val1 = fake_means_1[0, :, u, v+1]
val2 = fake_means_1[0, :, u+1, v]
val3 = fake_means_1[0, :, u+1, v+1]

fused_means = (val0 + val1 + val2 + val3) / 4
print("Fused means shape:", fused_means.shape)

# save_as_ply(pred1, pred1, recon_file)

print("predctions done")

