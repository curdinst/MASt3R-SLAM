import torch
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
import cv2



# Load images

load_config("config/base.yaml")

# device = "cuda"
# model = load_mast3r(device=device, path="checkpoints/MASt3R_gaussians_v1.pth")
# # model = load_mast3r(device=device)
# model.share_memory()
# print("loaded model")

# dataset_path = "datasets/tum/rgbd_dataset_freiburg1_room/"
dataset_path = "datasets/replica/office2/"
dataset = load_dataset(dataset_path)
h, w = dataset.get_img_shape()[0]
img_size = (h, w)

idx_f2k_path = "/home/curdin/master_thesis/outputs/25_05_26_point_correspondances/2025-05-26_11-56_replica_office2_0_it_window11/idx_f2k_71-2-0.pt"

idx_f2k = torch.load(idx_f2k_path)

valid_match_k_path = "/home/curdin/master_thesis/outputs/25_05_26_point_correspondances/2025-05-26_11-56_replica_office2_0_it_window11/valid_match_k_71-2-0.pt"
valid_match_k = torch.load(valid_match_k_path)

valid_kf_path = "/home/curdin/master_thesis/outputs/25_05_26_point_correspondances/2025-05-26_11-56_replica_office2_0_it_window11/valid_kf_71-2-0.pt"
valid_kf = torch.load(valid_kf_path) # valid_match_k & Qk

print(f"valid_match_k shape: {valid_match_k.shape} sum {valid_match_k.sum()}")
print(f"idx_f2k shape: {idx_f2k.shape}")

img1_idx, img2_idx = 0, 71
timestamp1, img1 = dataset[img1_idx]
timestamp2, img2 = dataset[img2_idx]

idx_f2k[~valid_match_k.squeeze(-1)] = 0
idx_f2k_2d = einops.rearrange(idx_f2k, "(h w) -> h w", h=h, w=w).unsqueeze(-1)
valid_kf_2d = einops.rearrange(valid_kf.squeeze(-1), "(h w) -> h w", h=h, w=w)
valid_match_k = einops.rearrange(valid_match_k.squeeze(-1), "(h w) -> h w", h=h, w=w)
print(f"idx_f2k_2d shape: {idx_f2k_2d.shape}")
print(f"idx_f2k_2d dtype: {idx_f2k_2d.max()} {idx_f2k_2d.min()} {idx_f2k_2d.dtype}")
mask = idx_f2k_2d[:, :, 0] > 0
idx_f2k_2d_u = idx_f2k_2d[:, :, 0] // h
idx_f2k_2d_v = idx_f2k_2d[:, :, 0] % h
idx_f2k_2d = torch.stack([idx_f2k_2d_u, idx_f2k_2d_v], dim=-1)
print(f"idx_f2k_2d shape: {idx_f2k_2d.shape}")

print(f"mask sum {mask.sum()}")
# Plot the mask as an image
plt.figure(figsize=(8, 6))
plt.imshow(valid_match_k.cpu().numpy(), cmap='gray', vmin=0, vmax=1) #black = 0, white = 1
plt.title("Correspondence Mask")
plt.axis('off')
plt.show()
exit()

# Downsample img1 to (h, w) if needed
if isinstance(img1, torch.Tensor):
    if img1.shape[-2:] != (h, w):
        img1 = torch.nn.functional.interpolate(img1.unsqueeze(0).float(), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).type_as(img1)
        img2 = torch.nn.functional.interpolate(img2.unsqueeze(0).float(), size=(h, w), mode='bilinear', align_corners=False).squeeze(0).type_as(img2)
elif isinstance(img1, np.ndarray):
    if img1.shape[0] != h or img1.shape[1] != w:
        img1 = cv2.resize(img1, (w, h), interpolation=cv2.INTER_LINEAR)
        img2 = cv2.resize(img2, (w, h), interpolation=cv2.INTER_LINEAR)
# Convert images to numpy arrays if needed
if isinstance(img1, torch.Tensor):
    img1_np = img1.permute(1, 2, 0).cpu().numpy() if img1.ndim == 3 else img1.cpu().numpy()
else:
    img1_np = img1
if isinstance(img2, torch.Tensor):
    img2_np = img2.permute(1, 2, 0).cpu().numpy() if img2.ndim == 3 else img2.cpu().numpy()
else:
    img2_np = img2

# Ensure images are uint8 for plotting
if img1_np.dtype != np.uint8:
    img1_np = (img1_np * 255).clip(0, 255).astype(np.uint8)
if img2_np.dtype != np.uint8:
    img2_np = (img2_np * 255).clip(0, 255).astype(np.uint8)

# Create a canvas with both images side by side
canvas = np.zeros((h, w * 2, img1_np.shape[2]), dtype=np.uint8)
canvas[:, :w] = img1_np
canvas[:, w:] = img2_np

# Plot correspondences
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(canvas)
num_points = 200  # limit number of lines for clarity
ys, xs = torch.where(idx_f2k_2d[..., 0] >= 0)
indices = torch.randperm(len(xs))[:num_points]
for i in indices:
    y, x = ys[i].item(), xs[i].item()
    u2, v2 = idx_f2k_2d[y, x].tolist()
    # Draw line from (x, y) in img1 to (u2 + w, v2) in img2
    color = plt.cm.jet(i / num_points)
    ax.plot([x, u2 + w], [y, v2], color=color, linewidth=0.5, alpha=0.7)
ax.axis('off')
plt.tight_layout()
plt.show()


