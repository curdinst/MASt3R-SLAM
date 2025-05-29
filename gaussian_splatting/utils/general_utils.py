#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import random
import sys
from datetime import datetime

import numpy as np
import torch


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def PILtoTorch2(pil_image):
    # resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    # def helper(step):
    #     if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
    #         # Disable this parameter
    #         return 0.0
    #     if lr_delay_steps > 0:
    #         # A kind of reverse cosine decay.
    #         delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
    #             0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
    #         )
    #     else:
    #         delay_rate = 1.0
    #     t = np.clip(step / max_steps, 0, 1)
    #     log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    #     return delay_rate * log_lerp

    return helper
    # return helper(lr_init=lr_init, lr_final=lr_final,
    #               lr_delay_steps=lr_delay_steps, lr_delay_mult=lr_delay_mult, max_steps=max_steps)


def helper(
    step, lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
        # Disable this parameter
        return 0.0
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(
                        x.replace(
                            "\n",
                            " [{}]\n".format(
                                str(datetime.now().strftime("%d/%m %H:%M:%S"))
                            ),
                        )
                    )
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def slerp(q1: torch.Tensor, q2: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Performs Spherical Linear Interpolation (Slerp) between two batches of quaternions.
    This function is vectorized for (N, 4) tensors.

    Args:
        q1 (torch.Tensor): Start quaternions, shape (N, 4).
        q2 (torch.Tensor): End quaternions, shape (N, 4).
        t (torch.Tensor or float): Interpolation parameter (0.0 <= t <= 1.0).
                                    Can be a float or a (N, 1) tensor for per-row weights.

    Returns:
        torch.Tensor: Interpolated quaternions, shape (N, 4).
    """
    if not isinstance(t, torch.Tensor):
        # If t is a float, convert it to a tensor for broadcasting
        t = torch.tensor(t, dtype=q1.dtype, device=q1.device).reshape(1, 1)

    if not (0.0 <= t).all() and (t <= 1.0).all():
        raise ValueError("Interpolation parameter 't' must be between 0.0 and 1.0.")

    # Ensure quaternions are normalized
    q1_norm = torch.nn.functional.normalize(q1, p=2, dim=-1)
    q2_norm = torch.nn.functional.normalize(q2, p=2, dim=-1)

    # Compute dot product (cosine of the angle between quaternions)
    # Resulting shape: (N, 1)
    dot_product = torch.sum(q1_norm * q2_norm, dim=-1, keepdim=True)

    # Adjust sign if quaternions point in opposite directions to take the shortest path
    q2_aligned = torch.where(dot_product < 0, -q2_norm, q2_norm)
    dot_product = torch.abs(dot_product) # Use the absolute dot product for angle calculation

    # Clamp dot product to avoid numerical issues (e.g., beyond 1.0 due to float precision)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    # Compute the angle between the quaternions
    theta_0 = torch.acos(dot_product) # Shape (N, 1)

    # Handle the case where quaternions are very close (theta_0 near 0)
    threshold = 1e-6
    mask_close = (theta_0 < threshold) # Shape (N, 1)

    # For quaternions that are very close, use linear interpolation
    q_slerped_close = torch.nn.functional.normalize(
        (1 - t) * q1_norm + t * q2_aligned, p=2, dim=-1
    )

    # For other quaternions, use the standard Slerp formula
    sin_theta_0 = torch.sin(theta_0)
    sin_theta_0_safe = torch.where(sin_theta_0 == 0, torch.ones_like(sin_theta_0) * 1e-12, sin_theta_0)


    term1 = torch.sin((1 - t) * theta_0) / sin_theta_0_safe
    term2 = torch.sin(t * theta_0) / sin_theta_0_safe

    q_slerped_general = term1 * q1_norm + term2 * q2_aligned

    # Combine results based on the mask
    q_slerped = torch.where(mask_close, q_slerped_close, q_slerped_general)

    return q_slerped
