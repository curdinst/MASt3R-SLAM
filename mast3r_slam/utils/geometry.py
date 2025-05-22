import einops
import torch

# --- Intrinsics Transformations ---

def normalize_intrinsics(intrinsics, image_shape):
    '''Normalize an intrinsics matrix given the image shape'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] /= image_shape[1]
    intrinsics[..., 1, :] /= image_shape[0]
    return intrinsics


def unnormalize_intrinsics(intrinsics, image_shape):
    '''Unnormalize an intrinsics matrix given the image shape'''
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, :] *= image_shape[1]
    intrinsics[..., 1, :] *= image_shape[0]
    return intrinsics


# --- Quaternions, Rotations and Scales ---

def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    '''
    Convert the 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    '''

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(scale, rotation_xyzw):
    '''Build the 3x3 covariance matrix from the three dimensional scale and the
    four dimension quaternion'''
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ einops.rearrange(scale, "... i j -> ... j i")
        @ einops.rearrange(rotation, "... i j -> ... j i")
    )

def quaternion_to_matrix(quaternions):
    """
    Converts a batch of quaternions (x, y, z, w) to rotation matrices.
    Args:
        quaternions (torch.Tensor): A tensor of shape (..., 4) representing quaternions (x, y, z, w).
    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing rotation matrices.
    """
    x, y, z, w = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]

    xx, yy, zz = 2 * x * x, 2 * y * y, 2 * z * z
    xy, xz, yz = 2 * x * y, 2 * x * z, 2 * y * z
    wx, wy, wz = 2 * w * x, 2 * w * y, 2 * w * z

    matrix = torch.empty(quaternions.shape[:-1] + (3, 3), dtype=quaternions.dtype, device=quaternions.device)

    matrix[..., 0, 0] = 1 - yy - zz
    matrix[..., 0, 1] = xy - wz
    matrix[..., 0, 2] = xz + wy

    matrix[..., 1, 0] = xy + wz
    matrix[..., 1, 1] = 1 - xx - zz
    matrix[..., 1, 2] = yz - wx

    matrix[..., 2, 0] = xz - wy
    matrix[..., 2, 1] = yz + wx
    matrix[..., 2, 2] = 1 - xx - yy

    return matrix

def matrix_to_quaternion(matrix):
    """
    Converts a batch of rotation matrices to quaternions (x, y, z, w).
    Based on "Quaternions from Rotation Matrix and Translation Vector" by Mike Johnson.
    Args:
        matrix (torch.Tensor): A tensor of shape (..., 3, 3) representing rotation matrices.
    Returns:
        torch.Tensor: A tensor of shape (..., 4) representing quaternions (x, y, z, w).
    """
    batch_dim = matrix.shape[:-2]
    quaternions = torch.empty(batch_dim + (4,), dtype=matrix.dtype, device=matrix.device)

    t = torch.trace_matrix(matrix) # This is not directly available for batch
    # A workaround for batch trace:
    trace_val = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]

    # Case 1: trace_val > 0
    mask1 = trace_val > 0
    s1 = torch.sqrt(trace_val[mask1] + 1.0) * 2
    quaternions[mask1, 3] = 0.25 * s1
    quaternions[mask1, 0] = (matrix[mask1, 2, 1] - matrix[mask1, 1, 2]) / s1
    quaternions[mask1, 1] = (matrix[mask1, 0, 2] - matrix[mask1, 2, 0]) / s1
    quaternions[mask1, 2] = (matrix[mask1, 1, 0] - matrix[mask1, 0, 1]) / s1

    # Case 2: matrix[0,0] is the largest diagonal element
    mask2 = ~mask1 & (matrix[..., 0, 0] > matrix[..., 1, 1]) & (matrix[..., 0, 0] > matrix[..., 2, 2])
    s2 = torch.sqrt(1.0 + matrix[mask2, 0, 0] - matrix[mask2, 1, 1] - matrix[mask2, 2, 2]) * 2
    quaternions[mask2, 3] = (matrix[mask2, 2, 1] - matrix[mask2, 1, 2]) / s2
    quaternions[mask2, 0] = 0.25 * s2
    quaternions[mask2, 1] = (matrix[mask2, 0, 1] + matrix[mask2, 1, 0]) / s2
    quaternions[mask2, 2] = (matrix[mask2, 0, 2] + matrix[mask2, 2, 0]) / s2

    # Case 3: matrix[1,1] is the largest diagonal element
    mask3 = ~mask1 & ~mask2 & (matrix[..., 1, 1] > matrix[..., 2, 2])
    s3 = torch.sqrt(1.0 + matrix[mask3, 1, 1] - matrix[mask3, 0, 0] - matrix[mask3, 2, 2]) * 2
    quaternions[mask3, 3] = (matrix[mask3, 0, 2] - matrix[mask3, 2, 0]) / s3
    quaternions[mask3, 0] = (matrix[mask3, 0, 1] + matrix[mask3, 1, 0]) / s3
    quaternions[mask3, 1] = 0.25 * s3
    quaternions[mask3, 2] = (matrix[mask3, 1, 2] + matrix[mask3, 2, 1]) / s3

    # Case 4: matrix[2,2] is the largest diagonal element
    mask4 = ~mask1 & ~mask2 & ~mask3 # This should cover the remaining cases
    s4 = torch.sqrt(1.0 + matrix[mask4, 2, 2] - matrix[mask4, 0, 0] - matrix[mask4, 1, 1]) * 2
    quaternions[mask4, 3] = (matrix[mask4, 1, 0] - matrix[mask4, 0, 1]) / s4
    quaternions[mask4, 0] = (matrix[mask4, 0, 2] + matrix[mask4, 2, 0]) / s4
    quaternions[mask4, 1] = (matrix[mask4, 1, 2] + matrix[mask4, 2, 1]) / s4
    quaternions[mask4, 2] = 0.25 * s4

    # Normalize quaternions
    norm = torch.linalg.norm(quaternions, dim=-1, keepdim=True)
    quaternions = quaternions / norm

    return quaternions

def inverse_build_covariance_torch(covariance_matrix):
    """
    Inverse function for build_covariance, using PyTorch.
    Recovers the scale and rotation quaternion from a 3x3 covariance matrix.

    Args:
        covariance_matrix (torch.Tensor): A 3x3 symmetric positive semi-definite covariance matrix.
                                          Can be a batch of matrices (..., 3, 3).

    Returns:
        tuple: A tuple containing:
            - scale (torch.Tensor): A 3-dimensional tensor representing the original scale.
                                    Shape (..., 3).
            - rotation_xyzw (torch.Tensor): A 4-dimensional tensor representing the original quaternion (x, y, z, w).
                                            Shape (..., 4).
    """
    # 1. Perform Eigen Decomposition
    # torch.linalg.eigh returns eigenvalues in ascending order and corresponding eigenvectors.
    # For a symmetric matrix, eigvals returns real eigenvalues and eigvecs returns orthogonal eigenvectors.
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Ensure eigenvalues are non-negative (due to potential numerical precision issues)
    # Clamp to a small positive value to avoid issues with sqrt(negative number)
    eigenvalues = torch.clamp(eigenvalues, min=1e-9)

    # 2. Extract Scale
    # The original scales are the square root of the eigenvalues.
    scale = torch.sqrt(eigenvalues)

    # 3. Extract Rotation (Quaternion)
    # The eigenvectors matrix is the rotation matrix.
    # We need to handle potential reflections (determinant -1).
    # Check the determinant for the last two dimensions (the 3x3 matrix).
    det_eigenvectors = torch.linalg.det(eigenvectors)

    # If the determinant is -1, flip the sign of one of the eigenvectors to make it a proper rotation.
    # It's arbitrary which column to flip, typically the last one.
    # Need to handle batch dimensions.
    # Create a mask for matrices with determinant < 0.
    mask_reflection = det_eigenvectors < 0

    # Apply the flip only to the matrices in the batch that are reflections.
    # We need to use `where` or direct indexing carefully.
    if mask_reflection.any():
        # Create a new eigenvectors tensor to modify conditionally
        eigenvectors_corrected = eigenvectors.clone()
        eigenvectors_corrected[mask_reflection, :, 0] *= -1 # Flip the first column for those matrices

        # Ensure the corrected matrix is still orthonormal if needed (e.g., QR decomposition)
        # In practice, with eigh, this one flip should be enough for proper rotation.
        # It maintains orthogonality and changes determinant from -1 to 1.

        rotation_matrix = eigenvectors_corrected
    else:
        rotation_matrix = eigenvectors

    # Convert the rotation matrix to a quaternion
    rotation_xyzw = matrix_to_quaternion(rotation_matrix)

    return scale, rotation_xyzw

# --- Projections ---

def homogenize_points(points):
    """Append a '1' along the final dimension of the tensor (i.e. convert xyz->xyz1)"""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]


def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    camera_space_points = torch.einsum('b v i j , h w j -> b v h w i', intrinsics.inverse(), pixel_space_points)
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum('b v i j , b v h w j -> b v h w i', c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum('b u i j , b v u h w j -> b v u h w i', intrinsics, camera_space_points)
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum('b u i j , b v h w j -> b v u h w i', c2w.inverse(), world_space_points)
    return camera_space_points[..., :3]


def unproject_depth(depth, intrinsics, c2w):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """

    # Compute indices of pixels
    h, w = depth.shape[-3], depth.shape[-2]
    x_grid, y_grid = torch.meshgrid(
        torch.arange(w, device=depth.device, dtype=torch.float32),
        torch.arange(h, device=depth.device, dtype=torch.float32),
        indexing='xy'
    )  # (h, w), (h, w)

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(pixel_space_points, depth, intrinsics)  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points
