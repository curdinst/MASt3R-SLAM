import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
import einops
import numpy as np
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray, quat_mult
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
import mast3r_slam.utils.geometry as geometry



def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                print("Refining pose with calibration")
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{keyframe.frame_id} {t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

def save_frame_poses(
            save_dir,
            filename,
            timestamps,
            frame_poses
            ):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    filepath = save_dir / filename
    with open(filepath, "w") as f:
        for key in frame_poses.keys():
            stamp = timestamps[key]
            pose = frame_poses[key]
            x, y, z, qx, qy, qz, qw, s = pose.reshape(-1)
            f.write(f"{key} {stamp} {x} {y} {z} {qx} {qy} {qz} {qw} {s}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    print(f"C_conf_threshold: {c_conf_threshold}")
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            print("USING CLAIB")
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)
    return pointclouds, colors

def save_gaussian_map(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    masks_dir = savedir / f"masks_{filename[:-4]}"
    masks_dir.mkdir(exist_ok=True, parents=True)
    scales, rotations, means, sh, opacities = [], [], [], [], []
    num_gaussians = 0
    keyframe_ids = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if keyframe.SH is None:
            print(f"Keyframe {keyframe.frame_id} has no SH, skipping.")
            continue
        keyframe_ids.append(keyframe.frame_id)
        # print(f"Keyframe {keyframe.frame_id} has SH, saving.")
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        means_new = pW
        sh_resized = einops.rearrange(keyframe.SH, "hw c d -> hw (c d)")
        sh_new = sh_resized.cpu().numpy()
        scales_new = (keyframe.T_WC.data[0,-1] * keyframe.scales).cpu().numpy()
        opacities_new = keyframe.opacities.cpu().numpy()
        w_rotations = quat_mult(keyframe.T_WC.data, keyframe.rotations).cpu().numpy()
        # w_rotations = keyframe.rotations.cpu().numpy()
        rotations_new = w_rotations
        print(f"keyframe.offsets mean: {keyframe.offsets.mean()}, min {keyframe.offsets.min()}, max {keyframe.offsets.max()}")
        w_means = keyframe.T_WC.act(keyframe.X_canon + keyframe.offsets).cpu().numpy()

        print(f"shape of kf conf: {keyframe.get_average_conf().cpu().numpy().astype(np.float32).shape}")
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        
        valid_tensor = torch.tensor(valid, dtype=torch.bool)
        valid_tensor = valid_tensor & keyframe.gaussian_mask.clone().detach().to(device="cpu")
        valid = valid_tensor.cpu().numpy()
        # torch.save(valid_tensor, masks_dir / f"{keyframe.frame_id}.pt")
        rotations.append(rotations_new[valid])
        scales.append(scales_new[valid])
        means.append(means_new[valid])
        sh.append(sh_new[valid])
        opacities.append(opacities_new[valid])
        print(f"Valid points: {valid.sum()/len(valid)}")
        num_gaussians += rotations_new[valid].shape[0]
        print(f"num gaussians: {rotations_new[valid].shape[0]}")
        print(f"keyframes: {keyframe_ids}")
        # rotations.append(rotations_new)
        # scales.append(scales_new)
        # means.append(means_new)
        # sh.append(sh_new)
        # opacities.append(opacities_new)
        # num_gaussians += rotations_new.shape[0]


    # for i  in range(len(pointclouds)):
    #     pcd = pointclouds[i]
    #     mean = means[i]
    #     diff = np.abs(pcd -mean)
    #     diff_mean = diff.mean(axis=0)
    #     print
    #     print(f"diff mean: {diff_mean}, diff min : {diff.min(axis=0)}, diff max: {diff.max(axis=0)}")
    #     torch.save(pcd, masks_dir / f"{keyframe.frame_id}.pt") 
    if len(sh) < 2:
        print("Not enough keyframes with SH, skipping saving.")
        return
    print(f"length gaussian array: {len(sh)}")
    print(f"Total number of Gaussians: {num_gaussians}")
    scales = np.concatenate(scales, axis=0)
    rotations = np.concatenate(rotations, axis=0)
    means = np.concatenate(means, axis=0)
    sh = np.concatenate(sh, axis=0)
    opacities = np.concatenate(opacities, axis=0)
    # save_gaussian_new_ply(
    #     savedir / filename,
    #     scales,
    #     rotations,
    #     pointclouds,
    #     sh,
    #     opacities
    # )
    
    save_gaussian_new_ply(
        savedir / filename,
        scales,
        rotations,
        means,
        sh,
        opacities
    )


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)

def save_as_ply(pred1, pred2, save_path):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale

    pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
    pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])
    # Collect the Gaussian parameters
    means = torch.stack([pred1["means"], pred2["means"]], dim=1)
    covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

    # Rearrange the tensors to the correct shape
    means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
    harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()

    # Convert the covariance matrices to quaternions and scales
    rotations, scales = covariance_to_quaternion_and_scale(covariances)

    # Construct the attributes
    rest = np.zeros_like(means)
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)
    print("Saved PLY file to", save_path)

def save_gaussian_new_ply(save_path, S, R, M, SH, O):
    """Save the 3D Gaussians as a point cloud in the PLY format.
    Adapted loosely from PixelSplat"""

    def construct_list_of_attributes(num_rest: int) -> list[str]:
        '''Construct a list of attributes for the PLY file format. This
        corresponds to the attributes used by online readers, such as
        https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
        attributes = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(3):
            attributes.append(f"f_dc_{i}")
        for i in range(num_rest):
            attributes.append(f"f_rest_{i}")
        attributes.append("opacity")
        for i in range(3):
            attributes.append(f"scale_{i}")
        for i in range(4):
            attributes.append(f"rot_{i}")
        return attributes

    def covariance_to_quaternion_and_scale(covariance):
        '''Convert the covariance matrix to a four dimensional quaternion and
        a three dimensional scale vector'''

        # Perform singular value decomposition
        U, S, V = torch.linalg.svd(covariance)

        # The scale factors are the square roots of the eigenvalues
        scale = torch.sqrt(S)
        scale = scale.detach().cpu().numpy()

        # The rotation matrix is U*Vt
        rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
        rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

        # Use scipy to convert the rotation matrix to a quaternion
        rotation = Rotation.from_matrix(rotation_matrix_np)
        quaternion = rotation.as_quat()

        return quaternion, scale

    # Collect the Gaussian parameters
    # means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    # covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    # harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    # opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)

    # print("S.shape", S.shape)
    # print("R.shape", R.shape)
    # print("M.shape", M.shape)
    # print("SH.shape", SH.shape)
    # print("O.shape", O.shape)

    # means = M.detach().cpu().numpy()
    # # covariances = C
    # harmonics = SH[..., 0].detach().cpu().numpy()
    # opacities = O.detach().cpu().numpy()

    means = M
    harmonics = SH
    opacities = O
    rotations = R
    scales = S

    # Rearrange the tensors to the correct shape
    # means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    # # covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
    # harmonics = einops.rearrange(harmonics, "hw c d-> hw (c d)").detach().cpu().numpy()
    # print("harmonics.shape", harmonics.shape)
    # opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()

    # Convert the covariance matrices to quaternions and scales
    # rotations, scales = covariance_to_quaternion_and_scale(covariances)
    # rotations = R.detach().cpu().numpy()
    # scales = S.detach().cpu().numpy()
    # Construct the attributes
    rest = np.zeros_like(means)
    
    attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    # Save the point cloud
    point_cloud = PlyElement.describe(elements, "vertex")
    scene = PlyData([point_cloud])
    scene.write(save_path)
    print("Saved PLY file to", save_path)
