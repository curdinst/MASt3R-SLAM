import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching
from mast3r_slam.utils import sh_utils
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement
from PIL import Image
import os
import mast3r_slam.utils.geometry as geometry

# def save_as_ply(pred1, pred2 = None, save_path=None):
#     """Save the 3D Gaussians as a point cloud in the PLY format.
#     Adapted loosely from PixelSplat"""
#     pred2_exists = pred2 is not None

#     def construct_list_of_attributes(num_rest: int) -> list[str]:
#         '''Construct a list of attributes for the PLY file format. This
#         corresponds to the attributes used by online readers, such as
#         https://niujinshuchong.github.io/mip-splatting-demo/index.html'''
#         attributes = ["x", "y", "z", "nx", "ny", "nz"]
#         for i in range(3):
#             attributes.append(f"f_dc_{i}")
#         for i in range(num_rest):
#             attributes.append(f"f_rest_{i}")
#         attributes.append("opacity")
#         for i in range(3):
#             attributes.append(f"scale_{i}")
#         for i in range(4):
#             attributes.append(f"rot_{i}")
#         return attributes

#     def covariance_to_quaternion_and_scale(covariance):
#         '''Convert the covariance matrix to a four dimensional quaternion and
#         a three dimensional scale vector'''
#         print(f"covariance.shape {covariance.shape}")
#         # Perform singular value decomposition
#         U, S, V = torch.linalg.svd(covariance)

#         # The scale factors are the square roots of the eigenvalues
#         scale = torch.sqrt(S)
#         scale = scale.detach().cpu().numpy()

#         # The rotation matrix is U*Vt
#         rotation_matrix = torch.bmm(U, V.transpose(-2, -1))
#         rotation_matrix_np = rotation_matrix.detach().cpu().numpy()

#         # Use scipy to convert the rotation matrix to a quaternion
#         rotation = Rotation.from_matrix(rotation_matrix_np)
#         quaternion = rotation.as_quat()

#         return quaternion, scale

#     pred1['covariances'] = geometry.build_covariance(pred1['scales'], pred1['rotations'])
#     if pred2_exists: pred2['covariances'] = geometry.build_covariance(pred2['scales'], pred2['rotations'])
#     # Collect the Gaussian parameters
#     if pred2_exists: 
#         means = torch.stack([pred1["means"], pred2["means"]], dim=1)
#         covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
#         harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
#         opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
#     else:
#         means = pred1["means"]
#         covariances = pred1["covariances"]
#         harmonics = pred1["sh"]
#         opacities = pred1["opacities"]
#     # Rearrange the tensors to the correct shape
#     # means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
#     # covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
#     # harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
#     # opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
#     means = means[0]
#     covariances = covariances[0]
#     harmonics = harmonics[0]
#     opacities = opacities[0]

#     # Convert the covariance matrices to quaternions and scales
#     rotations, scales = covariance_to_quaternion_and_scale(covariances)


#     means = means.detach().cpu().numpy()
#     covariances = covariances.detach().cpu().numpy()
#     harmonics = harmonics.detach().cpu().numpy()
#     opacities = opacities.detach().cpu().numpy()
#     # rotations = rotations.detach().cpu().numpy()
#     # scales = scales.detach().cpu().numpy()

#     # Construct the attributes
#     rest = np.zeros_like(means)
#     attributes = np.concatenate((means, rest, harmonics, opacities, np.log(scales), rotations), axis=-1)
#     dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
#     elements = np.empty(attributes.shape[0], dtype=dtype_full)
#     elements[:] = list(map(tuple, attributes))

#     # Save the point cloud
#     point_cloud = PlyElement.describe(elements, "vertex")
#     scene = PlyData([point_cloud])
#     scene.write(save_path)
#     print("Saved PLY file to", save_path)

def save_as_ply(pred1, pred2, save_path, as_list=False, grad_coarseness=False):
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
    # if "means_in_other_view" in pred2.keys():
    #     means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    # else:
    #     means = torch.stack([pred1["means"], pred2["means"]], dim=1)
    # covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    # harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
    # opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
    if True:
        means = torch.cat((pred1["means"], pred2["means"]), dim=1).unsqueeze(0)
        covariances = torch.cat((pred1["covariances"], pred2["covariances"]), dim=1).unsqueeze(0)
        harmonics = torch.cat((pred1["sh"], pred2["sh"]), dim=1).unsqueeze(0).squeeze(-1)
        opacities = torch.cat((pred1["opacities"], pred2["opacities"]), dim=1).unsqueeze(0)
    else:
        if "means_in_other_view" in pred2.keys():
            means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        else:
            means = torch.stack([pred1["means"], pred2["means"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)[..., 0]  # Only use the first harmonic
        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
        
    means = pred1["means"].unsqueeze(0)  # Remove the batch dimension
    covariances = pred1["covariances"].unsqueeze(0)  # Remove the batch dimension
    harmonics = pred1["sh"].unsqueeze(0)[..., 0]  # Only use the first harmonic
    opacities = pred1["opacities"].unsqueeze(0)  # Remove the batch dimension

    if True:
        means = einops.rearrange(means[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v n i j -> (v n) i j")
        harmonics = einops.rearrange(harmonics[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view n xyz -> (view n) xyz").detach().cpu().numpy()
    elif not as_list:
        # Rearrange the tensors to the correct shape
        means = einops.rearrange(means[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v h w i j -> (v h w) i j")
        harmonics = einops.rearrange(harmonics[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view h w xyz -> (view h w) xyz").detach().cpu().numpy()
    else:
         # Rearrange the tensors to the correct shape
        means = einops.rearrange(means[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
        covariances = einops.rearrange(covariances[0], "v hw i j -> (v hw) i j")
        harmonics = einops.rearrange(harmonics[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
        opacities = einops.rearrange(opacities[0], "view hw xyz -> (view hw) xyz").detach().cpu().numpy()
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

def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C and Q: (...xHxW)
        # X and D: (...xHxWxF)
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )
    
    # if frame_i.feat is None or frame_j.feat is None:
    #     print("Encoding both images")
    #     (frame_i.img_true_shape, frame_j.img_true_shape), (frame_i.feat, frame_j.feat), (frame_i.pos, frame_j.pos) = model.(frame_i.img, frame_j.img)

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])

        (res11_512, res11_256, res11_128, res11_coarseness_pred) = res11
        (res21_512, res21_256, res21_128, res21_coarseness_pred) = res21
        (res22_512, res22_256, res22_128, res22_coarseness_pred) = res22
        (res12_512, res12_256, res12_128, res12_coarseness_pred) = res12
        res = [res11_512, res21_512, res22_512, res12_512]

        # res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

        # P(T|indizien) = P(T, indizien) / P(indizien)

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)

    (res11_512, res11_256, res11_128, res11_coarseness_pred) = res11
    (res21_512, res21_256, res21_128, res21_coarseness_pred) = res21

    res = [res11_512, res21_512]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    # print("frame_i.img.shape", frame_i.img.shape)
    # print("frame_j.img.shape", frame_j.img.shape)
    # print(f"frame_i.img_true_shape", frame_i.img_true_shape)
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )
    # if frame_i.feat is None or frame_j.feat is None:
    #     # print(f"Encoding both images, frame_i.feat = {frame_i.feat.shape if frame_i.feat else None}, frame_j.feat = {frame_j.feat.shape if frame_j.feat else None}")
    #     # print(f"frame_i.img.shape = {frame_i.img.shape}, frame_j.img.shape = {frame_j.img.shape}")
    #     frame_j_img_reshaped = einops.rearrange(frame_j.img, "(b c) h w -> b c h w ", b=1)
    #     (frame_i.img_true_shape, frame_j.img_true_shape), (frame_i.feat, frame_j.feat), (frame_i.pos, frame_j.pos) = model._encode_symmetrized(frame_i.img, frame_j_img_reshaped, frame_i.img_true_shape, frame_j.img_true_shape)

    feat1, feat2 = frame_i.feat, frame_j.feat
    # print(f"feat1 shape: {feat1.shape}, feat2 shape: {feat2.shape}")
    pos1, pos2 = frame_i.pos, frame_j.pos
    # print(f"pos1 shape: {pos1.shape}, pos2 shape: {pos2.shape}")
    # print(pos1)
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    (res11_512, res11_256, res11_128, res11_coarseness_pred) = res11
    (res21_512, res21_256, res21_128, res21_coarseness_pred) = res21

    # Save frame_i.img as a PNG image
    img_to_save = frame_i.img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Convert tensor to numpy array
    img_to_save = ((img_to_save /2 + 0.5) * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    img_to_save = Image.fromarray(img_to_save)  # Convert to PIL Image
    os.makedirs("output_images", exist_ok=True)  # Ensure the output directory exists
    img_to_save.save("logs/frame_i.png")  # Save the image
    img_to_save = frame_j.img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # Convert tensor to numpy array
    img_to_save = ((img_to_save /2 + 0.5) * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8
    img_to_save = Image.fromarray(img_to_save)  # Convert to PIL Image
    os.makedirs("output_images", exist_ok=True)  # Ensure the output directory exists
    img_to_save.save("logs/frame_j.png")  # Save the image

    # res11_512['sh'], res21_512['sh'] = add_frame_color_to_sh(frame_i=frame_i, frame_j=frame_j, SHii=res11_512['sh'], SHji=res21_512['sh'])
    img_sh11 = get_img_sh(frame_i, res11_512['sh'])
    img_sh21 = get_img_sh(frame_j, res21_512['sh'])
    res11_512, mask11_used_gaussians, coarseness_pred11 = use_coarseness_prediction(res11, img_sh11)
    res21_512, mask21_used_gaussians, coarseness_pred21 = use_coarseness_prediction(res21, img_sh21)
    MASKS = torch.stack([mask11_used_gaussians, mask21_used_gaussians])
    COARSE_PRED = torch.stack([coarseness_pred11, coarseness_pred21])
    res = [res11_512, res21_512]
    
    X, C, D, Q, S, R, SH, O, M  = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0], r["scales"][0], r["rotations"][0], r["sh"][0], r["opacities"][0], r["means"][0]) for r in res]
    )
    # print("Gaussians sh shape ", res[0][]))
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    S, R, SH, O, M = torch.stack(S), torch.stack(R), torch.stack(SH), torch.stack(O), torch.stack(M)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q, S, R, SH, O, M, MASKS, COARSE_PRED


def use_coarseness_prediction(model_output, img_sh):
    (pred_512, pred_256, pred_128, coarseness) = model_output

    valid = (pred_512['conf'] > 1.5)

    # Downsample the SH coefficients
    pred_512['sh'] = pred_512['sh'] + img_sh
    sh_256 = (img_sh[:,::2,::2,:] + img_sh[:,1::2,::2,:] + img_sh[:,::2,1::2,:] + img_sh[:,1::2,1::2,:])/ 4.0
    pred_256['sh'] = pred_256['sh'] + sh_256
    sh_128 = (sh_256[:,::2,::2,:] + sh_256[:,1::2,::2,:] + sh_256[:,::2,1::2,:] + sh_256[:,1::2,1::2,:])/ 4.0
    pred_128['sh'] = pred_128['sh'] + sh_128

    means = pred_512['means']
    pred_256['means'] = (means[:,::2,::2,:] + means[:,1::2,::2,:] + means[:,::2,1::2,:] + means[:,1::2,1::2,:]) / 4.0
    means_256 = pred_256['means']
    pred_128['means'] = (means_256[:,::2,::2,:] + means_256[:,1::2,::2,:] + means_256[:,::2,1::2,:] + means_256[:,1::2,1::2,:]) / 4.0

    classes = torch.argmax(coarseness, dim=1) # coarseness: (b, c, h, w) -> classes: (b, h, w)
    # print(f"calasses.shape: {classes.shape}, coarseness.shape: {coarseness.shape}")   
    mask_512_use = (classes == 0) & valid
    mask_256_use = (classes == 1) & valid
    mask_128_use = (classes == 2) & valid

    coarseness_pred = torch.cat((mask_512_use, mask_256_use, mask_128_use), dim=0) # out: (3, h, w)

    # Save mask_512_use as an image
    mask_256_use_256 = mask_256_use[:,::2,::2] & mask_256_use[:,1::2,::2] & mask_256_use[:,::2,1::2] & mask_256_use[:,1::2,1::2]
    mask_128_use_256 = mask_128_use[:,::2,::2] & mask_128_use[:,1::2,::2] & mask_128_use[:,::2,1::2] & mask_128_use[:,1::2,1::2]
    # print(f"mask_128_use.shape 2: {mask_128_use.shape}")
    mask_128_use_128 = mask_128_use_256[:,::2,::2] & mask_128_use_256[:,1::2,::2] & mask_128_use_256[:,::2,1::2] & mask_128_use_256[:,1::2,1::2]
    # print(f"mask_128_use.shape 3: {mask_128_use_256.shape}")
    # print(f"mask_128_xor_256.shape: {mask_128_xor_256.shape}, mask_128_xor_128.shape: {mask_128_xor_128.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")
    # mask_128_xor_256_512 = torch.nn.functional.interpolate(mask_128_xor_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    # mask_128_xor_128_256 = torch.nn.functional.interpolate(mask_128_xor_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    # print(f"mask_128_xor_128_256.shape: {mask_128_xor_128_256.shape}, mask_128_xor_256_512.shape: {mask_128_xor_256_512.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")


    # mask_128_use_128 to use on 128 resolution
    mask_128_use_128_upsampled_256 = torch.nn.functional.interpolate(mask_128_use_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_128_to_256 = mask_128_use_256 & ~mask_128_use_128_upsampled_256
    mask_256_use_256 = mask_256_use_256 | mask_128_to_256

    mask_128_use_256_upsampled_512 = torch.nn.functional.interpolate(mask_128_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_128_to_512 = mask_128_use & ~mask_128_use_256_upsampled_512

    mask_256_upsampled_512 = torch.nn.functional.interpolate(mask_256_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_256_to_512 = mask_256_use & ~mask_256_upsampled_512
    mask_512_use = mask_512_use | mask_256_to_512 | mask_128_to_512

    mask_256_indices = torch.nonzero(mask_256_use_256)
    u_256, v_256 = mask_256_indices[:, 1]*2, mask_256_indices[:, 2]*2
    mask_128_indices = torch.nonzero(mask_128_use_128)
    u_128, v_128 = mask_128_indices[:, 1]*4, mask_128_indices[:, 2]*4
    # print(f"u_256.shape: {u_256.shape}, v_256.shape: {v_256.shape}")
    # print(f"u_128.shape: {u_128.shape}, v_128.shape: {v_128.shape}")
    # print(f"u_256: {u_256}, v_256: {v_256}")
    # print(f"u_128: {u_128}, v_128: {v_128}")
    for key in pred_512.keys():
        # if key not in ["means", "means_in_other_view", "opacities", "sh", "rotations", "scales", "covariances"]:
        if key not in ["opacities", "sh", "rotations", "scales", "covariances"]:
            continue
        # pred_512[key][0,mask_512_use[0,...],...] Stays the same
        pred_512[key][0,u_256, v_256, ...] = pred_256[key][0,mask_256_use_256[0,...],...]
        pred_512[key][0,u_128, v_128, ...] = pred_128[key][0,mask_128_use_128[0,...],...]
    
    
    mask_used_gaussians = mask_512_use
    mask_used_gaussians[:,u_256, v_256] = True
    mask_used_gaussians[:,u_128, v_128] = True
    mask_used_gaussians = mask_used_gaussians.squeeze(0) # (b, h, w) -> (h, w)
    
    # pred_512['covariances'] = geometry.build_covariance(pred_512['scales'], pred_512['rotations'])
    # pred_256['covariances'] = geometry.build_covariance(pred_256['scales'], pred_256['rotations'])
    # pred_128['covariances'] = geometry.build_covariance(pred_128['scales'], pred_128['rotations'])
    # pred_combined = {}

    # mask_128_use_128 = torch.ones_like(mask_128_use_128).bool()

    # for key in pred_512.keys():
    #     if key not in ["means", "means_in_other_view", "opacities", "sh", "rotations", "scales", "covariances"]:
    #         continue
    #     b=0
    #     # pred_combined[key] = torch.cat([pred_512[key][b,mask_512_use[b,...],...], 
    #     #                                               pred_256[key][b,mask_256_use_256[b,...],...], 
    #     #                                               pred_128[key][b,mask_128_use_128[b,...],...]]
    #     #                                               , dim=0)
    #     pred_combined[key] = pred_128[key][b,mask_128_use_128[b,...],...]
    #     pred_combined[key] = pred_combined[key].unsqueeze(0) # add batch dimension
    
    # save_as_ply(pred_combined, pred_combined, save_path="logs/gaussians_mast3r_utils.ply", grad_coarseness=True)
    return pred_512, mask_used_gaussians, coarseness_pred #(mask in (h, w) format)

def use_coarseness_prediction_for_means(means, coarseness_pred):
    means_256 = (means[:,::2,::2,:] + means[:,1::2,::2,:] + means[:,::2,1::2,:] + means[:,1::2,1::2,:]) / 4.0
    means_128 = (means_256[:,::2,::2,:] + means_256[:,1::2,::2,:] + means_256[:,::2,1::2,:] + means_256[:,1::2,1::2,:]) / 4.0
    mask_512_use, mask_256_use, mask_128_use = coarseness_pred[0, None, ...], coarseness_pred[1, None, ...], coarseness_pred[2, None, ...]
    # Save mask_512_use as an image
    mask_256_use_256 = mask_256_use[:,::2,::2] & mask_256_use[:,1::2,::2] & mask_256_use[:,::2,1::2] & mask_256_use[:,1::2,1::2]
    mask_128_use_256 = mask_128_use[:,::2,::2] & mask_128_use[:,1::2,::2] & mask_128_use[:,::2,1::2] & mask_128_use[:,1::2,1::2]
    mask_128_use_128 = mask_128_use_256[:,::2,::2] & mask_128_use_256[:,1::2,::2] & mask_128_use_256[:,::2,1::2] & mask_128_use_256[:,1::2,1::2]
    # print(f"mask_128_use.shape 3: {mask_128_use_256.shape}")
    # print(f"mask_128_xor_256.shape: {mask_128_xor_256.shape}, mask_128_xor_128.shape: {mask_128_xor_128.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")
    # mask_128_xor_256_512 = torch.nn.functional.interpolate(mask_128_xor_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    # mask_128_xor_128_256 = torch.nn.functional.interpolate(mask_128_xor_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    # print(f"mask_128_xor_128_256.shape: {mask_128_xor_128_256.shape}, mask_128_xor_256_512.shape: {mask_128_xor_256_512.shape}, mask_256_xor_512.shape: {mask_256_xor_512.shape}")

    # mask_128_use_128 to use on 128 resolution
    mask_128_use_128_upsampled_256 = torch.nn.functional.interpolate(mask_128_use_128.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_128_to_256 = mask_128_use_256 & ~mask_128_use_128_upsampled_256
    mask_256_use_256 = mask_256_use_256 | mask_128_to_256

    mask_128_use_256_upsampled_512 = torch.nn.functional.interpolate(mask_128_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_128_to_512 = mask_128_use & ~mask_128_use_256_upsampled_512

    mask_256_upsampled_512 = torch.nn.functional.interpolate(mask_256_use_256.float().unsqueeze(1), scale_factor=2, mode='nearest').squeeze(1).bool()
    mask_256_to_512 = mask_256_use & ~mask_256_upsampled_512
    mask_512_use = mask_512_use | mask_256_to_512 | mask_128_to_512

    mask_256_indices = torch.nonzero(mask_256_use_256)
    u_256, v_256 = mask_256_indices[:, 1]*2, mask_256_indices[:, 2]*2
    mask_128_indices = torch.nonzero(mask_128_use_128)
    u_128, v_128 = mask_128_indices[:, 1]*4, mask_128_indices[:, 2]*4

    means[0,u_256, v_256, ...] = means_256[0,mask_256_use_256[0,...],...]
    means[0,u_128, v_128, ...] = means_128[0,mask_128_use_128[0,...],...]
    return means

def get_img_sh(frame, SH):
    # add frame colors to sh colors
    new_sh = torch.zeros_like(SH)
    new_sh[..., 0] = sh_utils.RGB2SH(einops.rearrange(frame.img/2.0+0.5, 'b c h w -> b h w c'))
    return new_sh

def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q, S, R, SH, O, M, MASKS, COARSE_PRED = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

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
    Maskii, Maskji = einops.rearrange(MASKS, "b h w -> b (h w)")
    Coarse_predii, Coarse_predji = einops.rearrange(COARSE_PRED, "b c h w -> b c (h w)")

    # # add frame colors to sh colors
    # new_sh1 = torch.zeros_like(SHii)
    # new_sh2 = torch.zeros_like(SHji)
    # new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(frame_i.img/2.0+0.5, 'b c h w -> b (h w) c'))
    # new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(frame_j.img/2.0+0.5, 'b c h w -> b (h w) c'))
    # SHii = SHii + new_sh1
    # SHji = SHji + new_sh2

    gaussian_params = (Sii, Rii, SHii, Oii, Mii, Maskii, Coarse_predii, Sji, Rji, SHji, Oji, Mji, Maskji, Coarse_predji)
    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji, gaussian_params


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512 or size == 256
    # print(f"image np shape: {img.shape}")
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    # print(f"W1: {W1}, H1: {H1}, size: {size}, square_ok: {square_ok}")
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        # img = _resize_pil_image(img, size)
    elif size == 256:
        img = _resize_pil_image(img, size)
    else:
        # our_method = True
        # if our_method:
        #     # Resize short side to 512
        #     W1, H1 = img.size
        #     img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
        #     W, H = img.size
        #     cx, cy = W//2, H//2
        #     half = min(cx, cy)
        #     img = img.crop((cx-half, cy-half, cx+half, cy+half))
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    # print(f"After resize: W: {W}, H: {H}, size: {size}, square_ok: {square_ok}")
    # print(f"W: {W}, H: {H}, size: {size}, square_ok: {square_ok}")
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    elif size == 256:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))
        # print(f"cx: {cx}, cy: {cy}, halfw: {halfw}, halfh: {halfh}, W: {W}, H: {H}")
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
