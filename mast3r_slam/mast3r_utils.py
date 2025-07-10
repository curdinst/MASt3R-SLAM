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
        res = [res11, res21, res22, res12]
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
    print(f"frame.img.shape: {frame.img.shape}, frame.true_shape: {frame.img_true_shape}")

    # downsampled_image = torch.nn.functional.interpolate(
    #     frame.img.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False
    # ).squeeze(0)
    # new_image_shape = frame.img_true_shape // 2

    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
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

    # downsampled_image_i = torch.nn.functional.interpolate(
    #     frame_i.img.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False
    # ).squeeze(0)
    # new_image_shape_i = frame_i.img_true_shape // 2
    # downsampled_image_j = torch.nn.functional.interpolate(
    #     frame_j.img.unsqueeze(0), scale_factor=0.5, mode="bilinear", align_corners=False
    # ).squeeze(0)
    # new_image_shape_j = frame_j.img_true_shape // 2
    
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
    res = [res11, res21]
    X, C, D, Q, S, R, SH, O, M  = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0], r["scales"][0], r["rotations"][0], r["sh"][0], r["opacities"][0], r["means"][0]) for r in res]
    )
    # print("Gaussians sh shape ", res[0][]))
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    S, R, SH, O, M = torch.stack(S), torch.stack(R), torch.stack(SH), torch.stack(O), torch.stack(M)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q, S, R, SH, O, M


def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    X, C, D, Q, S, R, SH, O, M = mast3r_asymmetric_inference(model, frame_i, frame_j)

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

    # add frame colors to sh colors
    new_sh1 = torch.zeros_like(SHii)
    new_sh2 = torch.zeros_like(SHji)
    new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(frame_i.img/2.0+0.5, 'b c h w -> b (h w) c'))
    new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(frame_j.img/2.0+0.5, 'b c h w -> b (h w) c'))
    SHii = SHii + new_sh1
    SHji = SHji + new_sh2

    gaussian_params = (Sii, Rii, SHii, Oii, Mii, Sji, Rji, SHji, Oji, Mji)
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
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
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
