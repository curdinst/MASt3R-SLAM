# import json
# import os
# import sys

# import einops
# import lightning as L
# import lpips
# import omegaconf
import torch
# import wandb

# # Add MAST3R and PixelSplat to the sys.path to prevent issues during importing
# sys.path.append('src/pixelsplat_src')
# sys.path.append('src/mast3r_src')
# sys.path.append('src/mast3r_src/dust3r')
# from thirdparty.splatt3r.src.mast3r_src.dust3r.dust3r.losses import L21
# from thirdparty.splatt3r.src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
# import thirdparty.splatt3r.data.scannetpp.scannetpp as scannetpp
# import thirdparty.splatt3r.src.mast3r_src.mast3r.model as mast3r_model
# import thirdparty.splatt3r.src.pixelsplat_src.benchmarker as benchmarker
# import thirdparty.splatt3r.src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
# import thirdparty.splatt3r.utils.compute_ssim as compute_ssim
# import thirdparty.splatt3r.utils.export as export
# import thirdparty.splatt3r.utils.geometry as geometry
# import thirdparty.splatt3r.utils.loss_mask as loss_mask
# import thirdparty.splatt3r.utils.sh_utils as sh_utils
# import workspace


# mast3r = torch.load('checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
# # splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/keep/25-08-11-20-19-28_epoch=15_step=65376.ckpt')
# splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/keep/25-08-19-22-44-28_epoch=11_step=49032.ckpt')

# # print(mast3r.keys())
# # print(mast3r['args'])
# # print("---------------------------------------------------------------")
# # print(splatt3r.keys())

# mast3r_gaussians = mast3r.copy()
# for key in splatt3r['state_dict'].keys():
#     key_modified = key.replace('encoder.', '')
#     mast3r_gaussians['model'][key_modified] = splatt3r['state_dict'][key]


# # MASt3R_gaussians_v1 = torch.load('checkpoints/MASt3R_gaussians_v1.pth', map_location='cpu')
# # MASt3R_gaussians_v1_keys = MASt3R_gaussians_v1['model'].keys()

# # for key in MASt3R_gaussians_v1_keys:
# #     if 'gaussian' in key:
# #         print(key)

# # print(MASt3R_gaussians_v1['model']['downstream_head1.dpt.act_postprocess.0.0.weight'])


# torch.save(mast3r_gaussians, 'checkpoints/MASt3R_gaussians_3stage.pth')



# ------------- Transfer a part of the model --------------


mast3r = torch.load('checkpoints/MASt3R_gaussians_3stage_single_map.pth', map_location='cpu')
# splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/keep/25-08-11-20-19-28_epoch=15_step=65376.ckpt')
# splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/keep/splatt3r_3stage_freq_pred.ckpt')
# splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/25-08-29-17-43-59_epoch=01_step=04086.ckpt', map_location='cpu')
splatt3r = torch.load('/home/curdinst/repos/splatt3r/checkpoints/keep/25-08-29-02-41-17_epoch=10_step=22473_pred2.ckpt', map_location='cpu')

# pretrained_mast3r_path: './checkpoints/25-08-29-02-41-17_epoch=08_step=18387.ckpt'


# print(mast3r.keys())
# print(mast3r['args'])
# print("---------------------------------------------------------------")
# print(splatt3r.keys())

mast3r_gaussians = mast3r.copy()
for key in splatt3r['state_dict'].keys():
    if 'gaussian' not in key or 'downstream_head1' in key:
        continue
    print(f"key: {key}")
    key_modified = key.replace('encoder.', '')
    mast3r_gaussians['model'][key_modified] = splatt3r['state_dict'][key]


# MASt3R_gaussians_v1 = torch.load('checkpoints/MASt3R_gaussians_v1.pth', map_location='cpu')
# MASt3R_gaussians_v1_keys = MASt3R_gaussians_v1['model'].keys()

# for key in MASt3R_gaussians_v1_keys:
#     if 'gaussian' in key:
#         print(key)

# print(MASt3R_gaussians_v1['model']['downstream_head1.dpt.act_postprocess.0.0.weight'])


torch.save(mast3r_gaussians, 'checkpoints/MASt3R_gaussians_3stage_single_map.pth')