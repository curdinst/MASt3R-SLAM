import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None
    SH: Optional[torch.Tensor] = None
    opacities: Optional[torch.Tensor] = None
    offsets: Optional[torch.Tensor] = None
    rotations: Optional[torch.Tensor] = None
    scales: Optional[torch.Tensor] = None
    N_guass: int = 0
    N_gauss_updates: int = 0

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor, scale: torch.Tensor=None, rotation: torch.Tensor=None, SH: torch.Tensor=None, opacity: torch.Tensor=None, mean: torch.Tensor=None):
        filtering_mode = config["tracking"]["filtering_mode"]
        # filtering_mode = "first"

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            # Gaussian params
            if scale is not None:
                self.SH = SH.clone()
                self.opacities = opacity.clone()
                self.offsets = mean.clone() - self.X_canon # only store offsets
                self.rotations = rotation.clone()
                self.scales = scale.clone()
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.SH = SH.clone()
                self.opacities = opacity.clone()
                self.offsets = mean.clone() - self.X_canon # only store offsets
                self.rotations = rotation.clone()
                self.scales = scale.clone()

        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            
            # Gaussian params
            gaussian_filtering_mode = ["weigtend_average", "recent", "first"][0]

            if gaussian_filtering_mode == "weigtend_average" and scale is not None and self.scales is not None:
                # self.SH = ((self.C.unsqueeze(1) * self.SH) + (C.unsqueeze(1) * SH)) / self.C.unsqueeze(1)
                self.SH = SH.clone()
                # print(f"C shape: {C.shape}, SH shape: {SH.shape}")
                self.opacities = ((self.C * self.opacities) + (C * opacity)) / (self.C  + C)
                self.offsets = ((self.C * self.offsets) + (C * (mean - X))) / (self.C  + C)
                self.rotations = ((self.C * self.rotations) + (C * rotation)) / (self.C  + C)
                self.scales = ((self.C * self.scales) + (C * scale)) / (self.C  + C)
            elif gaussian_filtering_mode == "recent" and scale is not None:
                self.SH = SH.clone()
                self.opacities = opacity.clone()
                self.offsets = mean.clone() - self.X_canon # only store offsets
                self.rotations = rotation.clone()
                self.scales = scale.clone()
            elif gaussian_filtering_mode == "first" and scale is not None and self.N_updates == 1:
                print("Save First Gaussian params")
                self.SH = SH.clone()
                self.opacities = opacity.clone()
                self.offsets = mean.clone() - self.X_canon # only store offsets
                self.rotations = rotation.clone()
                self.scales = scale.clone()
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1
        return

    # @added
    def update_gaussians(self, valid_mask: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor, SH: torch.Tensor, opacity: torch.Tensor, mean: torch.Tensor):
        filtering_mode = "recent" # "weighted_pointmap"  # config["tracking"]["filtering_mode"]
        # if self.N_guass == 0:
        #     self.SH = SH.clone()
        #     self.opacities = opacity.clone()
        #     self.offsets = mean.clone() - self.X_canon # only store offsets
        #     self.rotations = rotation.clone()
        #     self.scales = scale.clone()
        #     self.N_guass = 1
        #     self.N_gauss_updates = 1
        #     return
        
        if filtering_mode == "recent":
            self.SH[valid_mask] = SH.clone()
            self.opacities[valid_mask] = opacity.clone()
            self.offsets[valid_mask] = mean.clone() - self.X_canon[valid_mask] # only store offsets
            self.rotations[valid_mask] = rotation.clone()
            self.scales[valid_mask] = scale.clone()
        # elif filtering_mode == "weighted_pointmap":
        #     self.SH = ((self.C * self.SH) + (C * SH)) / self.C
        #     self.opacities = ((self.C * self.opacities) + (C * opacity)) / self.C
        #     self.offsets = ((self.C * self.offsets) + (C * (mean - X))) / self.C
        #     self.rotations = ((self.C * self.rotations) + (C * rotation)) / self.C
        #     self.scales = ((self.C * self.scales) + (C * scale)) / self.C
        #     self.N_gauss += 1
        self.N_gauss_updates += 1
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None



def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()
        self.gauss_opt_frame = manager.Value("i", -1)

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # Gaussian parameters
        self.SH = torch.zeros(h * w, 3, 1, device=device, dtype=dtype).share_memory_()
        self.opacities = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.offsets = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.rotations = torch.zeros(h * w, 4, device=device, dtype=dtype).share_memory_()
        self.scales = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos
            if frame.SH is not None:
                self.SH[:] = frame.SH
                self.opacities[:] = frame.opacities
                self.offsets[:] = frame.offsets
                self.rotations[:] = frame.rotations
                self.scales[:] = frame.scales

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
                self.SH,
                self.opacities,
                self.offsets,
                self.rotations,
                self.scales
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos

            frame.SH = self.SH
            frame.opacities = self.opacities
            frame.offsets = self.offsets
            frame.rotations = self.rotations
            frame.scales = self.scales
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1
    
    def set_gauss_opt_frameid(self, frame_id):
        with self.lock:
            self.gauss_opt_frame.value = frame_id
    
    def get_gauss_opt_frameid(self):
        with self.lock:
            return self.gauss_opt_frame.value


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on
        self.SH = torch.zeros(buffer, h * w, 3, 1, device=device, dtype=dtype).share_memory_()
        self.opacities = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.offsets = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.rotations = torch.zeros(buffer, h * w, 4, device=device, dtype=dtype).share_memory_()
        self.scales = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # put all of the data into a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
                self.SH[idx],
                self.opacities[idx],
                self.offsets[idx],
                self.rotations[idx],
                self.scales[idx]
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            if self.SH[idx] is not None:
                kf.SH = self.SH[idx]
                kf.opacities = self.opacities[idx]
                kf.offsets = self.offsets[idx]
                kf.rotations = self.rotations[idx]
                kf.scales = self.scales[idx]
            else:
                print("get SH is None")
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True
            if value.SH is not None:
                # print(f"setting SH of frame: {value.frame_id} with {value.SH.shape if value.SH is not None else str(None)}")
                # sum_non_none_SH = 0
                # for i in range(self.n_size.value):
                #     if self.SH[i] is not None:
                #         print(i)
                #         sum_non_none_SH += 1
                # print(f"non None SH: {sum_non_none_SH}")
                self.SH[idx] = value.SH
                self.opacities[idx] = value.opacities
                self.offsets[idx] = value.offsets
                self.rotations[idx] = value.rotations
                self.scales[idx] = value.scales
            else:
                print("set SH is None")
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            # print(f"Updating T_WC for idx: {idx}")
            # print(f"self.TWC[idx]: {self.T_WC[idx]}")
            # print(f"T_WCs.data: {T_WCs.data}")
            # print(f"Position corrections: {T_WCs.data[:,0,:3] - self.T_WC[idx][:,0,:3]}")
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
        
    # def update_gaussians(self, frame_idx, valid_mask, offset, scale, rotation, opacity, SH):
    #     with self.lock:
    #         self.SH[frame_idx][valid_mask] = SH
    #         self.opacities[frame_idx][valid_mask] = opacity
    #         self.offsets[frame_idx][valid_mask] = offset
    #         self.rotations[frame_idx][valid_mask] = rotation
    #         self.scales[frame_idx][valid_mask] = scale
