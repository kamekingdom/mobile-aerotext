# Author: kamekingdom (2025-08-20)
"""
Formal training script for a three-scale, anchor-free detector on format datasets.
Model = Proposed Method (CoordConv + ELAN + ASPP-ELAN + ECA, FPN-PAN neck, decoupled heads).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any, Iterable
import math
import os
import random
import time
import glob

import cv2  # type: ignore
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm


# ---------------------------
# Configuration
# ---------------------------

@dataclass
class TrainConfig:
    data_root: str                     # path to dataset root containing 'images/train', 'labels/train', etc.
    img_size: int = 640                # square input
    num_classes: int = 46              # change as needed
    batch_size: int = 16
    epochs: int = 100
    lr: float = 2.5e-4
    weight_decay: float = 5e-3
    warmup_epochs: float = 3.0
    workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./runs/train_proposed"
    seed: int = 42
    # detection head / losses
    strides: Tuple[int, int, int] = (8, 16, 32)   # P3, P4, P5
    box_iou_type: str = "ciou"                    # ['iou','giou','diou','ciou']
    cls_loss_gamma: float = 2.0                   # focal gamma
    cls_loss_alpha: float = 0.25                  # focal alpha
    obj_loss_lambda: float = 1.0
    cls_loss_lambda: float = 0.5
    box_loss_lambda: float = 7.5                  # emphasize box regression


# ---------------------------
# Utility functions
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore


def xywhn_to_xyxy(img_w: int, img_h: int, xywhn: np.ndarray) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] to pixel [x1,y1,x2,y2]."""
    cx = xywhn[:, 0] * img_w
    cy = xywhn[:, 1] * img_h
    w = xywhn[:, 2] * img_w
    h = xywhn[:, 3] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def letterbox(img: np.ndarray, new_size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize+pad to square, return image, scale ratio, padding."""
    h, w = img.shape[:2]
    r = min(new_size / h, new_size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_size - nw, new_size - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return out, r, (left, top)


def box_iou_ciou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    CIoU between boxes in [x1,y1,x2,y2] (N,4).
    """
    # Intersection
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    # Areas
    area_p = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    area_t = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union

    # enclosing box
    xc1 = torch.min(pred[:, 0], target[:, 0])
    yc1 = torch.min(pred[:, 1], target[:, 1])
    xc2 = torch.max(pred[:, 2], target[:, 2])
    yc2 = torch.max(pred[:, 3], target[:, 3])
    c2 = (xc2 - xc1).clamp(min=0) ** 2 + (yc2 - yc1).clamp(min=0) ** 2 + eps

    # centers and distance
    pcx = (pred[:, 0] + pred[:, 2]) / 2
    pcy = (pred[:, 1] + pred[:, 3]) / 2
    tcx = (target[:, 0] + target[:, 2]) / 2
    tcy = (target[:, 1] + target[:, 3]) / 2
    rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

    # aspect ratio term
    pw = (pred[:, 2] - pred[:, 0]).clamp(min=eps)
    ph = (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    tw = (target[:, 2] - target[:, 0]).clamp(min=eps)
    th = (target[:, 3] - target[:, 1]).clamp(min=eps)
    v = (4 / (math.pi ** 2)) * torch.pow(torch.atan(tw / th) - torch.atan(pw / ph), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    ciou = iou - (rho2 / c2 + v * alpha)
    return ciou


# ---------------------------
# Dataset
# ---------------------------

class Dataset(Dataset):
    """
    -format dataset.
    Each image has a corresponding labels file with lines: 'cls cx cy w h' (normalized).
    """

    def __init__(self, root: str, split: str, img_size: int) -> None:
        self.img_dir = os.path.join(root, "images", split)
        self.label_dir = os.path.join(root, "labels", split)
        self.img_size = img_size
        self.paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        assert len(self.paths) > 0, f"No images found in {self.img_dir}"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        assert img is not None, f"Failed to read {path}"
        h0, w0 = img.shape[:2]
        img, ratio, pad = letterbox(img, self.img_size)

        # labels
        label_path = os.path.join(self.label_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if os.path.exists(label_path):
            labels = []
            with open(label_path, "r") as f:
                for line in f.readlines():
                    vals = line.strip().split()
                    if len(vals) != 5:
                        continue
                    c = int(vals[0])
                    cx, cy, w, h = map(float, vals[1:])
                    # convert to pixel in resized image
                    xyxy = xywhn_to_xyxy(w0, h0, np.array([[cx, cy, w, h]], dtype=np.float32))[0]
                    # scale and pad
                    x1 = xyxy[0] * ratio + pad[0]
                    y1 = xyxy[1] * ratio + pad[1]
                    x2 = xyxy[2] * ratio + pad[0]
                    y2 = xyxy[3] * ratio + pad[1]
                    labels.append([c, x1, y1, x2, y2])
            labels_np = np.array(labels, dtype=np.float32) if len(labels) > 0 else np.zeros((0, 5), dtype=np.float32)
        else:
            labels_np = np.zeros((0, 5), dtype=np.float32)

        # to tensor
        img = img[:, :, ::-1]  # BGR->RGB
        img = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)
        labels_t = torch.from_numpy(labels_np)  # (N,5) [c,x1,y1,x2,y2]
        return img_t, labels_t


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = [b[1] for b in batch]
    return imgs, labels


# ---------------------------
# Model components
# ---------------------------

class ConvBNSiLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None, g: int = 1) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CoordConv2d(nn.Module):
    """Append normalized (x,y) coordinate channels."""
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, p: Optional[int] = None) -> None:
        super().__init__()
        self.conv = ConvBNSiLU(c_in + 2, c_out, k, s, p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing="ij")
        coord = torch.stack([xx.expand(b, 1, h, w), yy.expand(b, 1, h, w)], dim=1).squeeze(2)
        x = torch.cat([x, coord], dim=1)
        return self.conv(x)


class ECA(nn.Module):
    """Efficient Channel Attention: global avg pool + 1D conv."""
    def __init__(self, c: int, k_size: int = 3) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg(x)  # (B,C,1,1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B,C,1)
        y = self.act(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ELANBlock(nn.Module):
    """Simplified ELAN: two-level multi-branch conv aggregation."""
    def __init__(self, c_in: int, c_out: int, depth: int = 2) -> None:
        super().__init__()
        c_hidden = c_out // 2
        self.stem = ConvBNSiLU(c_in, c_hidden, 1, 1, 0)
        self.branches = nn.ModuleList([
            ConvBNSiLU(c_hidden, c_hidden, 3, 1),
            ConvBNSiLU(c_hidden, c_hidden, 3, 1),
        ])
        self.fuse = ConvBNSiLU(c_hidden * (2 + depth), c_out, 1, 1, 0)
        self.extra = nn.ModuleList([ConvBNSiLU(c_hidden, c_hidden, 3, 1) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(x)
        outs = [x0]
        x1 = self.branches[0](x0)
        outs.append(x1)
        x2 = self.branches[1](x1)
        outs.append(x2)
        xi = x2
        for m in self.extra:
            xi = m(xi)
            outs.append(xi)
        return self.fuse(torch.cat(outs, dim=1))


class ASPP_ELAN(nn.Module):
    """Parallel dilated convs + 1x1 fuse, ELAN-style."""
    def __init__(self, c_in: int, c_out: int, dilations: Tuple[int, int, int] = (1, 2, 3)) -> None:
        super().__init__()
        branches = []
        c_each = c_out // (len(dilations) + 1)
        branches.append(ConvBNSiLU(c_in, c_each, 1, 1, 0))
        for d in dilations:
            branches.append(ConvBNSiLU(c_in, c_each, 3, 1, d, g=1))
            branches[-1].conv.padding = d  # type: ignore
            branches[-1].conv.dilation = d  # type: ignore
        self.branches = nn.ModuleList(branches)
        self.fuse = ConvBNSiLU(c_each * len(branches), c_out, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [m(x) for m in self.branches]
        return self.fuse(torch.cat(outs, dim=1))


class Downsample(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = ConvBNSiLU(c_in, c_out, 3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DetectHead(nn.Module):
    """
    Decoupled head producing (cx,cy,w,h,obj,cls...) per location.
    For each scale, output channels = 4 + 1 + num_classes.
    """
    def __init__(self, c_in: int, num_classes: int) -> None:
        super().__init__()
        c_mid = max(64, c_in // 2)
        self.reg = nn.Sequential(
            ConvBNSiLU(c_in, c_mid, 3, 1),
            nn.Conv2d(c_mid, 4, 1, 1, 0)
        )
        self.obj = nn.Sequential(
            ConvBNSiLU(c_in, c_mid, 3, 1),
            nn.Conv2d(c_mid, 1, 1, 1, 0)
        )
        self.cls = nn.Sequential(
            ConvBNSiLU(c_in, c_mid, 3, 1),
            nn.Conv2d(c_mid, num_classes, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.reg(x), self.obj(x), self.cls(x)


class ProposedDetector(nn.Module):
    """
    Whole network assembling backbone, neck, and three-scale decoupled heads.
    """
    def __init__(self, num_classes: int = 46) -> None:
        super().__init__()
        # Stem
        self.stem0 = CoordConv2d(3, 64, 3, 2)      # Index 0 (CoordConv)
        self.stem1 = ConvBNSiLU(64, 128, 3, 2)     # Index 1

        # Stage 1
        self.b1 = ELANBlock(128, 128, depth=2)     # Index 2
        self.d1 = Downsample(128, 256)             # Index 3

        # Stage 2
        self.b2 = ELANBlock(256, 256, depth=2)     # Index 4
        self.d2 = Downsample(256, 512)             # Index 5

        # Stage 3
        self.b3 = ELANBlock(512, 512, depth=2)     # Index 6
        self.d3 = Downsample(512, 512)             # Index 7
        self.b4 = ELANBlock(512, 512, depth=2)     # Index 8

        # ASPP-ELAN
        self.spp = ASPP_ELAN(512, 512)             # Index 9

        # Neck (FPN-PAN)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 10
        self.p4_lateral = ConvBNSiLU(512, 256, 1, 1, 0)
        self.c4 = ELANBlock(256 + 512 // 2, 256, depth=2)  # concat with b3 reduced
        self.eca4 = ECA(256)                         # 13

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")  # 14
        self.p3_lateral = ConvBNSiLU(256, 128, 1, 1, 0)
        self.c3 = ELANBlock(128 + 256 // 2, 128, depth=2)  # 16
        self.eca3 = ECA(128)                         # 17

        # PAN bottom-up
        self.down_p4 = Downsample(128, 256)         # 21
        self.c4_pan = ELANBlock(256 + 256, 256, depth=2)  # 23
        self.eca4b = ECA(256)                        # 20 (after PAN)

        self.down_p5 = Downsample(256, 512)         # additional down
        self.c5_pan = ELANBlock(512 + 512, 512, depth=2)

        # Heads
        self.h3 = DetectHead(128, num_classes)
        self.h4 = DetectHead(256, num_classes)
        self.h5 = DetectHead(512, num_classes)

        # Reduce features for concatenations
        self.reduce_b3 = ConvBNSiLU(512, 256, 1, 1, 0)
        self.reduce_b2 = ConvBNSiLU(256, 128, 1, 1, 0)
        self.reduce_spp = ConvBNSiLU(512, 256, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Backbone
        x = self.stem0(x)     # (B,64,H/2,W/2)
        x = self.stem1(x)     # (B,128,H/4,W/4)
        p2 = self.b1(x)       # (B,128,H/4,W/4)
        x = self.d1(p2)       # (B,256,H/8,W/8)
        p3b = self.b2(x)      # (B,256,H/8,W/8)
        x = self.d2(p3b)      # (B,512,H/16,W/16)
        p4b = self.b3(x)      # (B,512,H/16,W/16)
        x = self.d3(p4b)      # (B,512,H/32,W/32)
        p5b = self.b4(x)      # (B,512,H/32,W/32)
        spp = self.spp(p5b)   # (B,512,H/32,W/32)

        # Neck: top-down
        p5r = self.reduce_spp(spp)
        p4r = self.reduce_b3(p4b)
        y4 = self.up1(p5r)
        y4 = torch.cat([y4, p4r], dim=1)
        y4 = self.c4(y4)
        y4 = self.eca4(y4)

        p3r = self.reduce_b2(p3b)
        y3 = self.up2(y4)
        y3 = torch.cat([y3, p3r], dim=1)
        y3 = self.c3(y3)
        y3 = self.eca3(y3)

        # PAN: bottom-up
        z4 = self.down_p4(y3)
        z4 = torch.cat([z4, y4], dim=1)
        z4 = self.c4_pan(z4)
        z4 = self.eca4b(z4)

        z5 = self.down_p5(z4)
        z5 = torch.cat([z5, spp], dim=1)
        z5 = self.c5_pan(z5)

        # Heads
        reg3, obj3, cls3 = self.h3(y3)
        reg4, obj4, cls4 = self.h4(z4)
        reg5, obj5, cls5 = self.h5(z5)
        return {
            "P3": (reg3, obj3, cls3),
            "P4": (reg4, obj4, cls4),
            "P5": (reg5, obj5, cls5),
        }


# ---------------------------
# Target assignment (anchor-free)
# ---------------------------

class TargetAssigner:
    """
    Assign targets to grids at different strides.
    Simple size-based scale selection + nearest grid cell assignment.
    """

    def __init__(self, img_size: int, strides: Tuple[int, int, int], num_classes: int) -> None:
        self.img = img_size
        self.strides = strides
        self.num_classes = num_classes

    def select_scale(self, w: float, h: float) -> int:
        """Return scale index 0:P3(8),1:P4(16),2:P5(32) based on max side."""
        m = max(w, h)
        if m < 64:
            return 0
        elif m < 128:
            return 1
        else:
            return 2

    def build_targets(
        self,
        labels: List[torch.Tensor],
        device: torch.device
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        For a batch, build dense target maps for P3,P4,P5.
        Returns dict with keys 'P3','P4','P5'; each has
          'tbox': (B,4,H,W), 'tobj': (B,1,H,W), 'tcls': (B,C,H,W), 'mask': (B,1,H,W)
        """
        B = len(labels)
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for si, stride in enumerate(self.strides):
            S = self.img // stride
            out[f"P{si+3}"] = {
                "tbox": torch.zeros((B, 4, S, S), device=device),
                "tobj": torch.zeros((B, 1, S, S), device=device),
                "tcls": torch.zeros((B, self.num_classes, S, S), device=device),
                "mask": torch.zeros((B, 1, S, S), device=device, dtype=torch.bool),
            }

        for b, lab in enumerate(labels):
            if lab.numel() == 0:
                continue
            cls = lab[:, 0].long()
            x1, y1, x2, y2 = lab[:, 1], lab[:, 2], lab[:, 3], lab[:, 4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = (x2 - x1).clamp(min=1.0)
            h = (y2 - y1).clamp(min=1.0)
            for i in range(lab.shape[0]):
                scale_id = self.select_scale(float(w[i]), float(h[i]))
                stride = self.strides[scale_id]
                S = self.img // stride
                gx = int(torch.clamp(cx[i] / stride, 0, S - 1).item())
                gy = int(torch.clamp(cy[i] / stride, 0, S - 1).item())
                key = f"P{scale_id+3}"
                out[key]["tbox"][b, :, gy, gx] = torch.stack([x1[i], y1[i], x2[i], y2[i]])
                out[key]["tobj"][b, 0, gy, gx] = 1.0
                out[key]["tcls"][b, cls[i], gy, gx] = 1.0
                out[key]["mask"][b, 0, gy, gx] = True
        return out


# ---------------------------
# Loss
# ---------------------------

class DetectionLoss(nn.Module):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        preds: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        targets: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds: dict P3,P4,P5 -> (reg(4), obj(1), cls(C))
               shapes: (B,Ch,H,W)
        targets: dense maps; see TargetAssigner
        """
        device = next(iter(preds.values()))[0].device
        total = torch.zeros(1, device=device)
        logs: Dict[str, float] = {}

        for s, stride in zip(["P3", "P4", "P5"], self.cfg.strides):
            reg, obj, cls = preds[s]
            B, _, H, W = reg.shape

            # decode boxes: (sigmoid center offset + grid) * stride, w/h = exp()
            gy, gx = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
            gx = gx.float(); gy = gy.float()
            px = (torch.sigmoid(reg[:, 0]) + gx) * stride
            py = (torch.sigmoid(reg[:, 1]) + gy) * stride
            pw = (torch.exp(reg[:, 2]) * stride).clamp(min=1.0)
            ph = (torch.exp(reg[:, 3]) * stride).clamp(min=1.0)
            pred_xyxy = torch.stack([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], dim=1)

            tgt = targets[s]
            mask = tgt["mask"].squeeze(1)  # (B,H,W)

            # Box loss (only positives)
            if mask.any():
                p_sel = pred_xyxy.permute(0, 2, 3, 1)[mask]  # (Npos,4)
                t_sel = tgt["tbox"].permute(0, 2, 3, 1)[mask]  # (Npos,4)
                ciou = box_iou_ciou(p_sel, t_sel)
                l_box = (1.0 - ciou).mean()
            else:
                l_box = torch.zeros(1, device=device).mean()

            # Obj loss (BCE with logits)
            l_obj = nn.functional.binary_cross_entropy_with_logits(
                obj, tgt["tobj"], reduction="mean"
            )

            # Cls loss (focal on logits)
            l_cls = sigmoid_focal_loss(
                cls, tgt["tcls"], reduction="mean", gamma=self.cfg.cls_loss_gamma, alpha=self.cfg.cls_loss_alpha
            )

            loss = (self.cfg.box_loss_lambda * l_box
                    + self.cfg.obj_loss_lambda * l_obj
                    + self.cfg.cls_loss_lambda * l_cls)
            total = total + loss
            logs[f"{s}/box"] = float(l_box.detach().item())
            logs[f"{s}/obj"] = float(l_obj.detach().item())
            logs[f"{s}/cls"] = float(l_cls.detach().item())

        return total, logs


# ---------------------------
# Training loop
# ---------------------------

def cosine_lr(optimizer: torch.optim.Optimizer, base_lr: float, epoch: int, epochs: int, warmup: float) -> None:
    """Set LR per epoch with linear warmup + cosine decay."""
    if epoch < warmup:
        lr = base_lr * (epoch + 1) / max(1.0, warmup)
    else:
        t = (epoch - warmup) / max(1.0, epochs - warmup)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Data
    train_set = Dataset(cfg.data_root, "train", cfg.img_size)
    val_set = Dataset(cfg.data_root, "val", cfg.img_size)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True, collate_fn=collate_fn)

    # Model
    model = ProposedDetector(num_classes=cfg.num_classes).to(cfg.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))  # type: ignore
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = DetectionLoss(cfg)
    assigner = TargetAssigner(cfg.img_size, cfg.strides, cfg.num_classes)

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        cosine_lr(optimizer, cfg.lr, epoch, cfg.epochs, cfg.warmup_epochs)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        epoch_loss = 0.0
        for imgs, labels in pbar:
            imgs = imgs.to(cfg.device, non_blocking=True)
            labels = [l.to(cfg.device) for l in labels]

            targets = assigner.build_targets(labels, torch.device(cfg.device))
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.device.startswith("cuda"))):  # type: ignore
                preds = model(imgs)
                loss, logs = criterion(preds, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}",
                              "P3b": f"{logs['P3/box']:.3f}", "P4b": f"{logs['P4/box']:.3f}", "P5b": f"{logs['P5/box']:.3f}"})

        avg_train = epoch_loss / len(train_loader)

        # Validation (loss only; mAP calculation omitted for brevity)
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for imgs, labels in val_loader:
                imgs = imgs.to(cfg.device, non_blocking=True)
                labels = [l.to(cfg.device) for l in labels]
                targets = assigner.build_targets(labels, torch.device(cfg.device))
                preds = model(imgs)
                loss, _ = criterion(preds, targets)
                val_loss += float(loss.item())
            avg_val = val_loss / max(1, len(val_loader))

        print(f"[Epoch {epoch+1}] train {avg_train:.4f} | val {avg_val:.4f}")

        # Save
        save_path = os.path.join(cfg.save_dir, "last.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__}, save_path)
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg.__dict__},
                       os.path.join(cfg.save_dir, "best.pt"))

    print("Training finished. Artifacts are stored in:", cfg.save_dir)


# ---------------------------
# Entry point / CLI
# ---------------------------

def parse_args() -> TrainConfig:
    import argparse
    p = argparse.ArgumentParser(description="Train Proposed Detector on format dataset")
    p.add_argument("--data_root", type=str, required=True, help="dataset root containing images/{train,val} and labels/{train,val}")
    p.add_argument("--img_size", type=int, default=640)
    p.add_argument("--num_classes", type=int, default=46)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, default="./runs/train_proposed")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        img_size=args.img_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        workers=args.workers,
        save_dir=args.save_dir,
        seed=args.seed
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
