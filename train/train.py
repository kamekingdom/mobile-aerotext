"""
minimal from-scratch implementation (PyTorch ≥2.1)
===========================================================

Major components implemented
---------------------------
* ConvBnAct          – unified conv + BN + SiLU
* RepConvBnAct       – re-parameterisable conv (train) ↔ single conv (inference)
* RepNBottleneck     – DetectModel core residual unit without depth-wise split
* RepNCSP            – Cross-Stage-Partial variant using RepNBottleneck
* ELANBlock          – 4-way ELAN aggregation block (as in GELAN)
* GELANBackbone      – Stage-stacked backbone producing P3/P4/P5 feature maps
* PANRNeck           – Bidirectional FPN/PAN with reparameterised convs
* DetectHeadAF       – Anchor-free decoupled head for cls/obj/box regression

Output tensor shape for stride {8,16,32} is (N, 5+nc, H, W)
where 5 = (cx, cy, w, h, objectness).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Utility layers
# -----------------------------------------------------------------------------

def autopad(k: int, p: int | None = None) -> int:
    """Padding to keep same spatial dim when stride = 1."""
    return k // 2 if p is None else p


class ConvBnAct(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, g: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConvBnAct(nn.Module):
    """Train-time 3×3 + 1×1 conv branches that merge into a single conv for inference."""

    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1):
        super().__init__()
        self.training_branch = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, autopad(k), bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(inplace=True),
        )
        self.id_branch = (
            None
            if c_in != c_out or s != 1
            else nn.Sequential(
                nn.BatchNorm2d(c_in)
            )
        )
        # Placeholder for merged conv
        self.fused = None

    def forward(self, x):
        if self.fused is not None:
            return self.fused(x)
        y = self.training_branch(x)
        if self.id_branch is not None:
            y = y + self.id_branch(x)
        return F.silu(y, inplace=True)

    # --- fusion utils ---------------------------------------------------------
    def fuse(self):
        """Fuse the train branches into a single conv for inference."""
        if self.fused is not None:
            return  # already fused
        # get parameters
        conv3, bn3, _ = self.training_branch
        w3 = bn3.weight / torch.sqrt(bn3.running_var + bn3.eps)
        b3 = bn3.bias - bn3.running_mean * w3
        w3 = w3.reshape(-1, 1, 1, 1) * conv3.weight
        b3 = b3 + (conv3.bias or 0)

        if self.id_branch is not None:
            bn1 = self.id_branch[0]
            w_id = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)
            b_id = bn1.bias - bn1.running_mean * w_id
            w_id = w_id.reshape(-1, 1, 1, 1)
            # create identity kernel
            id_kernel = torch.zeros_like(w3)
            c = w3.size(1)
            for i in range(c):
                id_kernel[i, i, 1, 1] = 1.0
            w3 += w_id * id_kernel
            b3 += b_id
        fused_conv = nn.Conv2d(conv3.in_channels, conv3.out_channels,
                               kernel_size=3, stride=conv3.stride,
                               padding=autopad(3), bias=True)
        fused_conv.weight.data.copy_(w3)
        fused_conv.bias.data.copy_(b3)
        self.fused = nn.Sequential(fused_conv, nn.SiLU(inplace=True))
        # free memory
        del self.training_branch, self.id_branch


# -----------------------------------------------------------------------------
# Core building blocks
# -----------------------------------------------------------------------------
class RepNBottleneck(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.cv1 = RepConvBnAct(c, c // 2, k=1)
        self.cv2 = RepConvBnAct(c // 2, c)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class RepNCSP(nn.Module):
    """Cross-Stage Partial structure with RepN bottlenecks."""

    def __init__(self, c: int, n: int = 3):
        super().__init__()
        c_ = int(c * 0.5)
        self.part1 = nn.Conv2d(c, c_, 1, 1, bias=False)
        self.part2 = nn.Sequential(
            *[RepNBottleneck(c_) for _ in range(n)],
            nn.Conv2d(c_, c_, 1, 1, bias=False),
        )
        self.concat_conv = nn.Conv2d(c_, c, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y1 = self.part1(x)
        y2 = self.part2(x)
        y = torch.cat((y1, y2), 1)
        y = self.concat_conv(y)
        y = self.bn(y)
        return self.act(y)


class ELANBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, n: int = 4):
        super().__init__()
        self.cv1 = nn.Conv2d(c_in, c_out, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c_in, c_out, 1, 1, bias=False)
        blocks = []
        for _ in range(n):
            blocks.append(RepNCSP(c_out, n=1))
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(c_out * (n + 2), c_out, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = [self.cv1(x), self.cv2(x)]
        for i, blk in enumerate(self.blocks):
            y.append(blk(y[-1]))
        out = torch.cat(y, 1)
        out = self.out_conv(out)
        out = self.bn(out)
        return self.act(out)


# -----------------------------------------------------------------------------
# Backbone + Neck
# -----------------------------------------------------------------------------
class GELANBackbone(nn.Module):
    """Produces P3 (stride 8), P4 (16), P5 (32)"""

    def __init__(self, ch: int = 3, width=1.0, depth=1.0):
        super().__init__()
        def c(n):
            return int(n * width)

        def d(n):
            return max(round(n * depth), 1)

        self.stem = ConvBnAct(ch, c(32), 3, 2)
        # Stage 1 (stride 4)
        self.stage1 = nn.Sequential(
            ConvBnAct(c(32), c(64), 3, 2),
            ELANBlock(c(64), c(64), n=d(1)),
        )
        # Stage 2 (stride 8) – P3 out
        self.stage2 = nn.Sequential(
            ConvBnAct(c(64), c(128), 3, 2),
            ELANBlock(c(128), c(128), n=d(2)),
        )
        # Stage 3 (stride 16) – P4 out
        self.stage3 = nn.Sequential(
            ConvBnAct(c(128), c(256), 3, 2),
            ELANBlock(c(256), c(256), n=d(3)),
        )
        # Stage 4 (stride 32) – P5 out
        self.stage4 = nn.Sequential(
            ConvBnAct(c(256), c(512), 3, 2),
            ELANBlock(c(512), c(512), n=d(1)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class PANRNeck(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        def c(n):
            return int(n * width)

        # top-down
        self.cv5 = ConvBnAct(c(512), c(256), 1, 1)
        self.cv4 = ConvBnAct(c(256), c(128), 1, 1)

        # final convs
        self.p3_out = ConvBnAct(c(256), c(128), 3, 1)
        self.p4_out = ConvBnAct(c(384), c(256), 3, 1)
        self.p5_out = ConvBnAct(c(512), c(512), 3, 1)

    def forward(self, p3, p4, p5):
        u4 = F.interpolate(self.cv5(p5), scale_factor=2, mode="nearest")
        p4 = torch.cat((u4, p4), 1)  # 512→256+256 = 512
        u3 = F.interpolate(self.cv4(p4), scale_factor=2, mode="nearest")
        p3 = torch.cat((u3, p3), 1)

        p3 = self.p3_out(p3)
        p4 = self.p4_out(p4)
        p5 = self.p5_out(p5)
        return p3, p4, p5


# -----------------------------------------------------------------------------
# Detection head (anchor-free, decoupled)
# -----------------------------------------------------------------------------
class DetectHeadAF(nn.Module):
    def __init__(self, nc: int = 80, width=1.0):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.strides = torch.tensor([8, 16, 32])
        act_ch = [128, 256, 512]
        self.cls_convs = nn.ModuleList([ConvBnAct(c, c, 3, 1) for c in act_ch])
        self.reg_convs = nn.ModuleList([ConvBnAct(c, c, 3, 1) for c in act_ch])
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, nc, 1, 1) for c in act_ch])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, 4, 1, 1) for c in act_ch])
        self.obj_preds = nn.ModuleList([nn.Conv2d(c, 1, 1, 1) for c in act_ch])

    def forward(self, feats: List[torch.Tensor]):
        out = []
        for i, f in enumerate(feats):
            cls = self.cls_preds[i](self.cls_convs[i](f))
            box = self.reg_preds[i](self.reg_convs[i](f))
            obj = self.obj_preds[i](f)
            y = torch.cat((box, obj, cls), 1)
            out.append(y)
        return out  # list of feature maps


# -----------------------------------------------------------------------------
# Detection Model wrapper
# -----------------------------------------------------------------------------
class DetectModel(nn.Module):
    def __init__(self, nc: int = 80, width: float = 1.0, depth: float = 1.0):
        super().__init__()
        self.backbone = GELANBackbone(width=width, depth=depth)
        self.neck = PANRNeck(width=width)
        self.head = DetectHeadAF(nc=nc, width=width)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        return self.head([p3, p4, p5])


# -----------------------------------------------------------------------------
# Build helper
# -----------------------------------------------------------------------------

def build_model(variant: str = "c", nc: int = 80) -> DetectModel:
    presets = {
        "t": dict(width=0.375, depth=0.33),
        "s": dict(width=0.5, depth=0.50),
        "m": dict(width=0.75, depth=0.75),
        "c": dict(width=1.0, depth=1.0),
        "e": dict(width=1.25, depth=1.20),
    }
    assert variant in presets, f"unknown variant {variant}"
    return DetectModel(nc=nc, **presets[variant])


if __name__ == "__main__":
    # quick sanity check
    net = build_model("c", nc=80)
    net.eval()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ys = net(x)
    for i, y in enumerate(ys):
        print(f"P{i+3}", y.shape)