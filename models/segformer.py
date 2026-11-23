"""
D7 SegFormer:纯 Transformer 分层主干(4 个 stage,分辨率逐级减半）+ 轻量 MLP 解码器。
输入: (B,3,256,256)  输出: (B,1,256,256)
结构:没有 CNN 卷积主干，图像一开始就被 PatchEmbed 切成 token;
多尺度特征在 4 个 stage 上分别提取，解码器把它们对齐到同一尺寸融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU
from models.trans_blocks import PatchEmbed, TransformerBlock


class Stage(nn.Module):
    """一个 Transformer stage:PatchEmbed 切 token -> 若干 TransformerBlock -> 还原特征图。"""

    def __init__(self, in_ch: int, dim: int, num_heads: int, depth: int,
                 patch_size: int, stride: int, sr_ratio: int):
        super().__init__()
        self.embed = PatchEmbed(in_ch, dim, patch_size=patch_size, stride=stride)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio=4, sr_ratio=sr_ratio)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, H, W = self.embed(x)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, H, W)
        return feat


class SegFormer(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 decoder_dim: int = 256):
        super().__init__()

        # 4 个分层 stage：通道 32 -> 64 -> 160 -> 256，分辨率 128 -> 64 -> 32 -> 16
        self.stage1 = Stage(in_channels, 32, num_heads=2, depth=2, patch_size=7, stride=2, sr_ratio=8)
        self.stage2 = Stage(32, 64, num_heads=2, depth=2, patch_size=3, stride=2, sr_ratio=4)
        self.stage3 = Stage(64, 160, num_heads=5, depth=2, patch_size=3, stride=2, sr_ratio=2)
        self.stage4 = Stage(160, 256, num_heads=8, depth=2, patch_size=3, stride=2, sr_ratio=1)

        # 解码器：每个 stage 特征先 1x1 卷积对齐到 decoder_dim，再统一上采样融合
        self.linear_c1 = nn.Conv2d(32, decoder_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(64, decoder_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(160, decoder_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(256, decoder_dim, kernel_size=1)

        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim * 4, decoder_dim),
            ConvBNReLU(decoder_dim, decoder_dim),
        )
        self.head = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 分层主干，取出 4 个尺度特征
        f1 = self.stage1(x)   # (B, 32, 128, 128)
        f2 = self.stage2(f1)  # (B, 64, 64, 64)
        f3 = self.stage3(f2)  # (B, 160, 32, 32)
        f4 = self.stage4(f3)  # (B, 256, 16, 16)

        # 2) 解码器：1x1 对齐通道 -> 全部上采样到 f2 的尺寸(64x64) -> 通道拼接 -> 融合
        # 对齐四个特征并上采样到同一尺寸，再拼接
        c1 = F.interpolate(self.linear_c1(f1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        c2 = self.linear_c2(f2)
        c3 = F.interpolate(self.linear_c3(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        c4 = F.interpolate(self.linear_c4(f4), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        cat = torch.cat([c1, c2, c3, c4], dim=1)

        fuse = self.fuse(cat)
        #融合特征上采样回原图 256x256，再过输出头
        fuse = F.interpolate(fuse, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(fuse)
