"""
D3 基线 UNet。
输入: (B, 3, 256, 256)  输出: (B, 1, 256, 256)
通道: 64 -> 128 -> 256 -> 512, bottleneck 1024。
"""

import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, Down, Up


class DoubleConv(nn.Module):
    """UNet 里一个 stage 的两次 3x3 卷积：ConvBNReLU x2。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()

        # ---- 编码器：4 个 DoubleConv + 4 个 Down ----
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.down1 = Down()
        self.down2 = Down()
        self.down3 = Down()
        self.down4 = Down()

        # ---- 瓶颈 ----
        self.bottleneck = DoubleConv(512, 1024)

        # ---- 解码器：4 个 Up + 4 个 DoubleConv ----
        # Up 先把通道减半；拼接 skip 后通道翻倍，再进 DoubleConv 压回。
        self.up1 = Up(1024, 512)
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = Up(512, 256)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = Up(256, 128)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = Up(128, 64)
        self.dec4 = DoubleConv(128, 64)

        # ---- 输出头：1x1 卷积压到 num_classes 通道 ----
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 编码器：保存每层 skip，再下采样
        e1 = self.enc1(x)            # (B, 64, 256, 256)
        e2 = self.enc2(self.down1(e1))  # (B, 128, 128, 128)
        e3 = self.enc3(self.down2(e2))  # (B, 256, 64, 64)
        e4 = self.enc4(self.down3(e3))  # (B, 512, 32, 32)
        b = self.bottleneck(self.down4(e4))  # (B, 1024, 16, 16)

        # 2) 解码器：上采样 -> 与对应 skip 拼接 -> DoubleConv

        # 拼接用 torch.cat([上采样结果, skip], dim=1)，dim=1 是通道维。
        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1),e3],dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2),e2],dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3),e1],dim=1))


        # 3) 输出头
        return self.head(d4)
