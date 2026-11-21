"""
D5 DeepLabV3+：轻量自定义主干 + ASPP 多尺度空洞卷积 + 低层特征解码器。
输入: (B,3,256,256)  输出: (B,1,256,256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks import ConvBNReLU, Down


class ASPP(nn.Module):
    """空洞空间金字塔池化：并行 1x1、不同膨胀率 3x3 空洞卷积 + 图像池化分支。"""

    def __init__(self, in_ch: int, out_ch: int = 256, rates=(1, 6, 12, 18)):
        super().__init__()
        self.b0 = ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0)
        # 空洞卷积：padding=rate 保持尺寸不变
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[1], dilation=rates[1]),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[2], dilation=rates[2]),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rates[3], dilation=rates[3]),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        # 图像级池化分支：全局平均池化 -> 1x1 卷积 -> 上采样回原尺寸
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0))
        self.project = nn.Conv2d(out_ch * 5, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0=self.b0(x)
        b1=self.b1(x)
        b2=self.b2(x)
        b3=self.b3(x)
        p=self.pool(x)

        #经过全局池化结果是1x1 上采样回到和输入一摸一样的高宽
        p = F.interpolate(p,size=x.shape[-2:],mode='bilinear',align_corners=False)
        #五个分支通道拼接
        out = torch.cat([b0,b1,b2,b3,p],dim=1)
        out=self.project(out)
        return out

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # 主干：逐级下采样，low 为 /4 低层特征，high 为 /16 高层特征
        self.layer1 = nn.Sequential(
            ConvBNReLU(in_channels, 64, stride=2), ConvBNReLU(64, 64))
        self.layer2 = nn.Sequential(
            Down(), ConvBNReLU(64, 128), ConvBNReLU(128, 128))   # low: /4
        self.layer3 = nn.Sequential(
            Down(), ConvBNReLU(128, 256), ConvBNReLU(256, 256))  # /8
        self.layer4 = nn.Sequential(
            Down(), ConvBNReLU(256, 512), ConvBNReLU(512, 512))  # high: /16

        self.aspp = ASPP(512, 256)
        self.low_proj = ConvBNReLU(128, 48, kernel_size=1, padding=0)
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, padding=1),
            ConvBNReLU(256, 256, padding=1),
        )
        self.head = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l1 = self.layer1(x)
        low = self.layer2(l1)          # (B,128,64,64)
        h = self.layer3(low)
        high = self.layer4(h)          # (B,512,16,16)

        asp = self.aspp(high)          # (B,256,16,16)

        # 1. asp 上采样 ×4，从16,16 → 64,64，和low一样大
        asp = F.interpolate(asp, size=low.shape[-2:], mode='bilinear', align_corners=False)
        # 2. low通道压缩
        low_feat = self.low_proj(low)
        # 通道拼接
        fuse = torch.cat([asp, low_feat], dim=1)
        # 3. 解码器融合
        dec = self.decoder(fuse)
        # 再上采样 ×4，回到原图256×256
        dec = F.interpolate(dec, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return self.head(dec)
