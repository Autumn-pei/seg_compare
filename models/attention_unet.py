"""
D5 Attention UNet:在跳跃连接上加注意力门(AG)。
输入: (B,3,256,256)  输出: (B,1,256,256)
"""

import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, Down, Up
from models.unet import DoubleConv


class AttentionGate(nn.Module):
    """
    注意力门：用门控 g 给跳跃特征 x 加权。
    F_g: 门控通道  F_x: 跳跃特征通道  F_int: 中间通道
    """

    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        # W_g：处理门控信号g，1×1卷积把通道映射到F_int
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        # W_x：处理跳跃特征x，1×1卷积映射到同样F_int通道，才能相加
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int),
        )
        # psi：生成注意力权重图alpha，输出1通道、sigmoid压到0~1
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 1. 分别把g和x投影到同一个中间通道空间
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 2. 两个特征相加，ReLU融合
        a = self.relu(g1 + x1)
        # 3. 生成0~1的注意力掩码alpha，和x长宽一模一样
        alpha = self.psi(a)
        # 4. 权重乘原始跳跃特征：重要区域保留，背景压低
        return x * alpha


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        c = [64, 128, 256, 512]

        # 编码器（同 UNet）
        self.enc1 = DoubleConv(in_channels, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.down1, self.down2, self.down3, self.down4 = Down(), Down(), Down(), Down()
        self.bottleneck = DoubleConv(c[3], 1024)

        # 解码器上采样
        self.up1 = Up(1024, c[3])
        self.up2 = Up(c[3], c[2])
        self.up3 = Up(c[2], c[1])
        self.up4 = Up(c[1], c[0])

        # 注意力门：(门控通道, 跳跃通道, 中间通道)
        self.ag1 = AttentionGate(c[3], c[3], c[3] // 2)
        self.ag2 = AttentionGate(c[2], c[2], c[2] // 2)
        self.ag3 = AttentionGate(c[1], c[1], c[1] // 2)
        self.ag4 = AttentionGate(c[0], c[0], c[0] // 2)

        # 解码器卷积（拼接后通道翻倍）
        self.dec1 = DoubleConv(c[3] * 2, c[3])
        self.dec2 = DoubleConv(c[2] * 2, c[2])
        self.dec3 = DoubleConv(c[1] * 2, c[1])
        self.dec4 = DoubleConv(c[0] * 2, c[0])

        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

       # 第1级解码
        g1 = self.up1(b)
        a1 = self.ag1(g1, e4)
        d1 = self.dec1(torch.cat([g1, a1], dim=1))

        # 第2级解码
        g2 = self.up2(d1)
        a2 = self.ag2(g2, e3)
        d2 = self.dec2(torch.cat([g2, a2], dim=1))

        # 第3级解码
        g3 = self.up3(d2)
        a3 = self.ag3(g3, e2)
        d3 = self.dec3(torch.cat([g3, a3], dim=1))

        # 第4级解码
        g4 = self.up4(d3)
        a4 = self.ag4(g4, e1)
        d4 = self.dec4(torch.cat([g4, a4], dim=1))

        return self.head(d4)