"""
D5 UNet++（嵌套跳跃连接）。
输入: (B,3,256,256)  输出: (B,1,256,256)
通道: 64 -> 128 -> 256 -> 512, bottleneck 1024。
节点 x[i][j]：j=0 为编码器；j>0 融合 同层前节点 + 深层上采样。
"""

import torch
import torch.nn as nn

from models.blocks import Down, Up
from models.unet import DoubleConv


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        c = [64, 128, 256, 512]

        # ---- 编码器 x[i][0] 
        self.x00 = DoubleConv(in_channels, c[0])
        self.x10 = DoubleConv(c[0], c[1])
        self.x20 = DoubleConv(c[1], c[2])
        self.x30 = DoubleConv(c[2], c[3])
        self.x40 = DoubleConv(c[3], 1024)
        self.pool = Down()

        # ---- 上采样块：深层 -> 当前层，通道对齐到当前行 ----
        self.up10 = Up(c[1], c[0])
        self.up20 = Up(c[2], c[1])
        self.up30 = Up(c[3], c[2])
        self.up40 = Up(1024, c[3])
        self.up11 = Up(c[1], c[0])
        self.up21 = Up(c[2], c[1])
        self.up31 = Up(c[3], c[2])
        self.up12 = Up(c[1], c[0])
        self.up22 = Up(c[2], c[1])
        self.up13 = Up(c[1], c[0])

        # ---- 嵌套融合节点（输入 = 同行通道 x2，拼接而来）----
        self.x01 = DoubleConv(c[0] + c[0], c[0])
        self.x11 = DoubleConv(c[1] + c[1], c[1])
        self.x21 = DoubleConv(c[2] + c[2], c[2])
        self.x31 = DoubleConv(c[3] + c[3], c[3])
        self.x02 = DoubleConv(c[0] + c[0], c[0])
        self.x12 = DoubleConv(c[1] + c[1], c[1])
        self.x22 = DoubleConv(c[2] + c[2], c[2])
        self.x03 = DoubleConv(c[0] + c[0], c[0])
        self.x13 = DoubleConv(c[1] + c[1], c[1])
        self.x04 = DoubleConv(c[0] + c[0], c[0])

        self.head = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        # 第 1 列嵌套节点
        x01 = self.x01(torch.cat([x00, self.up10(x10)], dim=1))
        x11 = self.x11(torch.cat([x10, self.up20(x20)], dim=1))
        x21 = self.x21(torch.cat([x20, self.up30(x30)], dim=1))
        x31 = self.x31(torch.cat([x30, self.up40(x40)], dim=1))

        # 第 2 列
        x02 = self.x02(torch.cat([x01, self.up11(x11)], dim=1))
        x12 = self.x12(torch.cat([x11, self.up21(x21)], dim=1))
        x22 = self.x22(torch.cat([x21, self.up31(x31)], dim=1))

        # 第 3 列
        x03 = self.x03(torch.cat([x02, self.up12(x12)], dim=1))
        x13 = self.x13(torch.cat([x12, self.up22(x22)], dim=1))

        # 第 4 列（最深融合节点）
        x04 = self.x04(torch.cat([x03, self.up13(x13)], dim=1))

        return self.head(x04)
