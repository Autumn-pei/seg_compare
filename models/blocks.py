"""
D3 共用积木块:ConvBNReLU / Down / Up。
UNet、UNet++、Attention UNet 复用
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    一个基础卷积块:Conv2d -> BatchNorm2d -> ReLU。
    常用 3x3 卷积,padding=1 保持尺寸不变。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x


class Down(nn.Module):
    """下采样:2x2 最大池化，尺寸减半，通道数不变。"""

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class Up(nn.Module):
    """
    上采样：转置卷积把尺寸放大一倍，通道减半。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

#测试：
if __name__ == "__main__":
    # 造一个假输入：batch=2, channel=3, H=256, W=256
    test_x = torch.randn(1, 3, 256, 256)

    block = ConvBNReLU(3, 64)
    y1 = block(test_x)
    print("ConvBNReLU输出形状:", y1.shape) # [2,64,256,256]，高宽不变

    pool = Down()
    y2 = pool(y1)
    print("Down池化后形状:", y2.shape) # [2,64,128,128]，宽高减半

    up_layer = Up(64, 32)
    y3 = up_layer(y2)
    print("Up上采样后形状:", y3.shape) # [2,32,256,256]，宽高放大回去
