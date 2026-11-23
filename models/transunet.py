"""
D7 TransUNet:CNN 编码器提特征 -> Transformer 瓶颈建模全局依赖 -> 上采样解码器跳连回原图。
输入: (B,3,256,256)  输出: (B,1,256,256)
结构：前半段像 UNet 编码器；瓶颈处把 16x16 特征图展平成 token 过 Transformer;
      后半段解码器和 UNet 一样靠跳跃连接补细节。
"""

import torch
import torch.nn as nn

from models.blocks import ConvBNReLU, Down, Up


class TransUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 dim: int = 256, depth: int = 4, num_heads: int = 4):
        super().__init__()

        # CNN 编码器：逐级下采样，保留跳跃特征
        self.enc1 = ConvBNReLU(in_channels, 64)              # (B, 64, 256, 256)
        self.down1 = Down()
        self.enc2 = ConvBNReLU(64, 128)                      # (B, 128, 128, 128)
        self.down2 = Down()
        self.enc3 = ConvBNReLU(128, 256)                     # (B, 256, 64, 64)
        self.down3 = Down()
        self.enc4 = ConvBNReLU(256, 512)                     # (B, 512, 32, 32)
        self.down4 = Down()
        self.bottleneck = ConvBNReLU(512, 512)               # (B, 512, 16, 16)

        # Transformer 瓶颈 
        self.proj = nn.Conv2d(512, dim, kernel_size=1)
        # 16x16 = 256 个 token；位置编码可学习
        self.pos_embed = nn.Parameter(torch.zeros(1, 16 * 16, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 2,
            batch_first=True, norm_first=True, dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        #解码器：上采样 + 跳跃连接拼接
        self.up1 = Up(dim, 256)
        self.dec1 = ConvBNReLU(256 + 512, 256)
        self.up2 = Up(256, 128)
        self.dec2 = ConvBNReLU(128 + 256, 128)
        self.up3 = Up(128, 64)
        self.dec3 = ConvBNReLU(64 + 128, 64)
        self.up4 = Up(64, 32)
        self.dec4 = ConvBNReLU(32 + 64, 32)

        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) CNN 编码器，保存跳跃特征
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))          # (B, 512, 16, 16)

        # 2) Transformer 瓶颈：特征图 -> token 序列 -> 加位置编码 -> Transformer -> 还原特征图
        t = self.proj(b)                              # (B, dim, 16, 16)
        B, C, H, W = t.shape
        # 完成 token 化
        tokens = t.flatten(2).transpose(1, 2)         # (B, H*W, C)
        # 加位置编码
        tokens = tokens+self.pos_embed
        # 过 encoder
        tokens = self.encoder(tokens)
        # 再 reshape 回特征图
        t = tokens.transpose(1, 2).reshape(B, C, H, W)

        # 3) 解码器：上采样 -> 与对应跳跃特征在通道维拼接 -> 卷积融合
        #补全 4 级上采样 + cat + 卷积
        d1 = self.dec1(torch.cat([self.up1(t), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))


        return self.head(d4)
