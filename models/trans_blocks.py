"""
D7 Transformer 共用积木块:PatchEmbed / Mlp / Attention / TransformerBlock
TransUNet、SegFormer 复用；
- 图像特征图: (B, C, H, W)
- token 序列: (B, N, C),N = H*W
"""

import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Transformer 前馈网络:Linear -> GELU -> Dropout -> Linear -> Dropout。"""

    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_ch, out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Attention(nn.Module):
    """
    多头自注意力。sr_ratio>1 时先对特征图做空间缩减(SegFormer 的高效注意力），
    降低 K/V 序列长度;sr_ratio=1 就是标准全注意力(TransUNet 瓶颈用）。
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """一个 Transformer 编码块:LN -> 自注意力 -> 残差 -> LN -> MLP -> 残差。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4,
                 sr_ratio: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, sr_ratio, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * mlp_ratio, dim, dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """
    重叠块嵌入：用带 stride 的卷积把特征图切成 patch,展平成 token 序列。
    返回 (tokens, H, W)。
    """

    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 7, stride: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W
