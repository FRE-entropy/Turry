# coding=utf-8
# @Time      :2025/3/1 下午7:47
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class ImageTokenize(nn.Module):
    def __init__(self, in_dim=128, out_dim=768):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_dim, in_dim // 4),
            nn.ReLU(),
            nn.Linear(in_dim // 4, in_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(in_dim, out_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 14 * 14, out_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        # 通道注意力压缩
        attn = self.se(x).view(B, C, 1)  # [B,C,1]
        x = x.permute(0, 2, 3, 1) * attn  # [B,H,W,C]
        # 投影并添加位置编码
        x = self.proj(x).view(B, H * W, -1) + self.pos_embed
        return x  # [B,196,768]


if __name__ == "__main__":
    pass
