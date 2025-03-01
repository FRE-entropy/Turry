# coding=utf-8
# @Time      :2025/3/1 下午7:41
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class GaussianSpatialAttention(nn.Module):
    def __init__(self, height, width, sigma=0.5, learnable=False):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=learnable)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width)
        )
        dist = x**2 + y**2
        self.register_buffer('base_mask', torch.exp(-dist / (2 * sigma**2)))

    def forward(self, x):
        # x: [B, C, H, W]
        mask = self.base_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        return x * mask


class ChannelGating(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate_weights = self.gate(x)  # [B,C,1,1]
        # 对边缘区域进行通道剪枝（保留前k个通道）
        B, C, H, W = x.shape
        center = x[:, :, H // 4:-H // 4, W // 4:-W // 4]  # 中心区域
        edge_pruned = x * gate_weights
        return torch.cat([center, edge_pruned], dim=1)


class CenterFocusCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            GaussianSpatialAttention(112, 112, sigma=0.2, learnable=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            GaussianSpatialAttention(56, 56, sigma=0.3, learnable=True),
            ChannelGating(in_channels=128, reduction=4)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        x = self.conv1(x)  # [B,64,112,112]
        x = self.conv2(x)  # [B,128,56,56]
        x = self.adaptive_pool(x)  # [B,128,14,14]
        return x


if __name__ == "__main__":
    pass
