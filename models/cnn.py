# coding=utf-8
# @Time      :2025/3/1 下午7:41
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768, patch_size=16):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        output = self.conv(x)
        return output


if __name__ == "__main__":
    pass
