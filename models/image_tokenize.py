# coding=utf-8
# @Time      :2025/3/1 下午7:47
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class ImageTokenize(nn.Module):
    def __init__(self, img_size=224, in_chans=3, embed_dim=768, num_tokens=64):
        super().__init__()
        # 1. 卷积提取特征图（生成num_tokens个通道）
        self.cnn = nn.Sequential(
            nn.Conv2d(in_chans, num_tokens, kernel_size=16, stride=16),  # 降维至14x14
            nn.GELU(),
            nn.Conv2d(num_tokens, num_tokens, kernel_size=1),  # 通道调整
        )
        # 2. 展平为图像Token
        self.proj = nn.Linear(num_tokens, embed_dim)  # 投影到与文本相同的维度
        # 3. 图像位置编码
        self.img_pos_embed = nn.Parameter(torch.randn(1, (img_size//16)**2, embed_dim))

    def forward(self, x):
        # 输入x: [B, 3, 224, 224]
        x = self.cnn(x)  # [B, num_tokens, 14, 14]
        x = x.flatten(2).permute(0, 2, 1)  # [B, 196, num_tokens]
        x = self.proj(x)  # [B, 196, embed_dim]
        x = x + self.img_pos_embed  # 添加位置编码
        return x  # 输出: [B, 196, 768]


if __name__ == "__main__":
    pass
