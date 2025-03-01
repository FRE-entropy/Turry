# coding=utf-8
# @Time      :2025/3/1 下午11:30
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class TextTokenize(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, max_length=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))

    def forward(self, x):
        # 输入x: [B, seq_len] (文本ID序列)
        x = self.token_embed(x)  # [B, seq_len, 768]
        x = x + self.pos_embed[:, :x.size(1), :]  # 添加位置编码
        return x  # 输出: [B, seq_len, 768]


if __name__ == "__main__":
    pass
