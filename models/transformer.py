# coding=utf-8
# @Time      :2025/3/1 下午7:42
# @Author    :FRE量子计算机

import torch
import torch.nn as nn
from models.cnn import CNN
from models.image_tokenize import ImageTokenize


class SpatialTokenTransformer(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=768, patch_size=16):
        super().__init__()
        # 初始化父类，确保当前类的所有父类都被正确初始化
        # 卷积层：将图像分割为patch并提取特征
        # nn.Conv2d是二维卷积层，用于处理图像数据
        # in_channels: 输入图像的通道数，默认为3（RGB图像）
        # hidden_dim: 卷积后输出的特征通道数，默认为768
        # kernel_size: 卷积核大小，这里设置为patch_size，即每个patch的大小
        # stride: 卷积步长，这里也设置为patch_size，使得每个patch不重叠
        self.conv = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        # 位置编码
        # nn.Parameter是可学习的参数，这里用于位置编码
        # torch.randn生成一个服从标准正态分布的随机张量
        # 1: 批量大小，(224//patch_size)**2: patch的数量，hidden_dim: 每个patch的特征维度
        self.position_embedding = nn.Parameter(
            torch.randn(1, (224//patch_size)**2, hidden_dim)
        )
        # Transformer编码器
        # nn.TransformerEncoder是Transformer的编码器部分
        # nn.TransformerEncoderLayer是Transformer编码器的一层
        # d_model: 输入和输出的特征维度，这里设置为hidden_dim
        # nhead: 多头注意力机制的头数，这里设置为8
        # num_layers: Transformer编码器的层数，这里设置为6
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )

    def forward(self, x):
        # 卷积提取特征并展平：[B, C, H, W] → [B, (H*W), D]
        features = self.conv(x).flatten(2).permute(0, 2, 1)  # [B, num_patches, D]
        # 添加位置编码
        features += self.position_embedding
        # Transformer处理
        output = self.transformer(features)
        return output


class MultiModalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=6
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=768, nhead=12),
            num_layers=6
        )
        self.modal_embed = nn.Embedding(2, 768)  # 0:图像, 1:文本
        self.gate = nn.Linear(768, 2)  # 门控权重

    def forward(self, img_tokens, text_tokens):
        # 添加模态嵌入
        img_tokens += self.modal_embed(torch.zeros(img_tokens.size(0), dtype=torch.int))
        text_tokens += self.modal_embed(torch.ones(text_tokens.size(0), dtype=torch.int))
        combined = torch.cat([img_tokens, text_tokens], dim=1)
        # 编码器处理（带门控）
        memory = self.encoder(combined)
        gate_weights = torch.softmax(self.gate(memory), dim=-1)
        memory = memory * gate_weights[:, :, 0].unsqueeze(-1)  # 图像门控
        # 解码器生成
        output = self.decoder(tgt=text_tokens, memory=memory)
        return output


if __name__ == "__main__":
    pass
