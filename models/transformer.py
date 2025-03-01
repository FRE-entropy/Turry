# coding=utf-8
# @Time      :2025/3/1 下午2:51
# @Author    :FRE量子计算机

import torch
import torch.nn as nn
from torchvision import models, transforms
from decord import VideoReader, cpu
import numpy as np


def load_video_frames(video_path, config):
    # 使用DECORD高效读取视频
    vr = VideoReader(video_path, ctx=cpu(0))

    # 均匀采样帧
    frame_indices = np.linspace(
        0, len(vr) - 1, num=config.max_frames, dtype=int
    )
    frames = vr.get_batch(list(frame_indices)).asnumpy()

    # 转换为PyTorch Tensor并预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.frame_size, config.frame_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return torch.stack([transform(frame) for frame in frames])


# 2. 帧特征提取（使用预训练ResNet）
class FrameEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1],  # 去掉最后的全连接层
            nn.Flatten(),
            nn.Linear(512, config.feature_dim)
        )

    def forward(self, x):
        # x: (batch, frames, C, H, W)
        batch_size, num_frames = x.shape[:2]
        x = x.view(-1, *x.shape[2:])  # 合并batch和frame维度
        features = self.feature_extractor(x)
        return features.view(batch_size, num_frames, -1)


# 3. 时间Transformer模型
class VideoTransformer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.positional_encoding = PositionalEncoding(config.feature_dim, config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.feature_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)

        self.classifier = nn.Linear(config.feature_dim, num_classes)

    def forward(self, x):
        # x: (batch, frames, feature_dim)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        return self.classifier(x)


# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    pass
