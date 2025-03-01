# coding=utf-8
# @Time      :2025/3/1 下午7:42
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


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
        img_tokens += self.modal_embed(torch.zeros(img_tokens.size(0), dtype=int))
        text_tokens += self.modal_embed(torch.ones(text_tokens.size(0), dtype=int))
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
