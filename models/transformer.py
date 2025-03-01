# coding=utf-8
# @Time      :2025/3/1 下午7:42
# @Author    :FRE量子计算机

import torch
import torch.nn as nn


class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=6, vocab_size=30522):
        super().__init__()
        self.modal_embed = nn.Embedding(2, 768)  # 0:图像, 1:文本
        # 编码器（处理图像+文本）
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=3072
            ),
            num_layers=num_layers
        )
        # 解码器（生成文本）
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=3072
            ),
            num_layers=num_layers
        )
        # 输出层
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, img_tokens, text_tokens, tgt_seq):
        """
        :param img_tokens:
        :param text_tokens:
        :param tgt_seq:
        :return:
        """
        # 添加模态嵌入
        img_tokens += self.modal_embed(torch.zeros(img_tokens.size(0), dtype=torch.int))
        text_tokens += self.modal_embed(torch.ones(text_tokens.size(0), dtype=torch.int))
        # 拼接图像和文本Token
        combined = torch.cat([img_tokens, text_tokens], dim=1)  # [B, 196+seq_len, 768]
        # 编码器处理
        memory = self.encoder(combined)
        # 解码器生成
        tgt_embed = self.text_processor(tgt_seq)
        output = self.decoder(
            tgt=tgt_embed,
            memory=memory
        )
        logits = self.output_layer(output)
        return logits  # 输出: [B, tgt_seq_len, vocab_size]


if __name__ == "__main__":
    pass
