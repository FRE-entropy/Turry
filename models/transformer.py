# coding=utf-8
# @Time      :2025/3/1 下午7:42
# @Author    :FRE量子计算机

import torch
import torch.nn as nn
from text_tokenize import TextTokenize  # 导入文本处理模块


class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=6, vocab_size=30522):
        super().__init__()
        # 模态嵌入层：用于区分图像和文本模态（0=图像，1=文本）
        self.modal_embed = nn.Embedding(2, embed_dim)

        # 文本处理器（处理目标序列的嵌入和位置编码）
        self.text_processor = TextTokenize(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        # Transformer编码器（处理图像和文本的联合输入）
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=3072,
                batch_first=True  # 关键修复：启用batch_first
            ),
            num_layers=num_layers
        )

        # Transformer解码器（用于生成目标文本序列）
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=3072,
                batch_first=True  # 关键修复：启用batch_first
            ),
            num_layers=num_layers
        )

        # 输出层：将解码器的输出映射到词汇表大小
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, img_tokens, text_tokens, tgt_seq):
        """
        前向传播流程
        :param img_tokens: 图像特征张量，形状 [B, img_seq_len, embed_dim]
        :param text_tokens: 文本特征张量，形状 [B, text_seq_len, embed_dim]
        :param tgt_seq: 目标序列（解码器输入），形状 [B, tgt_seq_len]
        :return: 输出logits，形状 [B, tgt_seq_len, vocab_size]
        """
        # 确保模态嵌入的设备与输入一致
        device = img_tokens.device

        # 修复1：模态嵌入维度调整
        # 图像模态嵌入（ID=0），形状从 [B, embed_dim] -> [B, 1, embed_dim]
        img_modal_embed = self.modal_embed(
            torch.zeros(img_tokens.size(0), dtype=torch.int).to(device)
        ).unsqueeze(1)  # 增加维度
        img_tokens += img_modal_embed  # 广播到所有图像token

        # 文本模态嵌入（ID=1）
        text_modal_embed = self.modal_embed(
            torch.ones(text_tokens.size(0), dtype=torch.int).to(device)
        ).unsqueeze(1)
        text_tokens += text_modal_embed  # 广播到所有文本token

        # 拼接图像和文本Token：[B, img_seq_len + text_seq_len, embed_dim]
        combined = torch.cat([img_tokens, text_tokens], dim=1)

        # 编码器处理（无需调整维度，batch_first=True已启用）
        memory = self.encoder(combined)

        # 处理目标序列（通过text_processor生成嵌入）
        tgt_embed = self.text_processor(tgt_seq)  # [B, tgt_seq_len, embed_dim]

        # 解码器生成
        output = self.decoder(
            tgt=tgt_embed,  # 解码器输入 [B, tgt_seq_len, embed_dim]
            memory=memory  # 编码器输出 [B, src_seq_len, embed_dim]
        )

        # 映射到词汇表
        logits = self.output_layer(output)
        return logits


if __name__ == "__main__":
    # 示例用法
    model = MultiModalTransformer()
    img = torch.randn(2, 196, 768)  # [B=2, img_seq_len=196, embed_dim=768]
    text = torch.randn(2, 50, 768)  # [B=2, text_seq_len=50, embed_dim=768]
    tgt = torch.randint(0, 30522, (2, 100))  # [B=2, tgt_seq_len=100]
    logits = model(img, text, tgt)
    print(logits.shape)  # 正确输出: [2, 100, 30522]