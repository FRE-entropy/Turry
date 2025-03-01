### **模型架构设计图**
```
图像输入 → CNN特征提取 → 展平为图像Token → 线性投影 → 图像位置编码
文本输入 → 词嵌入 → 文本位置编码
        ↓
    [图像Token] + [文本Token] → 拼接 → Transformer编码器 → Transformer解码器 → 文本输出
```

---

### **1. 输入处理模块**

#### **(1) 图像分支**
```python
class ImageProcessor(nn.Module):
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
```

#### **(2) 文本分支**
```python
class TextProcessor(nn.Module):
    def __init__(self, vocab_size=30522, embed_dim=768, max_length=512):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))

    def forward(self, x):
        # 输入x: [B, seq_len] (文本ID序列)
        x = self.token_embed(x)  # [B, seq_len, 768]
        x = x + self.pos_embed[:, :x.size(1), :]  # 添加位置编码
        return x  # 输出: [B, seq_len, 768]
```

---

### **2. 多模态融合Transformer**

#### **(1) 编码器-解码器结构**
```python
class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, num_layers=6):
        super().__init__()
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
```

---

### **3. 训练与推理细节**

#### **(1) 输入拼接方式**
- **图像Token数量**：`196`（14x14网格）
- **文本Token数量**：可变长度（如50）
- **总输入长度**：`196 + 50 = 246`（需调整位置编码长度）

#### **(2) 训练目标**
- **任务类型**：文本生成（如Image Captioning、VQA）
- **损失函数**：交叉熵损失（文本生成）
```python
criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
```

#### **(3) 推理生成**
- **自回归生成**：使用Beam Search或Nucleus Sampling
```python
def generate(self, img, text, max_length=50):
    generated = []
    for _ in range(max_length):
        logits = self(img, text, generated)
        next_token = logits.argmax(-1)[:, -1]
        generated.append(next_token)
    return generated
```

---

### **4. 性能优化技巧**

#### **(1) 图像Token压缩**
- **策略**：在CNN后添加自适应池化，减少Token数量
```python
nn.AdaptiveAvgPool2d((7,7))  # 将14x14 → 7x7，Token数从196降至49
```

#### **(2) 跨模态注意力**
- **设计**：在编码器中区分图像和文本的注意力区域
```python
# 使用掩码限制图像和文本的注意力范围
img_mask = torch.ones(196, 196)
text_mask = torch.ones(50, 50)
combined_mask = torch.block_diag(img_mask, text_mask)
```

#### **(3) 预训练策略**
1. **单模态预训练**：
   - 图像分支：在ImageNet上预训练CNN
   - 文本分支：加载BERT预训练权重
2. **多模态微调**：在目标任务（如COCO Caption）上联合训练

---

### **5. 模型变体扩展**

#### **(1) 轻量级版本**
```python
# 使用Tiny架构
embed_dim = 256
num_heads = 4
num_layers = 3
```

#### **(2) 多任务学习**
- **共享编码器**：同时支持VQA、Captioning、文本分类
- **任务标识符**：在输入中添加`[TASK_ID]` Token

#### **(3) 跨语言支持**
- **多语言词嵌入**：替换文本分支为多语言BERT
- **对齐损失**：约束图像和不同语言文本的嵌入空间一致性

---

### **总结**
该设计通过以下关键创新点实现多模态文本生成：
1. **动态Token拼接**：灵活处理可变长度的图像和文本输入。
2. **位置感知投影**：保留图像空间结构和文本序列位置。
3. **分层注意力**：编码器捕捉跨模态交互，解码器专注文本生成。  
建议在具体任务中调整图像Token数量和投影维度，平衡模型性能和计算成本。