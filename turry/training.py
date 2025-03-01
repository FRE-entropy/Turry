# coding=utf-8
# @Time      :2025/3/1 下午8:01
# @Author    :FRE量子计算机

# coding=utf-8
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from models import Turry
from transformers import BertTokenizer
from


# ---------------------
# 配置参数
# ---------------------
class Config:
    # 数据参数
    train_data_path = "data/coco/train"
    val_data_path = "data/coco/val"
    max_text_len = 50
    image_size = 224

    # 模型参数
    embed_dim = 768
    num_heads = 12
    num_layers = 6
    vocab_size = 30522  # 与BERT base一致

    # 训练参数
    batch_size = 32
    lr = 3e-5
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "checkpoints/"


config = Config()


# ---------------------
# 数据集类
# ---------------------
class CocoDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = [...]  # 加载COCO标注数据
        self.tokenizer = tokenizer
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config.image_size, config.image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]

        # 图像处理
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # 文本处理
        tokens = self.tokenizer.encode(
            caption,
            max_length=config.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return image, tokens.squeeze(0)


# ---------------------
# 训练模块
# ---------------------
def train():
    # 初始化
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = CocoDataset(config.train_data_path, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    model = Turry().to(config.device)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 训练循环
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, texts) in enumerate(train_loader):
            images = images.to(config.device)
            texts = texts.to(config.device)

            # 准备输入输出
            src_text = texts[:, :-1]  # 输入序列
            tgt_text = texts[:, 1:]  # 目标序列

            # 前向传播
            outputs = model(images, src_text)

            # 计算损失
            loss = criterion(
                outputs.view(-1, config.vocab_size),
                tgt_text.contiguous().view(-1)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        # 保存检查点
        avg_loss = total_loss / len(train_loader)
        torch.save(model.state_dict(), f"{config.save_dir}/model_epoch{epoch}.pt")
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f}")


# ---------------------
# 验证模块
# ---------------------
def validate(model, val_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(config.device)
            texts = texts.to(config.device)

            src_text = texts[:, :-1]
            tgt_text = texts[:, 1:]

            outputs = model(images, src_text)
            loss = criterion(
                outputs.view(-1, config.vocab_size),
                tgt_text.contiguous().view(-1)
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


if __name__ == "__main__":
    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 启动训练
    train()

    # 可选：加载最佳模型进行验证
    # model.load_state_dict(torch.load("checkpoints/model_best.pt"))
    # val_dataset = CocoDataset(config.val_data_path, tokenizer)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    # validate(model, val_loader)
