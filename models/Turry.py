# coding=utf-8
# @Time      :2025/3/1 下午10:48
# @Author    :FRE量子计算机

import torch
import torch.nn as nn
from models.cnn import CNN
from models.image_tokenize import ImageTokenize
from models.transformer import MultiModalTransformer


class Turry(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNN()
        self.image_tokenize = ImageTokenize()
        self.transformer = MultiModalTransformer()

    def forward(self, x):
        x = self.cnn(x)
        x = self.image_tokenize(x)
        x = self.transformer(x)
        return x


if __name__ == "__main__":
    turry = Turry()
    print(turry(torch.randn(2, 3, 224, 224)))
