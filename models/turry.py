# coding=utf-8
# @Time      :2025/3/1 下午10:48
# @Author    :FRE量子计算机

import torch
import torch.nn as nn
from models.image_tokenize import ImageTokenize
from models.text_tokenize import TextTokenize
from models.transformer import MultiModalTransformer


class Turry(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_tokenize = ImageTokenize()
        self.text_tokenize = TextTokenize()
        self.transformer = MultiModalTransformer()

    def forward(self, image, text):
        image_token = self.image_tokenize(image)
        text_token = self.text_tokenize(text)
        x = self.transformer(image_token, text_token, xxx)
        return x


if __name__ == "__main__":
    turry = Turry()
    print(turry(torch.randn(2, 3, 224, 224)))
