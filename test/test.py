# coding=utf-8
# @Time      :2025/3/1 下午8:12
# @Author    :FRE量子计算机

import numpy as np
import torch.nn as nn

data = np.array([
    [
        []
    ]
])

conv = nn.Conv2d(
    3, 10,
    kernel_size=3, stride=3
)

print(conv(data).flatten(2).permute(0, 2, 1))

if __name__ == "__main__":
    pass
