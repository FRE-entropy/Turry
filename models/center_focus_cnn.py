# coding=utf-8
# @Time      :2025/3/1 下午7:41
# @Author    :FRE量子计算机
class CenterFocusCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            GaussianSpatialAttention(112, 112, sigma=0.2, learnable=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            GaussianSpatialAttention(56, 56, sigma=0.3, learnable=True),
            ChannelGating(in_channels=128, reduction=4)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, x):
        x = self.conv1(x)  # [B,64,112,112]
        x = self.conv2(x)  # [B,128,56,56]
        x = self.adaptive_pool(x)  # [B,128,14,14]
        return x

if __name__ == "__main__":
    pass
