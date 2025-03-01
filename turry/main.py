# coding=utf-8
# @Time      :2025/3/1 下午1:29
# @Author    :FRE量子计算机

from utils.video_loader import load_video


class Config:
    max_frames = 32  # 最大处理帧数
    frame_size = 224  # 帧尺寸
    feature_dim = 512  # 特征维度
    num_heads = 8
    num_layers = 4
    dropout = 0.1


# 使用示例
if __name__ == "__main__":
    config = Config()
    num_classes = 10  # 根据任务修改

    # 模拟输入（假设batch_size=2）
    dummy_video = torch.randn(2, config.max_frames, 3, config.frame_size, config.frame_size)

    # 模型组件
    encoder = FrameEncoder(config)
    model = VideoTransformer(config, num_classes)

    # 前向传播
    features = encoder(dummy_video)  # (2, 32, 512)
    output = model(features)  # (2, 10)

    print("Output shape:", output.shape)
