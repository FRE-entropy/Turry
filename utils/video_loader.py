# coding=utf-8
# @Time      :2025/3/1 下午1:41
# @Author    :FRE量子计算机

from decord import VideoReader, cpu
import numpy as np
import cv2


def load_video(video_path, width=-1, height=-1):
    # 使用DECORD高效读取视频
    video = VideoReader(video_path, width=width, height=height, ctx=cpu(0))
    return video


def video_to_frames(video, frame_num, start=0, end=None):
    # 均匀采样帧
    frame_indices = np.linspace(
        start, end if end else len(video) - 1, num=frame_num, dtype=int
    )
    frames = video.get_batch(list(frame_indices)).asnumpy()

    return frames


def play_video(frames, fps=30):
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 显示当前帧
        cv2.imshow("Video", frame)

        # 按帧率控制播放速度，按'q'退出
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    vr = load_video("../data/video/a.mp4")
    frames = video_to_frames(vr, 1000)
    play_video(frames)
