import numpy as np
from decord import VideoReader, cpu
from PIL import Image


class VideoLoader:
    """
    VideoLoader based on decord + PIL

    功能：
    - 加载视频
    - 支持多种采样策略
    - 输出 PIL.Image list
    """

    def __init__(
        self,
        video_path,
        ctx=cpu(0)
    ):
        """
        Args:
            video_path (str): 视频路径
            ctx: decord context, 默认 cpu(0)
        """
        self.video_path = video_path
        self.vr = VideoReader(video_path, ctx=ctx)

        self.num_frames = len(self.vr)
        self.fps = self.vr.get_avg_fps()
        self.duration = self.num_frames / self.fps

    # ------------------------------------------------
    # 基本信息
    # ------------------------------------------------
    def __len__(self):
        return self.num_frames

    def get_fps(self):
        return self.fps

    def get_duration(self):
        return self.duration

    # ------------------------------------------------
    # 核心接口：按帧索引加载
    # ------------------------------------------------
    def get_frames_by_indices(self, indices):
        """
        根据帧索引加载帧

        Args:
            indices (List[int] | np.ndarray)

        Returns:
            List[PIL.Image]
        """
        indices = np.clip(indices, 0, self.num_frames - 1)
        frames = self.vr.get_batch(indices).asnumpy()  # (N, H, W, 3), RGB
        images = [Image.fromarray(frame) for frame in frames]
        return images

    # ------------------------------------------------
    # 采样策略 1：均匀采样 N 帧（最常用）
    # ------------------------------------------------
    def sample_uniform(self, num_frames):
        """
        均匀采样 num_frames 帧
        """
        if num_frames >= self.num_frames:
            indices = np.arange(self.num_frames)
        else:
            indices = np.linspace(
                0, self.num_frames - 1, num_frames
            ).astype(int)
        return self.get_frames_by_indices(indices)

    # ------------------------------------------------
    # 采样策略 2：按 FPS 采样（如 1 fps）
    # ------------------------------------------------
    def sample_by_fps(self, target_fps):
        """
        按目标 fps 采样

        Args:
            target_fps (float): 目标帧率，如 1.0
        """
        step = max(int(self.fps / target_fps), 1)
        indices = np.arange(0, self.num_frames, step)
        return self.get_frames_by_indices(indices)

    # ------------------------------------------------
    # 采样策略 3：时间区间采样
    # ------------------------------------------------
    def sample_time_range(self, start_sec, end_sec, num_frames):
        """
        在指定时间区间内均匀采样

        Args:
            start_sec (float)
            end_sec (float)
            num_frames (int)
        """
        start_frame = int(start_sec * self.fps)
        end_frame = int(end_sec * self.fps)
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, self.num_frames - 1)

        indices = np.linspace(
            start_frame, end_frame, num_frames
        ).astype(int)

        return self.get_frames_by_indices(indices)

    # ------------------------------------------------
    # 采样策略 4：帧间隔采样
    # ------------------------------------------------
    def sample_by_interval(self, interval=15):
        """
        按照指定的帧间隔采样

        Args:
            interval (int): 每隔 interval 帧取一帧
                            例如 interval=30 ≈ 1fps (当原视频≈30fps)
        """
        if interval <= 0:
            raise ValueError("interval must be a positive integer")

        indices = np.arange(0, self.num_frames, interval)
        return self.get_frames_by_indices(indices)

    # ------------------------------------------------
    # 采样策略 5：返回batch的帧间隔采样
    # ------------------------------------------------
    def iter_frames_by_interval(self, interval, batch_size):
        """
        按帧间隔采样，并以 batch 形式 yield PIL.Image list

        Args:
            interval (int)
            batch_size (int)
        """
        if interval <= 0:
            raise ValueError("interval must be positive")

        indices = np.arange(0, self.num_frames, interval)

        for batch_indices in chunk_indices(indices, batch_size):
            images = self.get_frames_by_indices(batch_indices)
            yield images


# 一个通用的 batch 切分器
def chunk_indices(indices, batch_size):
    for i in range(0, len(indices), batch_size):
        yield indices[i:i + batch_size]
