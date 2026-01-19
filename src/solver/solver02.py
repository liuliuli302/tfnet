from dataclasses import dataclass
from typing import Optional
import os
from tqdm import tqdm
from src.config.config import BasicConfig
from src.models.frame_caption import LlavaFrameCaptioner, LlavaFrameCaptionerConfig
from src.dataset.video_summarization_dataset import VideoSummarizationDataset, VideoSummarizationDatasetConfig
from src.utils.video_loader import VideoLoader
from torch.utils.data import DataLoader
import json


@dataclass
class Solver02Config(BasicConfig):
    # 帧字幕配置的文件路径
    frame_caption_config_file: str
    # Summe和TVSum数据集配置文件路径
    summe_dataset_config_file: str
    tvsum_dataset_config_file: str
    # 帧字幕提取的提示语
    frame_caption_prompt: str
    # 帧字幕提取的文件的保存文件夹
    caption_save_dir: str
    # 具体的字幕保存json路径
    summe_caption_json_file: Optional[str] = None
    tvsum_caption_json_file: Optional[str] = None


class Solver02:
    """
    Solver02: 第二批次的实验
    1 使用llava-next模型提取数据集的帧字幕
    2 按照场景切分聚合帧字幕为场景字幕
    3 基于场景字幕获得base score
    4 获取若干来源的frame对scence的贡献程度
    """

    def __init__(
        self,
        solver_config: Solver02Config,
    ):
        self.solver_config = solver_config
        # 加载帧字幕提取模型配置文件
        self.frame_caption_config = LlavaFrameCaptionerConfig.load_config_from_file(
            self.solver_config.frame_caption_config_file)

        # 加载数据集配置文件
        self.summe_dataset_config = VideoSummarizationDatasetConfig.load_config_from_file(
            self.solver_config.summe_dataset_config_file)
        self.tvsum_dataset_config = VideoSummarizationDatasetConfig.load_config_from_file(
            self.solver_config.tvsum_dataset_config_file)

        # 初始化数据集
        self.summe_dataset = VideoSummarizationDataset(
            self.summe_dataset_config)
        self.tvsum_dataset = VideoSummarizationDataset(
            self.tvsum_dataset_config)

    def _load_frame_caption_model(self):
        self.frame_caption_model = LlavaFrameCaptioner(
            self.frame_caption_config)

    def _frame_caption(self):
        # 加载帧字幕提取模型
        self._load_frame_caption_model()

        # 检测需要得到的两个json是否存在，如果存在则跳过
        summe_caption_json_file = self.solver_config.summe_caption_json_file
        tvsum_caption_json_file = self.solver_config.tvsum_caption_json_file
        if os.path.exists(summe_caption_json_file) and os.path.exists(tvsum_caption_json_file):
            print("Frame captions already exist, skipping frame captioning.")
            return

        summe_caption_json = {}
        tvsum_caption_json = {}
        frame_caption_prompt = self.solver_config.frame_caption_prompt

        # 处理 SUMME数据集
        summe_dataloader = DataLoader(
            self.summe_dataset, batch_size=1, shuffle=False)

        for idx, item in tqdm(enumerate(summe_dataloader), desc="Processing SumMe dataset for frame caption...", total=len(summe_dataloader)):
            video_name = item['video_name'][0]
            video_path = item['video_path'][0]
            picks = item['picks'][0].tolist()

            summe_caption_json[video_name] = {}
            summe_caption_json[video_name]['picks'] = picks
            summe_caption_json[video_name]['captions'] = []

            video = VideoLoader(video_path)
            frames = video.get_frames_by_indices(picks)

            for frame in tqdm(frames, desc=f"Captioning frames for video {video_name}", total=len(frames)):
                caption = self.frame_caption_model.caption_image(
                    frame, prompt=frame_caption_prompt
                )
                summe_caption_json[video_name]['captions'].append(caption)

        # 保存SUMME的caption json
        os.makedirs(self.solver_config.caption_save_dir, exist_ok=True)
        with open(summe_caption_json_file, 'w') as f:
            json.dump(summe_caption_json, f, indent=4)

        # 处理 TVSum数据集
        tvsum_dataloader = DataLoader(
            self.tvsum_dataset, batch_size=1, shuffle=False)

        for idx, item in tqdm(enumerate(tvsum_dataloader), desc="Processing TVSum dataset for frame caption...", total=len(tvsum_dataloader)):
            video_name = item['video_name'][0]
            video_path = item['video_path'][0]
            picks = item['picks'][0].tolist()

            tvsum_caption_json[video_name] = {}
            tvsum_caption_json[video_name]['picks'] = picks
            tvsum_caption_json[video_name]['captions'] = []

            video = VideoLoader(video_path)
            frames = video.get_frames_by_indices(picks)

            for frame in tqdm(frames, desc=f"Captioning frames for video {video_name}", total=len(frames)):
                caption = self.frame_caption_model.caption_image(
                    frame, prompt=frame_caption_prompt
                )
                tvsum_caption_json[video_name]['captions'].append(caption)

        # 保存TVSum的caption json
        os.makedirs(self.solver_config.caption_save_dir, exist_ok=True)
        with open(tvsum_caption_json_file, 'w') as f:
            json.dump(tvsum_caption_json, f, indent=4)

    def run(self):
        # 进行帧字幕提取
        self._frame_caption()
