import asyncio
from dataclasses import dataclass
from typing import Optional
import os
from tqdm import tqdm
from src.models.scene_caption import SceneSummaryCaptionSummarizer, SceneSummaryCaptionSummarizerConfig
from src.config.config import BasicConfig
from src.models.frame_caption import LlavaFrameCaptioner, LlavaFrameCaptionerConfig, BlipFrameCaptionerConfig, BlipFrameCaptioner
from src.dataset.video_summarization_dataset import VideoSummarizationDataset, VideoSummarizationDatasetConfig
from src.utils.video_loader import VideoLoader
from torch.utils.data import DataLoader
import json
import numpy as np
import logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


@dataclass
class Solver02Config(BasicConfig):
    # 帧字幕提取模块的的配置文件路径
    frame_caption_config_file: str
    # 场景字幕总结模块的配置文件路径
    scene_caption_config_file: str
    # Summe和TVSum数据集配置文件路径
    summe_dataset_config_file: str
    tvsum_dataset_config_file: str
    # 帧字幕提取的文件的保存文件夹
    frame_caption_save_dir: str
    # 场景字幕提取的文件的保存文件夹
    scene_caption_save_dir: str


class Solver02:
    """
    Solver02: 第二批次的实验
    1 使用blip模型提取数据集的帧字幕
    2 按照场景切分聚合帧字幕为场景字幕，使用gpt5
    3 基于场景字幕获得base score
    4 获取若干来源的frame对scene的贡献程度
    """

    def __init__(
        self,
        solver_config: Solver02Config,
    ):
        self.solver_config = solver_config
        # 加载帧字幕提取模型配置文件
        self.frame_caption_config = BlipFrameCaptionerConfig.load_config_from_file(
            self.solver_config.frame_caption_config_file)

        # 加载场景字幕总结模型配置文件
        self.scene_caption_config = SceneSummaryCaptionSummarizerConfig.load_config_from_file(
            self.solver_config.scene_caption_config_file
        )

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
        self.frame_caption_model = BlipFrameCaptioner(
            self.frame_caption_config)
        self.solver_config.frame_caption_save_dir = os.path.join(
            self.solver_config.frame_caption_save_dir,
            self.frame_caption_config.model_name
        )

    def _load_scene_caption_model(self):
        self.scene_caption_model = SceneSummaryCaptionSummarizer(
            self.scene_caption_config)
        self.scene_caption_config.scene_caption_save_dir = os.path.join(
            self.solver_config.scene_caption_save_dir,
            self.scene_caption_config.model_name
        )

    def _frame_caption(self):
        frame_caption_model_name = self.frame_caption_config.model_name

        summe_frame_caption_json_file = os.path.join(
            self.solver_config.frame_caption_save_dir,
            frame_caption_model_name,
            "summe_frame_captions.json"
        )
        self.summe_frame_caption_json_file = summe_frame_caption_json_file

        tvsum_frame_caption_json_file = os.path.join(
            self.solver_config.frame_caption_save_dir,
            frame_caption_model_name,
            "tvsum_frame_captions.json"
        )
        self.tvsum_frame_caption_json_file = tvsum_frame_caption_json_file

        # 如果已有结果则跳过
        if os.path.exists(self.summe_frame_caption_json_file) and os.path.exists(self.tvsum_frame_caption_json_file):
            logging.info(
                "Frame captions json already exist, skipping frame captioning.")
            return

        # 加载帧字幕提取模型
        self._load_frame_caption_model()

        summe_frame_caption_json_file = self.summe_frame_caption_json_file
        tvsum_frame_caption_json_file = self.tvsum_frame_caption_json_file

        summe_frame_caption_json = {}
        tvsum_frame_caption_json = {}

        if os.path.exists(summe_frame_caption_json_file):
            with open(summe_frame_caption_json_file, 'r') as f:
                summe_frame_caption_json = json.load(f)

        if os.path.exists(tvsum_frame_caption_json_file):
            with open(tvsum_frame_caption_json_file, 'r') as f:
                tvsum_frame_caption_json = json.load(f)

        # 处理 SUMME数据集
        summe_dataloader = DataLoader(
            self.summe_dataset, batch_size=1, shuffle=False)

        for idx, item in tqdm(enumerate(summe_dataloader), desc="Processing SumMe dataset for frame caption...", total=len(summe_dataloader)):

            video_name = item['video_name'][0]

            if video_name in summe_frame_caption_json.keys():
                print(
                    f"Frame captions for video {video_name} already exist, skipping...")
                continue

            video_path = item['video_path'][0]
            picks = item['picks'][0].tolist()

            summe_frame_caption_json[video_name] = {}
            summe_frame_caption_json[video_name]['picks'] = picks
            summe_frame_caption_json[video_name]['captions'] = []

            video = VideoLoader(video_path)
            frames = video.get_frames_by_indices(picks)

            for frame in tqdm(frames, desc=f"Captioning frames for video {video_name}", total=len(frames)):
                caption = self.frame_caption_model.caption_image(
                    frame, prompt=self.frame_caption_config.frame_caption_prompt
                )
                summe_frame_caption_json[video_name]['captions'].append(
                    caption)

            # 保存SUMME的caption json
            os.makedirs(self.solver_config.frame_caption_save_dir,
                        exist_ok=True)
            with open(summe_frame_caption_json_file, 'w') as f:
                json.dump(summe_frame_caption_json, f, indent=4)

        # 处理 TVSum数据集
        tvsum_dataloader = DataLoader(
            self.tvsum_dataset, batch_size=1, shuffle=False)

        for idx, item in tqdm(enumerate(tvsum_dataloader), desc="Processing TVSum dataset for frame caption...", total=len(tvsum_dataloader)):
            video_name = item['video_name'][0]

            if video_name in tvsum_frame_caption_json.keys():
                print(
                    f"Frame captions for video {video_name} already exist, skipping...")
                continue

            video_path = item['video_path'][0]
            picks = item['picks'][0].tolist()

            tvsum_frame_caption_json[video_name] = {}
            tvsum_frame_caption_json[video_name]['picks'] = picks
            tvsum_frame_caption_json[video_name]['captions'] = []

            video = VideoLoader(video_path)
            frames = video.get_frames_by_indices(picks)

            for frame in tqdm(frames, desc=f"Captioning frames for video {video_name}", total=len(frames)):
                caption = self.frame_caption_model.caption_image(
                    frame, prompt=self.frame_caption_config.frame_caption_prompt
                )
                tvsum_frame_caption_json[video_name]['captions'].append(
                    caption)

            # 保存TVSum的caption json
            os.makedirs(self.solver_config.frame_caption_save_dir,
                        exist_ok=True)
            with open(tvsum_frame_caption_json_file, 'w') as f:
                json.dump(tvsum_frame_caption_json, f, indent=4)

    def _scene_caption(self):
        scene_caption_model_name = self.scene_caption_config.model_name

        save_dir = os.path.join(
            self.solver_config.scene_caption_save_dir,
            scene_caption_model_name
        )

        summe_scene_caption_json_file = os.path.join(
            save_dir, "summe_scene_captions.json"
        )

        tvsum_scene_caption_json_file = os.path.join(
            save_dir,
            "tvsum_scene_captions.json"
        )

        if os.path.exists(summe_scene_caption_json_file) and os.path.exists(tvsum_scene_caption_json_file):
            logging.info(
                "Scene captions json already exist, skipping scene captioning.")
            return

        # 加载场景字幕总结模型
        self._load_scene_caption_model()

        # 定义场景字幕的保存路径
        os.makedirs(save_dir, exist_ok=True)

        # 加载输入的帧字幕文件
        summe_frame_caption_json_file = self.summe_frame_caption_json_file
        tvsum_frame_caption_json_file = self.tvsum_frame_caption_json_file

        # 检查是否存在，如果不存在，报错
        if not os.path.exists(summe_frame_caption_json_file) or not os.path.exists(tvsum_frame_caption_json_file):
            logging.info(
                "Frame captions json not found. Please run _frame_caption first.")
            return

        # 加载帧字幕结果文件
        with open(summe_frame_caption_json_file, 'r') as f:
            summe_frame_caption_json = json.load(f)
        with open(tvsum_frame_caption_json_file, 'r') as f:
            tvsum_frame_caption_json = json.load(f)

        # 准备场景字幕保存的 JSON 结构
        summe_scene_caption_json = {}
        tvsum_scene_caption_json = {}

        # 加载现有结果用于断点续写
        summe_scene_caption_json = json.load(open(summe_scene_caption_json_file, 'r')) if os.path.exists(
            summe_scene_caption_json_file) else {}
        tvsum_scene_caption_json = json.load(open(tvsum_scene_caption_json_file, 'r')) if os.path.exists(
            tvsum_scene_caption_json_file) else {}

        # 准备批量任务
        # task structure: (captions_list, picks_list) for batch API
        # meta structure: (dataset_name, video_name, scene_index) for result distribution
        batch_tasks = []
        batch_meta = []  # tuple(dataset_key, video_name, scene_idx)

        # 构建查找表以快速获取 Change Points
        # dataset.data_list values 包含 'video_name' 和 'change_points'
        cps_map = {
            'summe': {v['video_name']: v['change_points'] for v in self.summe_dataset.data_list.values()},
            'tvsum': {v['video_name']: v['change_points'] for v in self.tvsum_dataset.data_list.values()}
        }

        # 遍历两个数据集准备任务
        datasets_to_process = [
            ('summe', summe_frame_caption_json, summe_scene_caption_json),
            ('tvsum', tvsum_frame_caption_json, tvsum_scene_caption_json)
        ]

        logging.info("Preparing batch tasks...")
        # DEBUG: 设置一个小的限制用于验证
        # debug_limit = 5

        for ds_name, frame_data, scene_json in datasets_to_process:
            for video_name, data in frame_data.items():
                if video_name in scene_json:
                    continue

                cps = cps_map[ds_name].get(video_name)
                if cps is None:
                    continue

                # Ensure cps is list-like
                if isinstance(cps, np.ndarray):
                    cps = cps.tolist()

                frame_captions = data['captions']
                picks = data['picks']

                # 遍历每个场景切分 (Change Points)
                for scene_idx, (start, end) in enumerate(cps):
                    scene_picks = []
                    scene_captions = []
                    # 筛选属于当前场景的 picks 和 captions
                    for idx, pick_frame in enumerate(picks):
                        if start <= pick_frame < end:
                            scene_picks.append(pick_frame)
                            scene_captions.append(frame_captions[idx])

                    if scene_picks:
                        batch_tasks.append((scene_captions, scene_picks))
                        batch_meta.append((ds_name, video_name, scene_idx))
                        # 仅用于 DEBUG
            #             if len(batch_tasks) >= debug_limit:
            #                 break
            # if len(batch_tasks) >= debug_limit:
            #     break

        if not batch_tasks:
            logging.info("No new scenes to caption.")
            return

        logging.info(f"Total scenes to caption: {len(batch_tasks)}")
        # 运行 Batch

        async def run_batch():
            captions_in = [t[0] for t in batch_tasks]
            picks_in = [t[1] for t in batch_tasks]
            return await self.scene_caption_model.batch_caption_scenes_async(captions_in, picks_in)

        logging.info("Sending batch request...")
        results = asyncio.run(run_batch())

        # 分发结果并保存
        # 临时存储: temp_results[ds_name][video_name][scene_idx] = summary
        temp_results = {'summe': {}, 'tvsum': {}}

        for i, res in enumerate(results):
            ds_name, video_name, scene_idx = batch_meta[i]
            if video_name not in temp_results[ds_name]:
                temp_results[ds_name][video_name] = {}
            temp_results[ds_name][video_name][scene_idx] = res

        # 将结果写入最终 JSON 结构
        dataset_configs = [
            ('summe', summe_scene_caption_json,
             summe_frame_caption_json, summe_scene_caption_json_file),
            ('tvsum', tvsum_scene_caption_json,
             tvsum_frame_caption_json, tvsum_scene_caption_json_file)
        ]

        for ds_name, final_json, frame_data, json_file in dataset_configs:
            repo_updates = temp_results[ds_name]
            if not repo_updates:
                continue

            for video_name, scenes_map in repo_updates.items():
                cps = cps_map[ds_name][video_name]
                if isinstance(cps, np.ndarray):
                    cps = cps.tolist()

                # 保证按照场景索引生成结构
                video_obj = {}
                for scene_idx, (start, end) in enumerate(cps):
                    # 获取该场景对应 picks / captions
                    scene_picks = []
                    scene_captions = []
                    picks = frame_data[video_name]['picks']
                    captions = frame_data[video_name]['captions']
                    for i, pick_frame in enumerate(picks):
                        if start <= pick_frame < end:
                            scene_picks.append(pick_frame)
                            scene_captions.append(captions[i])

                    video_obj[str(scene_idx)] = {
                        "scene_idx": scene_idx,
                        "picks": scene_picks,
                        "scene_caption": scenes_map.get(scene_idx, ""),
                        "change_points": [start, end]
                    }

                final_json[video_name] = video_obj

            with open(json_file, 'w') as f:
                json.dump(final_json, f, indent=4)

        print("Batch scene captioning completed.")

    def run(self):
        # 进行帧字幕提取
        self._frame_caption()
        # 进行场景字幕总结
        self._scene_caption()
