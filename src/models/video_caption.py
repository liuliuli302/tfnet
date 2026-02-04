import asyncio
from dataclasses import dataclass
import json
import os
import time
from typing import Dict, Any, List, Union
from tqdm import tqdm
from src.config.config import BasicConfig
from src.utils.llm_client import OpenAIClient, OpenAIClientConfig


@dataclass
class VideoSummaryCaptionSummarizerConfig(BasicConfig):
    model_name: str
    video_caption_prompt: Union[str, List[str]]
    llm_client_config_file_path: str
    max_retries: int
    retry_delay: int


class VideoSummaryCaptionSummarizer:
    """
    Generate video-level summaries from scene captions using an LLM client.
    """

    def __init__(self, config: VideoSummaryCaptionSummarizerConfig):
        if not isinstance(config, VideoSummaryCaptionSummarizerConfig):
            raise TypeError(
                "config must be VideoSummaryCaptionSummarizerConfig")
        self.config = config
        self.llm_client_config = OpenAIClientConfig.load_config_from_file(
            config.llm_client_config_file_path
        )

        self.llm_client = OpenAIClient(self.llm_client_config)

    def caption_video(self, scene_caption_list: List[str]) -> str:
        prompt_template = self.config.video_caption_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        scene_captions_str = "\n".join(
            [f"Scene {i+1}:\n{cap}" for i, cap in enumerate(scene_caption_list)])
        prompt = prompt_template.replace(
            "{scene_captions}", scene_captions_str)

        messages = [{"role": "user", "content": prompt}]

        response = ""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.generate(
                    {"messages": messages}).strip()
                return response
            except Exception as exc:
                print(
                    f"Error in caption_video (attempt {attempt + 1}/{self.config.max_retries}): {exc}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        return response

    async def caption_video_async(self, scene_caption_list: List[str]) -> str:
        prompt_template = self.config.video_caption_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        scene_captions_str = "\n".join(
            [f"Scene {i+1}:\n{cap}" for i, cap in enumerate(scene_caption_list)])
        prompt = prompt_template.replace(
            "{scene_captions}", scene_captions_str)
        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.generate_async(
            {"messages": messages},
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        return response.strip()

    async def batch_caption_videos_async(self, batch_scene_caption_lists: List[List[str]]) -> List[str]:
        """
        Batch version of caption_video_async using llm_client's batch_generate.
        """
        prompt_template = self.config.video_caption_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        prompts = []
        for scene_captions in batch_scene_caption_lists:
            scene_captions_str = "\n".join(
                [f"Scene {i+1}:\n{cap}" for i, cap in enumerate(scene_captions)])
            prompt = prompt_template.replace(
                "{scene_captions}", scene_captions_str)
            prompts.append(prompt)

        responses = await self.llm_client.batch_generate(
            prompts,
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        return [r.strip() if r else "" for r in responses]
