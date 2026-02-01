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
class SceneSummaryCaptionSummarizerConfig(BasicConfig):
    model_name: str
    scene_caption_prompt: Union[str, List[str]]
    llm_client_config_file_path: str
    max_retries: int
    retry_delay: int


class SceneSummaryCaptionSummarizer:
    """
    Generate per-scene summaries from frame captions using an LLM client.
    Requires change points (cps) to define scenes.
    """

    def __init__(self, config: SceneSummaryCaptionSummarizerConfig):
        if not isinstance(config, SceneSummaryCaptionSummarizerConfig):
            raise TypeError(
                "config must be SceneSummaryCaptionSummarizerConfig")
        self.config = config
        self.llm_client_config = OpenAIClientConfig.load_config_from_file(
            config.llm_client_config_file_path
        )

        self.llm_client = OpenAIClient(self.llm_client_config)

    def caption_scene(self, frame_caption_list: List[str], picks: List[int]) -> str:
        if len(frame_caption_list) != len(picks):
            raise ValueError(
                f"frame_caption_list length {len(frame_caption_list)} != picks length {len(picks)}")

        captions_parts = []
        for caption, pick in zip(frame_caption_list, picks):
            captions_parts.append(f"Frame {pick}:\n{caption}")
        frame_captions = "\n".join(captions_parts)

        prompt_template = self.config.scene_caption_prompt[0]

        prompt = prompt_template.format(frame_captions=frame_captions)
        messages = [{"role": "user", "content": prompt}]

        response = ""
        for attempt in range(self.config.max_retries):
            try:
                response = self.llm_client.generate(
                    {"messages": messages}).strip()
                return response
            except Exception as exc:
                print(
                    f"Error in caption_scene (attempt {attempt + 1}/{self.config.max_retries}): {exc}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        return response

    async def caption_scene_async(self, frame_caption_list: List[str], picks: List[int]) -> str:
        if len(frame_caption_list) != len(picks):
            raise ValueError(
                f"frame_caption_list length {len(frame_caption_list)} != picks length {len(picks)}")

        captions_parts = []
        for caption, pick in zip(frame_caption_list, picks):
            captions_parts.append(f"Frame {pick}:\n{caption}")
        frame_captions = "\n".join(captions_parts)

        if isinstance(self.config.scene_caption_prompt, list):
            prompt_template = "\n".join(self.config.scene_caption_prompt)
        else:
            prompt_template = self.config.scene_caption_prompt

        prompt = prompt_template.replace("{frame_captions}", frame_captions)
        messages = [{"role": "user", "content": prompt}]

        # 重试交给 llm_client.generate_async 处理
        response = await self.llm_client.generate_async(
            {"messages": messages},
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        return response.strip()

    async def batch_caption_scenes_async(self, batch_frame_caption_lists: List[List[str]], batch_picks: List[List[int]]) -> List[str]:
        """
        Batch version of caption_scene_async using llm_client's batch_generate.
        """
        if len(batch_frame_caption_lists) != len(batch_picks):
            raise ValueError(
                f"batch_frame_caption_lists length {len(batch_frame_caption_lists)} != batch_picks length {len(batch_picks)}")

        if isinstance(self.config.scene_caption_prompt, list):
            prompt_template = "\n".join(self.config.scene_caption_prompt)
        else:
            prompt_template = self.config.scene_caption_prompt

        prompts = []
        for frame_caption_list, picks in zip(batch_frame_caption_lists, batch_picks):
            if len(frame_caption_list) != len(picks):
                raise ValueError(
                    f"frame_caption_list length {len(frame_caption_list)} != picks length {len(picks)}")

            captions_parts = []
            for caption, pick in zip(frame_caption_list, picks):
                captions_parts.append(f"Frame {pick}:\n{caption}")
            frame_captions = "\n".join(captions_parts)

            prompt = prompt_template.replace(
                "{frame_captions}", frame_captions)
            prompts.append(prompt)

        # Call the batch API (带重试&失败填空)
        results, failed = await self.llm_client.batch_generate(
            prompts,
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            return_failed=True
        )
        if failed:
            print(f"batch_caption_scenes_async failed indices: {failed}")
        return [(res.strip() if res else "") for res in results]
