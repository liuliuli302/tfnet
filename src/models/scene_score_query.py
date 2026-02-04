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
class SceneScoreQueryConfig(BasicConfig):
    model_name: str
    scene_score_prompt: Union[str, List[str]]
    llm_client_config_file_path: str
    max_retries: int
    retry_delay: int


class SceneScoreQuery:
    """
    Query LLM for scene importance score based on scene caption and video caption.
    """

    def __init__(self, config: SceneScoreQueryConfig):
        if not isinstance(config, SceneScoreQueryConfig):
            raise TypeError(
                "config must be SceneScoreQueryConfig")
        self.config = config
        self.llm_client_config = OpenAIClientConfig.load_config_from_file(
            config.llm_client_config_file_path
        )

        self.llm_client = OpenAIClient(self.llm_client_config)

    async def query_score_async(self, scene_caption: str, video_caption: str) -> str:
        prompt_template = self.config.scene_score_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        prompt = prompt_template.replace(
            "{scene_caption}", scene_caption).replace(
            "{video_caption}", video_caption)

        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.generate_async(
            {"messages": messages},
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        return response.strip()

    async def batch_query_scores_async(self, scene_captions: List[str], video_captions: List[str]) -> List[str]:
        """
        Batch version of query_score_async using llm_client's batch_generate.
        Lengths of scene_captions and video_captions must match.
        """
        if len(scene_captions) != len(video_captions):
            raise ValueError(
                f"scene_captions length {len(scene_captions)} != video_captions length {len(video_captions)}")

        prompt_template = self.config.scene_score_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        prompts = []
        for scene_cap, video_cap in zip(scene_captions, video_captions):
            prompt = prompt_template.replace(
                "{scene_caption}", scene_cap).replace(
                "{video_caption}", video_cap)
            prompts.append(prompt)

        # Call the batch API (带重试&失败填空)
        results, failed = await self.llm_client.batch_generate(
            prompts,
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            return_failed=True
        )
        if failed:
            print(f"batch_query_scores_async failed indices: {failed}")
        return [(res.strip() if res else "") for res in results]
