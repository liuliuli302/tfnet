import asyncio
import base64
import json
import os
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Dict, List, Optional

import openai
import torch
from PIL import Image
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from src.config.config import BasicConfig


def _to_list(x):
    return x if isinstance(x, list) else ([x] if x is not None else [])


def _load_image(img):
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        raise TypeError("Image must be a path or PIL.Image")
    return img


def _image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class LLMBaseClient(ABC):
    @abstractmethod
    def generate(self, input_data: Dict[str, Any]) -> str:
        ...

    @abstractmethod
    async def generate_async(self, input_data: Dict[str, Any]) -> str:
        ...


class OpenAIClient(LLMBaseClient):
    def __init__(self, api_key: str, base_url: str, model: str = "gpt-4-vision-preview"):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url)

    def _prepare_messages(self, input_data: Dict[str, Any]):
        messages = input_data.get("messages")
        prompt = input_data.get("prompt")
        images = _to_list(input_data.get("image"))

        if messages is None:
            if prompt is None:
                raise ValueError(
                    "input_data must contain 'messages' or 'prompt'")
            messages = [{"role": "user", "content": prompt}]

        content = messages[0]["content"]
        if images:
            if isinstance(content, str):
                content = [content]
            elif not isinstance(content, list):
                raise TypeError("messages[0]['content'] must be str or list")

            for img in images:
                pil_img = _load_image(img)
                b64 = _image_to_base64(pil_img)
                content.append({"type": "image", "image": {"data": b64}})
            messages[0]["content"] = content
        return messages

    def generate(self, input_data: Dict[str, Any]) -> str:
        messages = self._prepare_messages(input_data)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages)
        return response.choices[0].message.content

    async def generate_async(self, input_data: Dict[str, Any]) -> str:
        messages = self._prepare_messages(input_data)
        response = await self.async_client.chat.completions.create(model=self.model, messages=messages)
        return response.choices[0].message.content


class BLIPClient(LLMBaseClient):
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        # placeholder to avoid lazy init cost elsewhere
        self.processor = torch.hub.load("pytorch/vision", "resnet18")
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_data: Dict[str, Any]) -> str:
        images = _to_list(input_data.get("image"))
        prompt = input_data.get("prompt", "")
        if not images:
            if prompt.strip():
                raise ValueError(
                    "BLIPClient requires image input for caption generation.")
            return ""

        captions = []
        for img in images:
            pil_img = _load_image(img)
            inputs = self.processor(
                images=pil_img, text=prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(**inputs)
            captions.append(self.processor.decode(
                output[0], skip_special_tokens=True))
        return "\n".join(captions)

    async def generate_async(self, input_data: Dict[str, Any]) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, input_data)


class LLMQueryConfig(BasicConfig):
    def __init__(self, query_type: str, prompt: str, max_concurrent: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.query_type = query_type
        self.prompt = prompt
        self.max_concurrent = max_concurrent


class LLMQueryRunner:
    """
    Encapsulates text-only LLM querying for frame captions and optional video summaries.
    Mirrors the class-style design used in frame_caption.
    """

    def __init__(self, model: LLMBaseClient, config: LLMQueryConfig):
        if not isinstance(config, LLMQueryConfig):
            raise TypeError("config must be LLMQueryConfig")
        self.model = model
        self.config = config

    def _build_content(self, frame_caption: str, video_summary: Optional[str]) -> str:
        query_type_list = self.config.query_type.split("_")
        if len(query_type_list) < 2:
            raise ValueError(
                "query_type must include modality information, e.g., text_ws")
        if query_type_list[1] == "ws":
            if video_summary is None:
                raise ValueError(
                    "video_summary required when query_type ends with 'ws'")
            return self.config.prompt.format(frame_caption=frame_caption, video_summary=video_summary)
        return self.config.prompt.format(frame_caption=frame_caption)

    async def query_llm_text_async(self, frame_caption: str, video_summary: Optional[str]):
        content = self._build_content(frame_caption, video_summary)
        input_data = {"messages": [{"role": "user", "content": content}]}
        llm_out = await self.model.generate_async(input_data)
        return llm_out, content

    def query_llm_text(self, frame_caption: str, video_summary: Optional[str]):
        content = self._build_content(frame_caption, video_summary)
        input_data = {"messages": [{"role": "user", "content": content}]}
        llm_out = self.model.generate(input_data)
        return llm_out, content

    async def process_video_text_async(self, frame_caption_file: Dict[str, Any], video_caption_file: Dict[str, Any], video_name: str, semaphore: asyncio.Semaphore):
        frame_caption = frame_caption_file[video_name]["captions"]
        frame_picks = frame_caption_file[video_name]["picks"]
        video_caption = video_caption_file[video_name].get("summary")

        async def process_single_frame(frame_idx, caption):
            async with semaphore:
                result, content = await self.query_llm_text_async(caption, video_caption)
                return {"frame_idx": frame_idx, "query": content, "llm_output": result}

        tasks = [process_single_frame(
            frame_idx, caption) for frame_idx, caption in zip(frame_picks, frame_caption)]
        llm_out_result = await atqdm.gather(*tasks, desc=f"Processing {video_name}")
        return {video_name: llm_out_result}

    async def process_dataset_text_async(self, frame_caption_file: Dict[str, Any], video_caption_file: Dict[str, Any]):
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        tasks = []
        for video_name in list(frame_caption_file.keys()):
            task = self.process_video_text_async(
                frame_caption_file, video_caption_file, video_name, semaphore)
            tasks.append(task)
        results = await atqdm.gather(*tasks, desc="Processing dataset")
        return results

    def process_video_text(self, frame_caption_file: Dict[str, Any], video_caption_file: Dict[str, Any], video_name: str):
        frame_caption = frame_caption_file[video_name]["captions"]
        frame_picks = frame_caption_file[video_name]["picks"]
        video_caption = video_caption_file[video_name].get("summary")

        llm_out_result = []
        for frame_idx, caption in tqdm(zip(frame_picks, frame_caption), desc=f"Processing {video_name}", leave=False):
            result, content = self.query_llm_text(caption, video_caption)
            llm_out_result.append(
                {"frame_idx": frame_idx, "query": content, "llm_output": result})
        return {video_name: llm_out_result}

    def process_dataset_text(self, frame_caption_file: Dict[str, Any], video_caption_file: Dict[str, Any]):
        if hasattr(self.model, "generate_async"):
            return asyncio.run(self.process_dataset_text_async(frame_caption_file, video_caption_file))
        results = []
        for video_name in frame_caption_file.keys():
            results.append(self.process_video_text(
                frame_caption_file, video_caption_file, video_name))
        return results

    @staticmethod
    def save_result(results: Any, output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {output_path}")


# Predefined clients similar to previous llm.py
deepseek = OpenAIClient(api_key="sk-dd060a90600d43f4923e24908eddee16",
                        base_url="https://api.deepseek.com/v1", model="deepseek-chat")
moonshot = OpenAIClient(api_key="sk-8JWkL64sxeKuap0E2U5P9aP2S7oafyTAaR5CxlBi7RJ9ZDX6",
                        base_url="https://api.moonshot.cn/v1", model="moonshot-v1-128k")


# Backward-compatible functional interface
async def process_dataset_text_async(model, query_type, prompt, frame_caption_file, video_caption_file, max_concurrent=5):
    runner = LLMQueryRunner(model, LLMQueryConfig(
        query_type=query_type, prompt=prompt, max_concurrent=max_concurrent))
    return await runner.process_dataset_text_async(frame_caption_file, video_caption_file)


def process_dataset_text(model, query_type, prompt, frame_caption_file, video_caption_file):
    runner = LLMQueryRunner(model, LLMQueryConfig(
        query_type=query_type, prompt=prompt))
    return runner.process_dataset_text(frame_caption_file, video_caption_file)


def save_result(results, output_path):
    LLMQueryRunner.save_result(results, output_path)


def test_module():
    mock_model = moonshot
    mock_query_type = "text_ws"
    mock_prompt = "基于以下视频摘要：'{video_summary}'，请判断以下帧描述：'{frame_caption}'，对于总结整个视频的重要性，并给出一个0-100之间的分数。请输出并仅仅输出一个分数"

    mock_frame_captions = {"video1": {
        "captions": ["一个人在切菜", "一个人在炒菜"], "picks": [10, 25]}}
    mock_video_captions = {"video1": {"summary": "这个视频展示了如何做一道家常菜。"}}

    print("--- 开始测试 process_dataset_text ---")
    results = process_dataset_text(model=mock_model, query_type=mock_query_type, prompt=mock_prompt,
                                   frame_caption_file=mock_frame_captions, video_caption_file=mock_video_captions)
    print("\n--- 测试完成 ---")
    print("LLM返回结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    assert isinstance(results, list)
    assert "video1" in results[0]
    print("\n测试函数结构正确。")


if __name__ == "__main__":
    test_module()
