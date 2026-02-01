import asyncio
import base64
from dataclasses import dataclass
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


@dataclass
class OpenAIClientConfig(BasicConfig):
    base_url: str
    model: str


class OpenAIClient:
    def __init__(self, config: OpenAIClientConfig):
        if not isinstance(config, OpenAIClientConfig):
            raise TypeError("config must be OpenAIClientConfig")

        self.config = config
        self.model = config.model
        self.base_url = config.base_url

        self.client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=config.base_url
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=config.base_url
        )

    def _prepare_messages(self, input_data: Dict[str, Any]):
        messages = input_data.get("messages")
        prompt = input_data.get("prompt")
        images = _to_list(input_data.get("image"))

        if not messages:
            if prompt is None:
                raise ValueError(
                    "input_data must contain 'messages' or 'prompt'")
            messages = [{"role": "user", "content": prompt}]

        # 做浅拷贝避免修改外部引用
        messages = [dict(m) for m in messages]
        first = messages[0]
        content = first.get("content", "")

        # 统一成 list[part]
        if isinstance(content, str):
            content_parts: List[Any] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_parts = content
        else:
            raise TypeError("messages[0]['content'] must be str or list")

        for img in images:
            b64 = _image_to_base64(_load_image(img))
            content_parts.append(
                {"type": "image", "image": {"data": b64, "media_type": "image/jpeg"}})

        first["content"] = content_parts if images else content
        messages[0] = first
        return messages

    def generate(self, input_data: Dict[str, Any]) -> str:
        messages = self._prepare_messages(input_data)
        response = self.client.chat.completions.create(
            model=self.model, messages=messages)
        return response.choices[0].message.content

    async def generate_async(
        self,
        input_data: Dict[str, Any],
        retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        messages = self._prepare_messages(input_data)

        for attempt in range(1, retries + 1):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model, messages=messages
                )
                return response.choices[0].message.content
            except Exception:
                if attempt < retries:
                    await asyncio.sleep(retry_delay * attempt)

        # 3 次失败后返回空值
        return ""

    async def batch_generate(
        self,
        prompts: List[str],
        batch_size: int = 16,
        retries: int = 3,
        retry_delay: float = 1.0,
        return_failed: bool = False
    ):
        """
        批量并发调用；单次失败重试；最终失败填充空值并记录失败位置。
        return_failed=True 时返回 (results, failed_indices)
        """
        if not prompts:
            return ([], []) if return_failed else []
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        results: List[Optional[str]] = [None] * len(prompts)
        failed: List[int] = []

        sem = asyncio.Semaphore(batch_size)

        async def sem_task(prompt: str, idx: int):
            async with sem:
                res = await self.generate_async(
                    {"prompt": prompt},
                    retries=retries,
                    retry_delay=retry_delay
                )
                return idx, res

        tasks = [sem_task(prompt, idx) for idx, prompt in enumerate(prompts)]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM batch_generate"):
            idx, res = await coro
            results[idx] = res
            if res == "":
                failed.append(idx)

        return (results, failed) if return_failed else results


@dataclass
class BLIPClientConfig(BasicConfig):
    model_name: str = "Salesforce/blip-image-captioning-base"


class BLIPClient:
    def __init__(self, config: BLIPClientConfig):
        if not isinstance(config, BLIPClientConfig):
            raise TypeError("config must be BLIPClientConfig")
        self.config = config
        # placeholder to avoid lazy init cost elsewhere
        self.processor = torch.hub.load("pytorch/vision", "resnet18")
        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained(config.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            config.model_name)

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


if __name__ == "__main__":
    async def main_test():
        # 2. Test OpenAIClient
        print("\n[OpenAIClient Test]")
        if "OPENAI_API_KEY" not in os.environ:
            print("Skipping OpenAIClient test (OPENAI_API_KEY not set)")
        else:
            try:
                # Use a dummy or real config
                config = OpenAIClientConfig(
                    base_url="https://www.dmxapi.cn/v1", model="gpt-5")
                client = OpenAIClient(config)
                print("OpenAIClient initialized.")

                # Uncomment to verify real connection
                res = client.generate({"prompt": "Hello"})
                print(f"Response: {res}")
            except Exception as e:
                print(f"OpenAIClient test failed: {e}")

    asyncio.run(main_test())
