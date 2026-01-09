from abc import ABC, abstractmethod
from PIL import Image
import os
import base64
from io import BytesIO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import asyncio

# 定义几个大模型
# 1 moonshot-v1-128k
# 2 deepseek


class LLMBaseClient(ABC):
    @abstractmethod
    def generate(self, input_data: dict) -> str:
        pass

    @abstractmethod
    async def generate_async(self, input_data: dict) -> str:
        pass


def to_list(x):
    return x if isinstance(x, list) else ([x] if x is not None else [])


def load_image(img):
    if isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"Image file not found: {img}")
        img = Image.open(img).convert("RGB")
    elif not isinstance(img, Image.Image):
        raise TypeError("Image must be a path or PIL.Image")
    return img


def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class OpenAIClient:
    def __init__(self, api_key, base_url="https://api.openai.com/v1", model="gpt-4-vision-preview"):
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        # 创建异步客户端
        self.async_client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url)

    def generate(self, input_data: dict) -> str:
        messages = input_data.get("messages")
        prompt = input_data.get("prompt")
        images = to_list(input_data.get("image"))

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
                pil_img = load_image(img)
                b64 = image_to_base64(pil_img)
                content.append({
                    "type": "image",
                    "image": {
                        "data": b64
                    }
                })
            messages[0]["content"] = content

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content

    async def generate_async(self, input_data: dict) -> str:
        """异步版本的生成方法"""
        messages = input_data.get("messages")
        prompt = input_data.get("prompt")
        images = to_list(input_data.get("image"))

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
                pil_img = load_image(img)
                b64 = image_to_base64(pil_img)
                content.append({
                    "type": "image",
                    "image": {
                        "data": b64
                    }
                })
            messages[0]["content"] = content

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content


class BLIPClient(LLMBaseClient):
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base")

    def generate(self, input_data: dict) -> str:
        images = to_list(input_data.get("image"))
        prompt = input_data.get("prompt", "")

        if not images:
            if prompt.strip():
                raise ValueError(
                    "BLIPClient requires image input for caption generation.")
            return ""

        captions = []
        for img in images:
            pil_img = load_image(img)
            inputs = self.processor(
                images=pil_img, text=prompt, return_tensors="pt")
            with torch.no_grad():
                output = self.model.generate(**inputs)
            captions.append(self.processor.decode(
                output[0], skip_special_tokens=True))

        return "\n".join(captions)

    async def generate_async(self, input_data: dict) -> str:
        """异步版本 - 对于本地模型，使用线程池执行"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, input_data)


deepseek = OpenAIClient(
    api_key="sk-dd060a90600d43f4923e24908eddee16",
    base_url="https://api.deepseek.com/v1", model="deepseek-chat"
)

moonshot = OpenAIClient(
    api_key="sk-8JWkL64sxeKuap0E2U5P9aP2S7oafyTAaR5CxlBi7RJ9ZDX6",
    base_url="https://api.moonshot.cn/v1", model="moonshot-v1-128k"
)
