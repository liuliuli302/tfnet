from dataclasses import dataclass
import traceback
import sys
import copy
import torch
from PIL import Image
import cv2
import numpy as np
from typing import List, Union, Optional, cast
import os
import yaml
from transformers import BlipProcessor, BlipForConditionalGeneration
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from src.config.config import BasicConfig
from decord import VideoReader
from src.utils.video_loader import VideoLoader


@dataclass
class BlipFrameCaptionerConfig(BasicConfig):
    model_path: str
    device: str
    model_name: str


class BlipFrameCaptioner:
    """
    BlipFrameCaptioner:
        - Frame-by-frame caption generation for video.
        - Based on BLIP model.
        - Supports beam search, top-p sampling, and more.
    """

    def __init__(self, config: BlipFrameCaptionerConfig):
        """
        Args:
            config (BlipFrameCaptionerConfig): 配置对象
        """
        if not isinstance(config, BlipFrameCaptionerConfig):
            raise TypeError("config参数必须为BlipFrameCaptionerConfig类型")
        self.config = config
        self.device = torch.device(config.device)
        self.model_name = config.model_name
        self._load_model(config.model_path)

    def _load_model(self, model_path: str):
        """
        Load the BLIP model and processor from the pretrained checkpoint.
        """
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def caption_image(
        self,
        image:  Image.Image,
        return_logits: bool = False,
    ):
        """
        Generate caption for a single image.

        :param image: The input image (either path or PIL Image).
        :param return_logits: If True, return the logits and sequence too.
        :return: Generated caption or dictionary with logits.
        """
        inputs = self.processor(
            images=image,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
        )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    @torch.no_grad()
    def caption_image_batch(self, image_batch: List[Image.Image], batch_size: int = 32):
        """
        Generate captions for all frames in the video.

        :param image_batch: List of PIL images.
        :param batch_size: Batch size for inference.
        :return: List of captions for each frame.
        """
        captions = []
        if not image_batch:
            return captions

        for i in range(0, len(image_batch), batch_size):
            batch_imgs = image_batch[i: i + batch_size]

            inputs = self.processor(
                images=batch_imgs,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(**inputs)
            batch_captions = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True)
            captions.extend(batch_captions)

        return captions


@dataclass
class LlavaFrameCaptionerConfig(BasicConfig):
    pretrained: str
    model_base: str
    model_name: str
    device: str
    device_map: str
    conv_template: str
    torch_dtype: str
    max_new_tokens: int
    frame_caption_prompt: str


class LlavaFrameCaptioner:
    """
    LlavaFrameCaptioner:
        - Frame-by-frame caption generation using LLaVA.
        - Matches the interface of BlipFrameCaptioner for drop-in use.
    """

    def __init__(self, config: LlavaFrameCaptionerConfig):
        if not isinstance(config, LlavaFrameCaptionerConfig):
            raise TypeError("config参数必须为LlavaFrameCaptionerConfig类型")
        self.config = config
        self._load_model()

    def _load_model(self):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=self.config.pretrained,
            model_base=self.config.model_base,
            model_name=self.config.model_name,
            torch_dtype=self.config.torch_dtype,
            device_map=self.config.device_map,
        )
        self.model.eval()
        self.model.to(self.config.device)

    @torch.no_grad()
    def _build_prompt(self, prompt: str) -> torch.Tensor:
        conv = copy.deepcopy(conv_templates[self.config.conv_template])
        question = f"{DEFAULT_IMAGE_TOKEN} {prompt}".strip()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_text,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        return input_ids.to(self.config.device)

    @torch.no_grad()
    def caption_image(self, image: Image.Image, prompt: str = "Please describe this image in detail."):
        input_ids = self._build_prompt(prompt)

        # [新增] 获取图像尺寸，某些 LLaVA 版本在处理多模态输入时需要此信息
        image_size = image.size  # (W, H)

        pixel_values = self.image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"].to(self.config.device, dtype=torch.float16)

        # [修改] 调用 generate 时显式指定 keyword arguments，并尝试传递 image_sizes
        # 注意: 这里 input_ids 传给 generate，但在某些 transformers 版本中第一个参数名为 inputs
        outputs = self.model.generate(
            input_ids,  # 尝试作为位置参数传递，或使用 input_ids=input_ids
            images=pixel_values,
            image_sizes=[image_size],  # (W, H)
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )

        caption = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)[0].strip()
        return caption

    @torch.no_grad()
    def caption_image_batch(
        self,
        image_batch: List[Image.Image],
        prompt: str = "Please describe each image in order.",
        batch_size: int = 8,
    ):
        captions: List[str] = []
        for img in image_batch:
            cap = self.caption_image(img, prompt=prompt)
            captions.append(cap)
        return captions


def main_llava():
    print(">>> 正在启动 LlavaFrameCaptioner 测试流程...")

    # -------------------------------------------------------------------------
    # 1. 环境与路径设置
    # -------------------------------------------------------------------------
    project_root = "/root/tfnet"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Info: 项目根目录已设置为 {project_root}")

    # -------------------------------------------------------------------------
    # 2. 导入项目模块 (需在 sys.path 设置后进行)
    # -------------------------------------------------------------------------
    try:
        from src.config.config import BasicConfig
        from src.models.frame_caption import LlavaFrameCaptioner, LlavaFrameCaptionerConfig
    except ImportError as e:
        print(f"Error: 无法导入项目模块: {e}")
        return

    # -------------------------------------------------------------------------
    # 3. Monkey Patch: 修复 BasicConfig 缺失 .get() 方法的问题
    # -------------------------------------------------------------------------
    if not hasattr(BasicConfig, 'get'):
        print("Info: 为 BasicConfig 应用 .get() 方法补丁...")

        def config_get(self, key, default=None):
            return getattr(self, key, default)
        BasicConfig.get = config_get

    # -------------------------------------------------------------------------
    # 4. 加载与修补配置
    # -------------------------------------------------------------------------
    config_path = os.path.join(
        project_root, "configs/model/frame_caption_llava.yaml")
    if not os.path.exists(config_path):
        print(f"Error: 配置文件未找到: {config_path}")
        return

    print(f"Info: 加载配置文件 {config_path}")
    try:
        config = LlavaFrameCaptionerConfig.load_config_from_file(config_path)
    except Exception as e:
        print(f"Error: 配置加载失败: {e}")
        return

    # 补充 YAML 中可能缺失的关键参数
    if not hasattr(config, 'conv_template'):
        # Qwen2 系列通常使用 qwen_1_5 或 qwen_2 模板
        print("Warning: Config 缺失 'conv_template'，设置默认值为 'qwen_1_5'")
        config.conv_template = 'qwen_1_5'

    if not hasattr(config, 'max_new_tokens'):
        config.max_new_tokens = 200

    # 强制检查设备
    if not torch.cuda.is_available():
        print("Warning:未检测到 CUDA，将强制使用 CPU 模式（速度较慢）")
        config.device = 'cpu'
        config.device_map = 'cpu'

    # -------------------------------------------------------------------------
    # 5. 初始化模型
    # -------------------------------------------------------------------------
    print("\n>>> 初始化模型 (首次加载可能需要下载权重)...")
    try:
        captioner = LlavaFrameCaptioner(config)
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("请确保已安装 llava: pip install git+https://github.com/haotian-liu/LLaVA.git")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # 6. 执行测试
    # -------------------------------------------------------------------------
    print("\n>>> 准备测试数据 (纯色图片)...")
    img_a = Image.new('RGB', (336, 336), color=(200, 50, 50))  # 红色
    img_b = Image.new('RGB', (336, 336), color=(50, 50, 200))  # 蓝色
    prompt = "Describe the color of the image."

    # Test 1: 单图
    print("\n--- Test 1: caption_image (单图) ---")
    try:
        result = captioner.caption_image(img_a, prompt=prompt)
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Test 1 Failed: {e}")
        traceback.print_exc()

    # Test 2: 批量
    print("\n--- Test 2: caption_image_batch (批量) ---")
    try:
        results = captioner.caption_image_batch(
            [img_a, img_b], prompt=prompt, batch_size=2)
        for i, res in enumerate(results):
            print(f"Image {i}: {res}")
    except Exception as e:
        print(f"Test 2 Failed: {e}")
        traceback.print_exc()

    print("\n>>> 全部测试完成")


def main_blip():
    print(">>> 正在启动 BlipFrameCaptioner 测试流程...")

    # -------------------------------------------------------------------------
    # 1. 环境与路径设置
    # -------------------------------------------------------------------------
    project_root = "/root/tfnet"
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"Info: 项目根目录已设置为 {project_root}")

    # -------------------------------------------------------------------------
    # 2. 导入项目模块 (需在 sys.path 设置后进行)
    # -------------------------------------------------------------------------
    try:
        from src.models.frame_caption import BlipFrameCaptioner, BlipFrameCaptionerConfig
    except ImportError as e:
        print(f"Error: 无法导入项目模块: {e}")
        return

    # -------------------------------------------------------------------------
    # 3. 构造假配置 (需要本地已下载 BLIP 权重)
    # -------------------------------------------------------------------------
    config = BlipFrameCaptionerConfig.load_config_from_file(
        "configs/model/frame_caption_blip.yaml"
    )

    # -------------------------------------------------------------------------
    # 4. 初始化模型
    # -------------------------------------------------------------------------
    print("\n>>> 初始化 Blip 模型 (首次加载可能需要下载权重)...")
    try:
        captioner = BlipFrameCaptioner(config)
        print("✅ Blip 模型初始化成功")
    except Exception as e:
        print(f"❌ Blip 模型初始化失败: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # 5. 执行测试 (假数据)
    # -------------------------------------------------------------------------
    print("\n>>> 准备测试数据 (纯色图片)...")
    img_red = Image.new('RGB', (224, 224), color=(255, 0, 0))
    img_green = Image.new('RGB', (224, 224), color=(0, 255, 0))
    img_blue = Image.new('RGB', (224, 224), color=(0, 0, 255))

    # Test 1: 单图
    print("\n--- Test 1: caption_image (单图) ---")
    try:
        result = captioner.caption_image(img_red)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Test 1 Failed: {e}")
        traceback.print_exc()

    # Test 2: 批量
    print("\n--- Test 2: caption_image_batch (批量) ---")
    try:
        results = captioner.caption_image_batch(
            [img_red, img_green, img_blue], batch_size=2)
        for i, res in enumerate(results):
            print(f"Image {i}: {res}")
    except Exception as e:
        print(f"Test 2 Failed: {e}")
        traceback.print_exc()

    print("\n>>> BlipFrameCaptioner 测试完成")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(
        "ignore",
        message=".*copying from a non-meta parameter.*"
    )
    main_blip()
