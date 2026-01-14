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


class BlipFrameCaptionerConfig(BasicConfig):

    def __init__(
        self,
        model_path: str = 'Salesforce/blip-image-captioning-base',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        prompt: str = "",
        return_logits: bool = False,
    ):
        """
        Generate caption for a single image.

        :param image: The input image (either path or PIL Image).
        :param prompt: Optional prompt for condition generation.
        :param return_logits: If True, return the logits and sequence too.
        :return: Generated caption or dictionary with logits.
        """
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
        )

        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    @torch.no_grad()
    def caption_image_batch(self, image_batch: List[Image.Image], prompt: str = "", batch_size: int = 32):
        """
        Generate captions for all frames in the video.

        :param image_batch: List of PIL images.
        :param prompt: Optional prompt for condition generation.
        :param batch_size: Batch size for inference.
        :return: List of captions for each frame.
        """
        captions = []
        if not image_batch:
            return captions

        for i in range(0, len(image_batch), batch_size):
            batch_imgs = image_batch[i: i + batch_size]
            # When processing a batch, processor expects a list of text logic if images is a list
            batch_prompts = [prompt] * len(batch_imgs)

            inputs = self.processor(
                images=batch_imgs,
                text=batch_prompts,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(**inputs)
            batch_captions = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True)
            captions.extend(batch_captions)

        return captions


class LlavaFrameCaptionerConfig(BasicConfig):

    def __init__(
        self,
        pretrained: str,
        model_name: Optional[str] = None,
        conv_template: str = "llava_v1",
        device: Optional[str] = None,
        device_map: str = "auto",
        max_new_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.model_name = model_name or get_model_name_from_path(pretrained)
        self.conv_template = conv_template
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens


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
        self.device = torch.device(config.device)
        self._load_model()

    def _load_model(self):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.config.pretrained,
            None,
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=self.config.device_map,
        )
        self.model.eval()

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
        return input_ids.to(self.device)

    @torch.no_grad()
    def caption_image(self, image: Image.Image, prompt: str = "Please describe this image in detail."):
        input_ids = self._build_prompt(prompt)
        pixel_values = self.image_processor.preprocess(
            image,
            return_tensors="pt",
        )["pixel_values"].to(self.device, dtype=torch.float16)

        outputs = self.model.generate(
            input_ids=input_ids,
            images=pixel_values,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.config.max_new_tokens,
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
        if not image_batch:
            return captions

        input_ids = self._build_prompt(prompt)

        for i in range(0, len(image_batch), batch_size):
            batch_imgs = image_batch[i: i + batch_size]
            pixel_values = self.image_processor.preprocess(
                batch_imgs,
                return_tensors="pt",
            )["pixel_values"].to(self.device, dtype=torch.float16)

            repeated_ids = input_ids.expand(pixel_values.shape[0], -1)

            outputs = self.model.generate(
                input_ids=repeated_ids,
                images=pixel_values,
                do_sample=False,
                temperature=0.0,
                max_new_tokens=self.config.max_new_tokens,
            )

            batch_captions = [cap.strip()
                              for cap in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
            captions.extend(batch_captions)

        return captions
