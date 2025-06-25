import json
import os
from pathlib import Path
import glob
from PIL import Image

import torch
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def load_images_from_paths(image_paths):
    """从路径列表加载图像"""
    images = []
    for path in image_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception:
                # Silently skip failed images
                continue
        # Silently skip missing files
    return images


def find_unprocessed_videos(video_dirs, output_dir, pathname="*.json"):
    """查找未处理的视频"""
    processed_videos = set()
    if os.path.exists(output_dir):
        processed_files = glob.glob(os.path.join(output_dir, pathname))
        processed_videos = {os.path.splitext(os.path.basename(f))[0] for f in processed_files}
    
    unprocessed_videos = []
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        if video_name not in processed_videos:
            unprocessed_videos.append(video_dir)
    
    return unprocessed_videos


class ImageCaptioner:
    def __init__(
        self,
        batch_size,
        frame_interval,
        imagefile_template,
        pretrained_model_name,
        dtype_str,
        output_dir,
    ):
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.imagefile_template = imagefile_template
        self.dtype = self._get_dtype(dtype_str)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(pretrained_model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            pretrained_model_name, torch_dtype=self.dtype
        )
        self.model.to(self.device)
        self.output_dir = Path(output_dir)

    def _get_dtype(self, dtype_str):
        return torch.float16 if dtype_str == "float16" else torch.float32

    def process_video(self, video_dir):
        video_name = os.path.basename(video_dir)
        video_captions = {}
        
        # 获取视频文件夹中的所有帧文件
        frame_patterns = ['*.jpg', '*.jpeg', '*.png']
        frame_files = []
        for pattern in frame_patterns:
            frame_files.extend(glob.glob(os.path.join(video_dir, pattern)))
        
        # 按文件名排序
        frame_files.sort()
        num_frames = len(frame_files)
        
        if num_frames == 0:
            return  # Silently skip empty directories

        for batch_start_frame in tqdm(
            range(0, num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {os.path.basename(video_dir)}",
            unit="batch",
            leave=False,
        ):
            batch_end_frame = min(
                batch_start_frame + (self.batch_size * self.frame_interval), num_frames
            )
            batch_frame_idxs = range(batch_start_frame, batch_end_frame, self.frame_interval)
            batch_frame_paths = [
                Path(video_dir) / self.imagefile_template.format(frame_idx)
                for frame_idx in batch_frame_idxs
            ]
            batch_raw_images = load_images_from_paths(batch_frame_paths)
            
            if not batch_raw_images:
                continue
                
            batch_inputs = self.processor(images=batch_raw_images, return_tensors="pt").to(
                self.device, dtype=self.dtype
            )

            generated_ids = self.model.generate(**batch_inputs)
            batch_generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            for frame_idx, generated_text in zip(batch_frame_idxs, batch_generated_text):
                generated_text = generated_text.strip()
                video_captions[frame_idx] = generated_text

        output_path = self.output_dir / f"{video_name}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(video_captions, f, indent=4)

    def caption_frames(self, frames_dir, frame_interval=15, batch_size=None, resume=True, pathname="*.json"):
        """
        为指定文件夹中的视频帧生成caption
        
        Args:
            frames_dir: 包含视频帧文件夹的目录路径，如 /frames
            frame_interval: 帧跳数，默认为15，表示每隔15帧抽取一帧做caption
            batch_size: 批处理大小，如果为None则使用初始化时的batch_size
            resume: 是否跳过已处理的视频
            pathname: 用于判断已处理视频的文件模式
        """
        # 设置输出目录为 /captions/Blip2/
        captions_output_dir = Path(frames_dir).parent / "captions" / "Blip2"
        captions_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时保存原来的设置
        original_output_dir = self.output_dir
        original_frame_interval = self.frame_interval
        original_batch_size = self.batch_size
        
        # 设置新的参数
        self.output_dir = captions_output_dir
        self.frame_interval = frame_interval
        if batch_size is not None:
            self.batch_size = batch_size
        self.frame_interval = frame_interval
        
        try:
            # 从视频帧文件夹获取所有视频目录
            video_dirs = []
            if os.path.exists(frames_dir):
                for item in os.listdir(frames_dir):
                    item_path = os.path.join(frames_dir, item)
                    if os.path.isdir(item_path):
                        video_dirs.append(item_path)
            else:
                raise FileNotFoundError(f"Video frames directory not found: {frames_dir}")
            
            if resume:
                video_dirs = find_unprocessed_videos(video_dirs, self.output_dir, pathname)

            # Add progress bar for video processing
            for video_dir in tqdm(video_dirs, desc="Processing videos", unit="video"):
                self.process_video(video_dir)
                
        finally:
            # 恢复原来的设置
            self.output_dir = original_output_dir
            self.frame_interval = original_frame_interval
            self.batch_size = original_batch_size


