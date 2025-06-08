import argparse
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
            except Exception as e:
                print(f"Warning: Failed to load image {path}: {e}")
        else:
            print(f"Warning: Image file not found: {path}")
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
            print(f"Warning: No frame files found in {video_dir}")
            return

        for batch_start_frame in tqdm(
            range(0, num_frames, self.batch_size * self.frame_interval),
            desc=f"Processing {video_dir}",
            unit="batch",
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


def run(
    video_frames_dir,
    batch_size,
    frame_interval,
    imagefile_template,
    pretrained_model_name,
    output_dir,
    dtype,
    resume,
    pathname,
):
    captioner = ImageCaptioner(
        batch_size,
        frame_interval,
        imagefile_template,
        pretrained_model_name,
        dtype,
        output_dir,
    )

    # 从视频帧文件夹获取所有视频目录
    video_dirs = []
    if os.path.exists(video_frames_dir):
        for item in os.listdir(video_frames_dir):
            item_path = os.path.join(video_frames_dir, item)
            if os.path.isdir(item_path):
                video_dirs.append(item_path)
    else:
        print(f"Error: Video frames directory not found: {video_frames_dir}")
        return
    
    if resume:
        video_dirs = find_unprocessed_videos(video_dirs, captioner.output_dir, pathname)

    for video_dir in video_dirs:
        captioner.process_video(video_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_frames_dir", type=str, required=True, 
                       help="Directory containing video frame folders")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--imagefile_template", type=str, default="frame_{:06d}.jpg")
    parser.add_argument(
        "--pretrained_model_name", type=str, default="Salesforce/blip2-opt-6.7b-coco"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Data type (float16 or float32)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pathname", type=str, default="*.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.video_frames_dir,
        args.batch_size,
        args.frame_interval,
        args.imagefile_template,
        args.pretrained_model_name,
        args.output_dir,
        args.dtype,
        args.resume,
        args.pathname,
    )
