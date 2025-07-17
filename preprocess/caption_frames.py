import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
from tqdm import tqdm
import json


# 加载预训练的 BLIP 模型和处理器
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


@torch.no_grad()
def caption_frame(image_path):
    """
    为单个图像帧生成标题
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')

    # 处理图像
    inputs = processor(image, return_tensors="pt").to(device)

    # 生成标题
    out = model.generate(**inputs, max_length=50, num_beams=5)

    # 解码生成的标题
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption


@torch.no_grad()
def batch_caption_frames(image_paths, batch_size=8):
    """
    批量处理多个图像帧
    """
    captions = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing frames", leave=False):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # 加载批次图像
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            batch_images.append(image)

        # 批量处理
        inputs = processor(batch_images, return_tensors="pt",
                           padding=True).to(device)

        outputs = model.generate(**inputs, max_length=50, num_beams=5)

        # 解码批次结果
        batch_captions = processor.batch_decode(
            outputs, skip_special_tokens=True)
        captions.extend(batch_captions)

    return captions


def process_dataset(frame_dirs, dataset_name, batch_size=128):
    """
    处理单个数据集的所有视频帧
    """
    video_names = os.listdir(frame_dirs)
    caption_result = {}

    for video_name in tqdm(video_names, desc=f"Processing {dataset_name} videos"):
        video_path = os.path.join(frame_dirs, video_name)
        frame_files = os.listdir(video_path)

        picks = [int(file.split('.')[0]) for file in frame_files]

        # 构建完整的帧路径列表
        frame_paths = [os.path.join(video_path, frame_file)
                       for frame_file in frame_files]
        # 批量处理所有帧
        captions = batch_caption_frames(frame_paths, batch_size)
        caption_result[video_name] = {
            'picks': picks,
            'captions': captions
        }

    return caption_result


if __name__ == "__main__":
    # 使用示例
    summe_frame_dirs = "/root/autodl-tmp/datasets/SumMe/frames/"
    tvsum_frame_dirs = "/root/autodl-tmp/datasets/TVSum/frames/"
    batch_size = 256

    # 结果文件路径
    output_dir = "/root/tfnet/data/captions/frame_caption/blip"

    # 处理两个数据集
    summe_caption_result = process_dataset(
        summe_frame_dirs, "SumMe", batch_size)
    tvsum_caption_result = process_dataset(
        tvsum_frame_dirs, "TVSum", batch_size)

    # 保存结果
    with open(f'{output_dir}/summe_captions.json', 'w') as f:
        json.dump(summe_caption_result, f, indent=2)

    with open(f'{output_dir}/tvsum_captions.json', 'w') as f:
        json.dump(tvsum_caption_result, f, indent=2)

    print(f"SumMe数据集处理完成，共{len(summe_caption_result)}个视频")
    print(f"TVSum数据集处理完成，共{len(tvsum_caption_result)}个视频")
