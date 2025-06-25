# /home/autodl-tmp/conda_envs/ETBench)

from decord import VideoReader, cpu
from pathlib import Path
import glob
import random
from torchvision import transforms
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import numpy as np
import pickle
import torch
from tqdm import tqdm
import json
import os

os.environ['XDG_CACHE_HOME'] = '/root/autodl-tmp/.cache'
print(
    f'-----------------------------ENV_XDG_CACHE_HOME: {os.getenv("XDG_CACHE_HOME")}')


device = 'cuda'
# 加载模型和预处理器，保持 vis_processors['eval'] 的原始状态
model, vis_processors, text_processors = load_model_and_preprocess(
    "blip2_image_text_matching", "coco", device=device, is_eval=True)

# REMOVE or COMMENT OUT the problematic modification:
# vis_processors = transforms.Compose([
# t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
# ])
# The get_visual_features function will now use vis_processors['eval'] directly from the loaded dict.


def loadvideo(fname, frame_interval=15, max_duration=None):
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    all_index = np.arange(0, len(vr), frame_interval, dtype=np.int32)
    if max_duration is not None:
        max_frames = round(max_duration * len(vr) / duration)
        all_index = all_index[:max_frames // frame_interval]
    vr.seek(0)
    buffer = vr.get_batch(all_index).numpy() # Ensure it's a numpy array
    # Permute to (N, C, H, W) if necessary, decord default is (N, H, W, C)
    # Assuming get_batch gives (N,H,W,C) uint8, permute to (N,C,H,W) for ToPILImage on (C,H,W)
    if buffer.ndim == 4 and buffer.shape[3] in [1, 3]: # Check if it's N, H, W, C
        buffer = np.ascontiguousarray(buffer.transpose(0, 3, 1, 2)) # N, C, H, W
    return buffer


@torch.no_grad()
def get_visual_features(video_path, frame_interval=None, max_duration=None, batch_size=128):
    # loadvideo returns a numpy array (N, C, H, W) of uint8 frames
    video_frames_numpy = loadvideo(video_path, frame_interval, max_duration)
    video_frames_tensor_uint8 = torch.from_numpy(video_frames_numpy)

    # Use the original evaluation processor from LAVIS
    # vis_processors is the global dictionary loaded by load_model_and_preprocess
    original_eval_processor = vis_processors['eval']
    
    to_pil_transform = transforms.ToPILImage()
    processed_images_list = []

    # Process each frame individually to ensure correct application of transforms
    for i in range(video_frames_tensor_uint8.size(0)):
        frame_chw_uint8 = video_frames_tensor_uint8[i]  # Single frame (C, H, W), uint8
        pil_image = to_pil_transform(frame_chw_uint8)   # Convert to PIL Image
        
        # Apply the full, original LAVIS visual processing pipeline
        processed_tensor = original_eval_processor(pil_image) # Output is a normalized float tensor
        processed_images_list.append(processed_tensor)
    
    # Stack processed frames into a batch
    if not processed_images_list:
        # Handle empty video or no frames selected case
        return np.array([]) # Or raise an error, or return None
        
    img_batch_processed = torch.stack(processed_images_list) # Shape: (N, C, H, W), float, normalized

    features = []
    # The batch_size here is for the model inference, not for visual preprocessing
    for bid in range(0, img_batch_processed.size(0), batch_size):
        batch_img_for_model = img_batch_processed[bid:bid+batch_size].to(device)
        with model.maybe_autocast(): # Ensure model.maybe_autocast() is appropriate here
            image_embeds = model.ln_vision(model.visual_encoder(batch_img_for_model))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_feats = model.vision_proj(query_output.last_hidden_state)
        features.append(image_feats.cpu().half()) # Using .half() for memory, ensure it's acceptable
    
    if not features:
        return np.array([])
        
    features = torch.cat(features, dim=0)
    return features.numpy()


if __name__ == '__main__':
    # 配置参数
    datasets = [
        {
            'name': 'SumMe',
            'input_root': '/root/autodl-tmp/data/SumMe/videos',
            'save_root': '/root/autodl-tmp/data/SumMe/features/visual'
        },
        {
            'name': 'TVSum',
            'input_root': '/root/autodl-tmp/data/TVSum/videos',
            'save_root': '/root/autodl-tmp/data/TVSum/features/visual'
        }
    ]
    frame_interval = 15
    batch_size = 256

    # 处理每个数据集
    for dataset in datasets:
        print(f"\n开始处理数据集: {dataset['name']}")
        print(f"输入目录: {dataset['input_root']}")
        print(f"输出目录: {dataset['save_root']}")

        input_root = dataset['input_root']
        save_root = dataset['save_root']

        # 检查输入目录是否存在
        if not os.path.exists(input_root):
            print(f"警告: 输入目录 {input_root} 不存在，跳过数据集 {dataset['name']}")
            continue

        os.makedirs(save_root, exist_ok=True)
        videos = sorted(os.listdir(input_root))

        print(f"找到 {len(videos)} 个视频文件")

        # random.shuffle(videos)
        for video_name in tqdm(videos, desc=f"处理 {dataset['name']}"):
            vid = video_name.split('.mp4')[0]
            save_path = os.path.join(save_root, vid+'.npy')
            if os.path.exists(save_path):
                continue
            video_path = os.path.join(input_root, video_name)
            features = get_visual_features(
                video_path, frame_interval=frame_interval, batch_size=batch_size)
            np.save(save_path, features)

        print(f"数据集 {dataset['name']} 处理完成！")
