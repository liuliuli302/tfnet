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
model, vis_processors, text_processors = load_model_and_preprocess(
    "blip2_image_text_matching", "coco", device=device, is_eval=True)
vis_processors = transforms.Compose([
    t for t in vis_processors['eval'].transform.transforms if not isinstance(t, transforms.ToTensor)
])


def loadvideo(fname, frame_interval=15, max_duration=None):
    vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    all_index = np.arange(0, len(vr), frame_interval, dtype=np.int32)
    if max_duration is not None:
        max_frames = round(max_duration * len(vr) / duration)
        all_index = all_index[:max_frames // frame_interval]
    vr.seek(0)
    buffer = vr.get_batch(all_index).permute(0, 3, 1, 2) / 255.
    return buffer


@torch.no_grad()
def get_visual_features(video_path, frame_interval=None, max_duration=None, batch_size=128):
    video = loadvideo(video_path, frame_interval, max_duration)
    img = vis_processors(video)
    features = []
    for bid in range(0, img.size(0), batch_size):
        batch_img = img[bid:bid+batch_size].to(device)
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(batch_img))
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
        features.append(image_feats.cpu().half())
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
    batch_size = 128

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
