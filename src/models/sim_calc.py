import h5py
import torch
import clip
import os
import numpy as np
from pathlib import Path
import time
import json

from tqdm import tqdm


def cal_consistency_coff(dataset_name, caption_json, exam_name):
    # 读取JSON文件
    with open(caption_json, 'r', encoding='utf-8') as f:
        txt_sum = json.load(f)

    # 构造key到video_name的映射
    key_name_dict = {}

    if dataset_name == 'tvsum':
        num_videos = 50
        # 对于tvsum数据集，使用video_name_dict.json文件
        try:
            video_name_dict_path = '/root/tfnet/data/video_name_dict.json'
            with open(video_name_dict_path, 'r', encoding='utf-8') as f:
                video_name_dict = json.load(f)

            # 直接使用映射：video_name_dict已经是video_name -> video_id的映射
            # 我们需要反转为video_id -> video_name
            for video_name, video_id in video_name_dict.items():
                key_name_dict[video_id] = video_name

            print(f"使用video_name_dict.json构建tvsum映射: {len(key_name_dict)} 个视频")

        except Exception as e:
            print(f"警告: 无法读取video_name_dict.json: {e}, 回退到原始方法")
            # 回退到原始方法
            video_names = sorted(txt_sum.keys())
            video_names.reverse()
            num_videos = 50
            for i in range(num_videos):
                key = f"video_{i+1}"
                key_name_dict[key] = video_names[i]
    else:
        # 对于summe数据集，使用原来的方法
        video_names = sorted(txt_sum.keys())
        num_videos = 25
        for i in range(num_videos):
            key = f"video_{i+1}"
            key_name_dict[key] = video_names[i]

    h5_path = f'data/feature/eccv16_dataset_{dataset_name}_ViT_L_14.h5'
    dataset = h5py.File(h5_path, 'r')

    # 加载CLIP模型
    clip_path = r"ViT-L/14"
    clip_model, preprocess = clip.load(clip_path, device="cuda")

    coffs = {}

    for i in tqdm(range(num_videos), desc=f"Processing {dataset_name} sim"):
        key = f"video_{i+1}"
        d = dataset[key]
        seq = d['features'][...]
        seq = torch.as_tensor(seq).cuda()

        # 通过构造的key从JSON中获取文本
        text_str = txt_sum[key_name_dict[key]]["summary"]
        texts = text_str.split('.')[:-1]

        # 过滤空文本
        texts = [t.strip() for t in texts if t.strip()]

        # 使用truncate=True自动截断
        text_tokens = clip.tokenize(texts, truncate=True).cuda()

        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)

        text_features /= text_features.norm(dim=-1, keepdim=True)
        seq /= seq.norm(dim=-1, keepdim=True)

        similarity_matrix = seq @ text_features.T
        sims, _ = torch.max(similarity_matrix, dim=1)  # 取最大值

        coffs[key_name_dict[key]] = sims.cpu().numpy().tolist()

    # 保存为json
    model_used = os.path.basename(caption_json).split(".")[0].split("_")[-1]
    with open(f"out/{exam_name}/sim/{dataset_name}_{model_used}_sim.json", 'w', encoding='utf-8') as f:
        json.dump(coffs, f, indent=4)
    dataset.close()
    return coffs


if __name__ == "__main__":
    exam = "exam01"

    coffs = cal_consistency_coff(
        dataset_name="summe",
        caption_json="/root/tfnet/data/captions/video_caption/deepseek/summe_summary_deepseek.json",
        exam_name=exam
    )

    coffs2 = cal_consistency_coff(
        dataset_name="tvsum",
        caption_json="/root/tfnet/data/captions/video_caption/deepseek/tvsum_summary_deepseek.json",
        exam_name=exam
    )

    coffs3 = cal_consistency_coff(
        dataset_name="summe",
        caption_json="/root/tfnet/data/captions/video_caption/moonshot/summe_summary_moonshot.json",
        exam_name=exam
    )

    coffs4 = cal_consistency_coff(
        dataset_name="tvsum",
        caption_json="/root/tfnet/data/captions/video_caption/moonshot/tvsum_summary_moonshot.json",
        exam_name=exam
    )
