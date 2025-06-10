# 文本特征提取模块，基于BLIP2模型提取文本特征

import json
import os

os.environ['XDG_CACHE_HOME'] = '/root/autodl-tmp/.cache'

from tqdm import tqdm
import torch
import pickle
import numpy as np
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
from pathlib import Path
import glob

device = 'cuda'
model, vis_processors, text_processors = load_model_and_preprocess(
    "blip2_image_text_matching", "coco", device=device, is_eval=True)

@torch.no_grad()
def get_text_features(caption, max_length=35):
    """
    提取单个文本caption的特征
    
    Args:
        caption: 单个文本caption字符串
        max_length: 最大文本长度
    
    Returns:
        numpy array: 文本特征 (1, 256)
    """
    if not isinstance(caption, str):
        raise ValueError("caption必须是字符串类型")
    
    # 文本tokenization
    text = model.tokenizer(
        [caption],  # 转换为单元素列表
        padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_tensors="pt"
    ).to(device)
    
    # 通过Q-Former提取文本特征
    with model.maybe_autocast():
        text_output = model.Qformer.bert(
            text.input_ids, 
            attention_mask=text.attention_mask, 
            return_dict=True
        )
        # 使用[CLS] token的特征 (第0个位置)
        text_feat = model.text_proj(text_output.last_hidden_state[:, 0, :])
    
    return text_feat.cpu().half().numpy()

def load_text_from_json(json_path):
    """从JSON文件加载文本数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # 如果是字典，尝试提取文本内容
        texts = []
        for key, value in data.items():
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, list):
                texts.extend([str(item) for item in value])
        return texts
    else:
        raise ValueError(f"不支持的JSON格式: {type(data)}")

def load_text_from_txt(txt_path):
    """从TXT文件加载文本数据"""
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def load_captions_from_directory(caption_dir, video_name):
    """
    从caption目录加载指定视频的caption文本
    支持多种文件格式：.json, .txt
    
    Returns:
        str: 单个caption字符串，如果有多个caption则合并
    """
    vid = video_name.split('.mp4')[0] if '.mp4' in video_name else video_name
    
    # 尝试不同的文件扩展名
    for ext in ['.json', '.txt']:
        caption_path = os.path.join(caption_dir, vid + ext)
        if os.path.exists(caption_path):
            if ext == '.json':
                captions_list = load_text_from_json(caption_path)
            else:
                captions_list = load_text_from_txt(caption_path)
            
            # 如果有多个caption，合并为一个长文本
            if isinstance(captions_list, list) and len(captions_list) > 0:
                if len(captions_list) == 1:
                    return captions_list[0]
                else:
                    # 合并多个caption为一个长文本
                    return " ".join(captions_list)
            elif isinstance(captions_list, str):
                return captions_list
    
    return ""

if __name__ == '__main__':
    # 配置参数
    datasets = [
        {
            'name': 'SumMe',
            'caption_root': '/root/autodl-tmp/data/SumMe/captions/Qwen25VL/',
            'save_root': '/root/autodl-tmp/data/SumMe/features/text',
            'video_list_path': '/root/autodl-tmp/data/SumMe/videos'
        },
        {
            'name': 'TVSum',
            'caption_root': '/root/autodl-tmp/data/TVSum/captions/Qwen25VL/',
            'save_root': '/root/autodl-tmp/data/TVSum/features/text',
            'video_list_path': '/root/autodl-tmp/data/TVSum/videos'
        }
    ]
    
    max_length = 500
    
    # 处理每个数据集
    for dataset in datasets:
        caption_root = dataset['caption_root']
        save_root = dataset['save_root']
        video_list_path = dataset['video_list_path']
        
        # 检查目录是否存在
        if not os.path.exists(caption_root) or not os.path.exists(video_list_path):
            continue
            
        os.makedirs(save_root, exist_ok=True)
        
        # 获取视频列表
        videos = sorted([f for f in os.listdir(video_list_path) if f.endswith('.mp4')])
        
        # 使用tqdm显示进度条
        for video_name in tqdm(videos, desc=f"处理 {dataset['name']}"):
            vid = video_name.split('.mp4')[0]
            save_path = os.path.join(save_root, vid + '.npy')
            
            if os.path.exists(save_path):
                continue
                
            # 加载caption文本
            caption = load_captions_from_directory(caption_root, video_name)
            
            if not caption:
                continue
            
            # 提取文本特征
            features = get_text_features(caption, max_length=max_length)
            np.save(save_path, features)
