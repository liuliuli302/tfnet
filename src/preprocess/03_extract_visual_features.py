#!/usr/bin/env python3
"""
视频视觉特征提取脚本，使用 BLIP-2 模型。
用法: python 03_extract_visual_features.py <选项>
"""

import os
import sys
import argparse
import torch 
from tqdm import tqdm # 导入 tqdm
import warnings # 导入 warnings 模块
warnings.filterwarnings("ignore") # 忽略所有警告 - 移动到全局作用域
 
# 将父目录 (src) 添加到 sys.path 以便导入 util.feature_extractor
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.feature_extractor import BLIP2FeatureExtractor

def arg_parser():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 BLIP-2 模型提取视频视觉特征。")
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="Salesforce/blip2-opt-2.7b", 
        help='要使用的 BLIP-2 模型名称 (默认: Salesforce/blip2-opt-2.7b)。'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default=None, 
        help='模型的设备 (例如 "cuda", "cpu", 默认: 自动检测)。'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=15,
        help="帧间隔，即每隔多少帧提取一次特征 (默认: 15，即提取第0、15、30、45...帧)。"
    )
    args = parser.parse_args()
    return args

def main():
    """主函数，为指定数据集中的视频提取视觉特征。"""
    args = arg_parser()

    # 初始化 BLIP2FeatureExtractor
    extractor = BLIP2FeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )

    datasets = [
        {
            "name": "SumMe",
            "frames_path": "/root/autodl-tmp/data/SumMe/frames", # SumMe 数据集帧路径
            "feature_output_path": "/root/autodl-tmp/data/SumMe/visual_features/BLIP2" # SumMe 视觉特征输出路径
        },
        {
            "name": "TVSum",
            "frames_path": "/root/autodl-tmp/data/TVSum/frames", # TVSum 数据集帧路径
            "feature_output_path": "/root/autodl-tmp/data/TVSum/visual_features/BLIP2" # TVSum 视觉特征输出路径
        }
    ]

    for dataset in datasets:
        print(f"正在处理数据集: {dataset['name']}")
        frames_dir = dataset["frames_path"]
        output_dir = dataset["feature_output_path"]

        os.makedirs(output_dir, exist_ok=True) # 创建输出目录，如果不存在

        if not os.path.isdir(frames_dir):
            print(f"警告: 未找到 {dataset['name']} 的帧目录: {frames_dir}。正在跳过。")
            continue

        # 使用 BLIP2FeatureExtractor 从已提取的帧中提取视觉特征
        extractor.extract_visual_features_from_frames(
            frames_folder=frames_dir,
            output_folder=output_dir,
            interval=args.interval
        )

    print("视频视觉特征提取完成。")

if __name__ == '__main__':
    main()