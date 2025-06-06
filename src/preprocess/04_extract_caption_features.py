#!/usr/bin/env python3
"""
字幕文本特征提取脚本，使用 BLIP-2 模型。
用法: python 04_extract_caption_features.py <选项>
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
    parser = argparse.ArgumentParser(description="使用 BLIP-2 模型提取字幕文本特征。")
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
    args = parser.parse_args()
    return args

def main():
    """主函数，为指定数据集中的字幕提取文本特征。"""
    args = arg_parser()

    # 初始化 BLIP2FeatureExtractor
    extractor = BLIP2FeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )

    datasets = [
        {
            "name": "SumMe",
            "captions_path": "/root/autodl-tmp/data/SumMe/captions/Qwen25VL", # SumMe 数据集字幕路径
            "feature_output_path": "/root/autodl-tmp/data/SumMe/text_features/BLIP2" # SumMe 文本特征输出路径
        },
        {
            "name": "TVSum",
            "captions_path": "/root/autodl-tmp/data/TVSum/captions/Qwen25VL", # TVSum 数据集字幕路径
            "feature_output_path": "/root/autodl-tmp/data/TVSum/text_features/BLIP2" # TVSum 文本特征输出路径
        }
    ]

    for dataset in datasets:
        print(f"正在处理数据集: {dataset['name']}")
        captions_dir = dataset["captions_path"]
        output_dir = dataset["feature_output_path"]

        os.makedirs(output_dir, exist_ok=True) # 创建输出目录，如果不存在

        if not os.path.isdir(captions_dir):
            print(f"警告: 未找到 {dataset['name']} 的字幕目录: {captions_dir}。正在跳过。")
            continue

        # 使用 BLIP2FeatureExtractor 从字幕中提取文本特征
        extractor.extract_caption_features(
            captions_folder=captions_dir,
            output_folder=output_dir
        )

    print("字幕文本特征提取完成。")


if __name__ == "__main__":
    main()