#!/usr/bin/env python3
"""
视频字幕脚本，使用 Qwen2.5-VL 模型。
用法: python 02_caption_videos.py <选项>
"""

import os
import sys
import argparse
import torch 
from tqdm import tqdm # 导入 tqdm
import warnings # 导入 warnings 模块
warnings.filterwarnings("ignore") # 忽略所有警告 - 移动到全局作用域
 
# 将父目录 (src) 添加到 sys.path 以便导入 handler.llm_client
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from handler.llm_client import Qwen25VLClient
# from util.captioner import caption_videos # 原始导入，被 Qwen25VLClient 逻辑替换

def arg_parser():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 Qwen2.5-VL 模型为视频添加字幕。")
    parser.add_argument(
        '--model_name', 
        type=str, 
        default="Qwen/Qwen2.5-VL-7B-Instruct", 
        help='要使用的 Qwen 模型名称 (默认: Qwen/Qwen2.5-VL-7B-Instruct)。'
    )
    parser.add_argument(
        '--device_map', 
        type=str, 
        default="auto", 
        help='模型的设备映射 (例如 "auto", "cuda:0", 默认: "auto")。'
    )
    parser.add_argument(
        '--caption_prompt',
        type=str,
        default="Describe this video in detail.",
        help="用于视频字幕的提示 (默认: 'Describe this video in detail.')。"
    )
    parser.add_argument(
        '--torch_dtype',
        type=str,
        default=None, # 如果未指定，则为 None, Qwen25VLClient 会处理 None
        help="模型加载的 Torch dtype，例如 'torch.bfloat16'。如果未设置，将使用模型默认值。"
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default=None, # 如果未指定，则为 None
        help="Attention 实现，例如 'flash_attention_2'。如果未设置，将使用模型默认值。"
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=8, # 默认帧数，可以根据需要调整
        help="从视频中采样的帧数 (默认: 8)。"
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=1024, 
        help="生成新 token 的最大数量 (默认: 1024)。"
    )
    args = parser.parse_args()

    # 如果指定，则将字符串 torch_dtype 转换为实际的 torch 对象
    if args.torch_dtype:
        if args.torch_dtype == 'torch.bfloat16':
            args.torch_dtype = torch.bfloat16
        elif args.torch_dtype == 'torch.float16':
            args.torch_dtype = torch.float16
        # 如果需要，添加更多 dtype
        else:
            print(f"警告: 不支持的 torch_dtype 字符串 '{args.torch_dtype}'。使用模型默认值。")
            args.torch_dtype = None
    return args

def main():
    """主函数，为指定数据集中的视频添加字幕。"""
    args = arg_parser()

    # 初始化 Qwen25VLClient
    # 注意: Qwen25VLClient期望 torch_dtype 是一个 torch 对象，而不是字符串。
    # arg_parser 处理基本的字符串到 torch.dtype 的转换。
    client = Qwen25VLClient(
        model_name=args.model_name,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation
    )

    datasets = [
        {
            "name": "SumMe",
            "video_path": "/root/autodl-tmp/data/SumMe/videos", # SumMe 数据集视频路径
            "caption_output_path": "/root/autodl-tmp/data/SumMe/captions/Qwen25VL" # SumMe 字幕输出路径
        },
        {
            "name": "TVSum",
            "video_path": "/root/autodl-tmp/data/TVSum/videos", # TVSum 数据集视频路径 (已修正)
            "caption_output_path": "/root/autodl-tmp/data/TVSum/captions/Qwen25VL" # TVSum 字幕输出路径
        }
    ]

    for dataset in datasets:
        print(f"正在处理数据集: {dataset['name']}")
        video_dir = dataset["video_path"]
        output_dir = dataset["caption_output_path"]

        os.makedirs(output_dir, exist_ok=True) # 创建输出目录，如果不存在

        if not os.path.isdir(video_dir):
            print(f"警告: 未找到 {dataset['name']} 的视频目录: {video_dir}。正在跳过。")
            continue

        video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f)) and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"在 {video_dir} 中未找到视频文件。正在跳过数据集 {dataset['name']}。")
            continue

        for video_filename in tqdm(video_files, desc=f"为 {dataset['name']} 添加字幕"): # 使用 tqdm 包装
            video_filepath = os.path.join(video_dir, video_filename) # 视频文件的完整路径
            
            # 以下检查现在在 video_files 列表推导式中处理，但保留以防万一
            # if not os.path.isfile(video_filepath):
            #     # 如果不是文件 (例如子目录)，则跳过
            #     print(f"正在跳过非文件项: {video_filepath}")
            #     continue

            # # 对常见视频扩展名的基本检查，可以扩展
            # if not video_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            #     print(f"正在跳过非视频文件 (按扩展名): {video_filepath}")
            #     continue
            
            # print(f"正在为视频添加字幕: {video_filepath}") # tqdm 会显示进度，所以这个可以注释掉或调整

            # 构建对话内容
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "path": video_filepath},
                        {"type": "text", "text": args.caption_prompt},
                    ],
                }
            ]
            
            caption_list = client.generate_response(conversation, num_frames=args.num_frames, max_new_tokens=args.max_new_tokens) # 生成字幕
            # 如果字幕列表不为空，则取第一个字幕，否则提供默认消息
            caption_text = caption_list[0] if caption_list else f"无法为 {video_filename} 生成字幕。"

            caption_filename = os.path.splitext(video_filename)[0] + ".txt" # 字幕文件名
            caption_filepath = os.path.join(output_dir, caption_filename) # 字幕文件的完整路径

            with open(caption_filepath, "w", encoding="utf-8") as f:
                f.write(caption_text) # 保存字幕
            print(f"字幕已保存到: {caption_filepath}")

    print("视频字幕处理完成。")

if __name__ == '__main__':
    # 如果在 arg_parser 中使用了 torch_dtype 转换，则在此处导入 torch
    # 以避免在未安装 torch 但使用 -h 调用脚本时出现导入错误
    main()