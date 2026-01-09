from tqdm import tqdm
from model.frame_seclection import nfs_from_lvnet
# 添加 process_dataset_text_async
from model.llm_query import process_dataset_text, save_result, process_dataset_text_async
from util.llm import deepseek, moonshot
import argparse
import yaml
import json
import os
from pathlib import Path
from itertools import product
import asyncio


def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/root/tfnet/config/exam01_config.yaml',)
    return parser.parse_args()


if __name__ == "__main__":
    parser = arg_parser()
    config = load_yaml(parser.config)

    # 1 进行NFS非均匀采样
    # 处理summe数据集
    # nfs_from_lvnet(
    #     frames_dir=config["datasets"]["summe"]["frames_dir"],
    #     out_dir=config["nfs"]["out_dir"],
    #     batch_size=config["nfs"]["batch_size"],
    #     dataset_name="summe",
    #     frame_interval=config["nfs"]["frame_interval"],
    #     divlam=config["nfs"]["5"]
    # )
    # 处理tvsum数据集
    # nfs_from_lvnet(
    #     frames_dir=config["datasets"]["tvsum"]["frames_dir"],
    #     out_dir=config["nfs"]["out_dir"],
    #     batch_size=config["nfs"]["batch_size"],
    #     dataset_name="tvsum",
    #     frame_interval=config["nfs"]["frame_interval"],
    #     divlam=config["nfs"]["5"]
    # )

    # 第二部分，查询llm
    """
    1 两个大模型，第一个是moonshot，第二个是deepseek
    2 有大概两种查询提示模板
    3 有四种查询方式，分别是文本，文本对比，视觉，视觉对比
    4 有两种对比的summary的来源，分别是moonshot和deepseek
    所以，总共有2 * 3 * 4 * 2= 48种查询方式
    """

    # 准备模型映射
    model_mapping = {
        "moonshot-v1-128k": moonshot,
        "deepseek-r1": deepseek
    }

    # 准备数据集列表
    datasets = ["summe", "tvsum"]

    # 准备摘要来源列表 (只对带摘要的查询类型有效)
    summary_sources = ["moonshot", "deepseek"]

    # 获取配置参数
    models = config["llmquery"]["models"]["text_modal"]
    query_types = config["llmquery"]["query_type"]
    prompts = config["llmquery"]["prompts"]

    # 创建输出目录
    output_base_dir = "/root/tfnet/out/exam01/llm_query"
    os.makedirs(output_base_dir, exist_ok=True)

    print("开始LLM查询实验...")
    print(
        f"总共需要处理: {len(models)} 模型 × {len(query_types)} 查询类型 × {len(datasets)} 数据集 × {len(summary_sources)} 摘要来源")

    # 使用itertools.product生成所有组合
    combinations = list(
        product(models, query_types, datasets, summary_sources))

    for model_name, query_type, dataset_name, summary_source in tqdm(combinations, leave=True, desc="Processing combinations"):

        # 检查查询类型是否需要摘要
        needs_summary = "_ws_" in query_type

        # 如果不需要摘要但当前是第二个摘要来源，跳过（避免重复）
        if not needs_summary and summary_source != "moonshot":
            continue

        # 构建输出文件名（提前构建用于检查）
        summary_suffix = f"_{summary_source}" if needs_summary else ""
        output_filename = f"{model_name}_{query_type}_{dataset_name}{summary_suffix}.json"
        output_path = os.path.join(output_base_dir, output_filename)

        # 检查文件是否已存在
        if os.path.exists(output_path):
            print(f"文件已存在，跳过: {output_filename}")
            continue

        print(
            f"\n处理组合: {model_name} - {query_type} - {dataset_name} - {summary_source}")

        try:
            # 获取模型
            model = model_mapping[model_name]

            # 获取提示模板
            prompt = prompts[query_type]

            # 加载帧字幕数据
            frame_caption_path = f"/root/tfnet/data/captions/frame_caption/blip/{dataset_name}_captions.json"
            with open(frame_caption_path, 'r', encoding='utf-8') as f:
                frame_captions = json.load(f)

            # 加载视频摘要数据（如果需要）
            video_captions = {}
            # if needs_summary:
            video_caption_path = f"/root/tfnet/data/captions/video_caption/{summary_source}/{dataset_name}_summary_{summary_source}.json"
            with open(video_caption_path, 'r', encoding='utf-8') as f:
                video_captions = json.load(f)

            # 执行查询
            max_concurrent = 20

            if hasattr(model, 'generate_async'):
                # 使用异步版本
                results = asyncio.run(process_dataset_text_async(
                    model=model,
                    query_type=query_type,
                    prompt=prompt,
                    frame_caption_file=frame_captions,
                    video_caption_file=video_captions,
                    max_concurrent=max_concurrent
                ))
            else:
                # 使用同步版本
                results = process_dataset_text(
                    model=model,
                    query_type=query_type,
                    prompt=prompt,
                    frame_caption_file=frame_captions,
                    video_caption_file=video_captions
                )

            # 保存结果
            save_result(results, output_path)

        except Exception as e:
            print(
                f"处理组合 {model_name}-{query_type}-{dataset_name}-{summary_source} 时出错: {str(e)}")
            continue

    print("\nLLM查询实验完成！")
