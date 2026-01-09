# 聚合每个视频的caption以获取整个视频的文本摘要
# 使用messages格式传入LLM客户端

from tqdm import tqdm
import os
import json
import sys
import time
sys.path.append("/root/tfnet/")
from src.util.llm import LLMBaseClient, OpenAIClient


def summarize_video_captions(input_file, output_file, llm_client: LLMBaseClient, max_retries=3, retry_delay=5):
    """
    聚合每个视频的caption生成视频摘要

    Args:
        input_file: 输入的caption JSON文件路径
        output_file: 输出的摘要JSON文件路径  
        llm_client: LLMBaseClient实例
        max_retries: 最大重试次数
        retry_delay: 重试间隔时间（秒）
    """
    # 加载caption数据
    with open(input_file, 'r', encoding='utf-8') as f:
        caption_data = json.load(f)

    summaries = {}

    # 如果输出文件已存在，加载已处理的数据以支持断点续传
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            summaries = json.load(f)
        print(
            f"Found existing output file, loaded {len(summaries)} existing summaries")

    # 对每个视频生成摘要
    for video_name, video_data in tqdm(caption_data.items(), desc="Generating summary from frame captions"):
        # 跳过已处理的视频
        if video_name in summaries:
            continue

        if 'captions' not in video_data:
            print(f"Warning: No captions found for video {video_name}")
            continue

        captions = video_data['captions']
        # 将所有帧的caption用换行符拼接
        file_content = "\n".join(captions)

        # 构建messages格式
        messages = [
            {
                "role": "system",
                "content": file_content,  # 视频帧描述内容
            },
            {
                "role": "user",
                "content": "This content contains descriptive text for the frame images of a video. Please summarize the main content of the video based on the descriptive text in a linear fashion and do not divide the points."
            }
        ]

        # 重试机制
        for attempt in range(max_retries):
            try:
                summary = llm_client.generate({"messages": messages})
                summaries[video_name] = {
                    'summary': summary.strip(),
                    'frame_count': len(captions)
                }

                # 每处理一个视频就保存一次，防止数据丢失
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)

                break  # 成功则跳出重试循环

            except Exception as e:
                print(
                    f"Error processing video {video_name} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(
                        f"Failed to process video {video_name} after {max_retries} attempts")
                    # 可以选择跳过该视频继续处理其他视频
                    continue

    print(f"Summaries saved to {output_file}")


def test():
    api_key = "sk-dd060a90600d43f4923e24908eddee16"
    test_data = {
        "test_video_1": {
            "picks": [0, 15, 30, 45],
            "captions": [
                "a cat sitting on a wooden floor",
                "a cat playing with a toy mouse",
                "a cat eating from a bowl",
                "a cat sleeping on a couch"
            ]
        },
        "test_video_2": {
            "picks": [0, 10, 20, 30],
            "captions": [
                "a man walking down a busy street",
                "the man enters a coffee shop",
                "he orders coffee at the counter",
                "he sits and drinks coffee while reading"
            ]
        },
        "test_video_3": {
            "picks": [0, 8, 16, 24, 32],
            "captions": [
                "a chef preparing ingredients in the kitchen",
                "chopping vegetables on a cutting board",
                "cooking pasta in boiling water",
                "adding sauce to the pasta",
                "plating the finished dish"
            ]
        },
        "test_video_4": {
            "picks": [0, 12, 24],
            "captions": [
                "children playing soccer in a park",
                "one child kicks the ball towards the goal",
                "the goalkeeper catches the ball"
            ]
        },
        "test_video_5": {
            "picks": [0, 5, 10, 15, 20, 25],
            "captions": [
                "a woman jogging along a forest trail",
                "she stops to drink water from a bottle",
                "continues jogging uphill through trees",
                "reaches a scenic viewpoint",
                "takes photos of the landscape",
                "begins jogging back down the trail"
            ]
        }
    }

    test_input = "/tmp/test_captions.json"
    test_output = "/root/tfnet/data/captions/video_caption/test_summaries.json"

    with open(test_input, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)

    client = OpenAIClient(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1", model="deepseek-chat"
    )

    summarize_video_captions(test_input, test_output, client)

    os.remove(test_input)


if __name__ == "__main__":

    summe_captions_json = "/root/tfnet/data/captions/frame_caption/blip/summe_captions.json"
    tvsum_captions_json = "/root/tfnet/data/captions/frame_caption/blip/tvsum_captions.json"

    out_dir = "/root/tfnet/data/captions/video_caption/gpt-41"
    os.makedirs(out_dir, exist_ok=True)

    summe_summary_json = os.path.join(out_dir, "summe_summary_gpt41.json")
    tvsum_summary_json = os.path.join(out_dir, "tvsum_summary_gpt41.json")

    deepseek = OpenAIClient(
        api_key="",
        base_url="https://api.deepseek.com/v1", model="deepseek-chat"
    )
    
    moonshot = OpenAIClient(
        api_key="",
        base_url="https://api.moonshot.cn/v1", model="moonshot-v1-128k"
    )
    
    gpt4 = OpenAIClient(
        api_key="",
        base_url="https://api.openai.com/v1", model="gpt-4"
    )
    
    summarize_video_captions(summe_captions_json, summe_summary_json, gpt4)
    summarize_video_captions(tvsum_captions_json, tvsum_summary_json, gpt4)
