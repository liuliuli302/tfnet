"""
视频摘要流程
1 非均匀帧采样
2 新的QueryLLM的流程
3 相似度计算
4 评估
"""

import json

from openai import OpenAI
from model.non_uniform_frame_sampling import temporal_scene_clustering_used
import warnings
warnings.filterwarnings("ignore")


def step01_nfs():
    frames_dir_summe = "/root/autodl-tmp/data/SumMe/frames"
    frames_dir_tvsum = "/root/autodl-tmp/data/TVSum/frames"
    out_dir = "/root/tfnet/out"
    batch_size = 2048
    frame_interval = 5

    temporal_scene_clustering_used(frames_dir_summe, out_dir, batch_size, "SumMe", frame_interval)
    temporal_scene_clustering_used(frames_dir_tvsum, out_dir, batch_size, "TVSum", frame_interval)


def step02_query_llm(prompt):
    
    client = OpenAI(
        api_key="sk-dd060a90600d43f4923e24908eddee16",
        base_url="https://api.deepseek.com"
    )
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant"},
    #         {"role": "user", "content": "Hello"},
    #     ],
    #     stream=False
    # )
    # print(response.choices[0].message.content)
    
    summe_nfs_json = "/root/tfnet/out/SumMe_nfs_output.json"
    tvsum_nfs_json = "/root/tfnet/out/TVSum_nfs_output.json"
    # 加载这两个json文件
    with open(summe_nfs_json, "r") as f:
        summe_nfs_data = json.load(f)
    with open(tvsum_nfs_json, "r") as f:
        tvsum_nfs_data = json.load(f)
    
    summe_message_list = []
    summe_message_list = []
    
    for video_name, video_data in summe_nfs_data.items():
        message = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        out_frames = video_data["out_frames"]
        output_result = video_data["output_result"]
        num_frame_picks = video_data["num_frame_picks"]
        picks = []
        for frame_name in out_frames:
            # 在这里调用新的QueryLLM的流程
            frame_num = frame_name.split("/")[-1].split(".")[0].split("_")[-1]
            frame_num = int(frame_num)
            picks.append(frame_num)

        
        
        
        
        # 在这里调用新的QueryLLM的流程
    for video_name, video_data in tvsum_nfs_data.items():
        out_frames = video_data["out_frames"]
        output_result = video_data["output_result"]
        num_frame_picks = video_data["num_frame_picks"]
        # 在这里调用新的QueryLLM的流程


if __name__ == "__main__":
    step01_nfs()
