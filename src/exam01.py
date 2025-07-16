"""
视频摘要流程
1 非均匀帧采样
2 新的QueryLLM的流程
3 相似度计算
4 评估

step01_nfs: 已找到文献Too Many Frames, Not All Useful: Efficient Strategies for Long-Form Video QA
使用该文献的开源算法

step02_query_llm：查询大模型，拟采用M3SUM: A NOVEL UNSUPERVISED LANGUAGE-GUIDED VIDEO SUMMARIZATION
的开源算法
"""

import json
from openai import OpenAI
from model.non_uniform_frame_sampling import temporal_scene_clustering_used
import warnings
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")


def step01_nfs(
    summe_frame_dir,
    tvsum_frame_dir,
    nfs_dir,
    batch_size,
    frame_interval
):
    temporal_scene_clustering_used(
        summe_frame_dir, nfs_dir, batch_size, "SumMe", frame_interval)
    temporal_scene_clustering_used(
        tvsum_frame_dir, nfs_dir, batch_size, "TVSum", frame_interval)


def step02_query_llm(
    data_dir="/root/autodl-tmp/data",
    nfs_dir="/root/tfnet/out",
    process_prompt="",
    llm_query_out_dir="/root/tfnet/out/llm_query_out",
):
    client = OpenAI(api_key="sk-e79076e67324476095076a50bf2a3f12",
                    base_url="https://api.deepseek.com")
    summe_nfs_json = Path(nfs_dir, "SumMe_nfs_output.json")
    tvsum_nfs_json = Path(nfs_dir, "TVSum_nfs_output.json")
    summe_frame_caption_dir = Path(data_dir, "SumMe", "captions", "Blip2")
    tvsum_frame_caption_dir = Path(data_dir, "TVSum", "captions", "Blip2")
    """
    nfs_json 格式
    {
        video_name: {
            out_frames: a list of frame paths,
            output_result: 聚类的结果list,
            num_frame_picks: 一个int数字
        },
        ...
    }
    """
    with open(summe_nfs_json, "r") as f:
        summe_nfs_json = json.load(f)
    with open(tvsum_nfs_json, "r") as f:
        tvsum_nfs_json = json.load(f)

    def process_video(out_frames, frame_caption_json, process_prompt, video_caption):
        llm_query_out = {}
        for frame_path in tqdm(out_frames, desc="Processing frames"):
            frame_idx = int(Path(frame_path).name.split(".")[0].split("_")[-1])
            caption = frame_caption_json[str(frame_idx)]

            prompt = process_prompt + "Video Caption: " + video_caption + "\n"
            prompt += f"Frame {frame_idx}: {caption}\n"
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "user", "content": f"{prompt}"},
                    ]
                )
                llm_query_out[str(frame_idx)] = response.choices[0].message.content
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                llm_query_out[str(frame_idx)] = "0"
        return llm_query_out

    def process_dataset(dataset_name, nfs_json, frame_caption_dir, llm_query_out_dir):
        llm_query_out = {}
        for video_name, video_info in tqdm(nfs_json.items(), desc=f"Processing {dataset_name} videos"):
            out_frames = video_info["out_frames"]
            # output_result = video_info["output_result"]
            # num_frame_picks = video_info["num_frame_picks"]
            frame_caption_json = Path(frame_caption_dir, f"{video_name}.json")
            video_caption = Path(frame_caption_dir).parent / "Qwen25VL" / f"{video_name}.txt"

            with open(video_caption, "r") as f:
                video_caption = f.read().strip()

            with open(frame_caption_json, "r") as f:
                frame_caption_json = json.load(f)

            response_content = process_video(
                out_frames, frame_caption_json, process_prompt, video_caption)
            llm_query_out[video_name] = response_content

        # 确保输出目录存在
        Path(llm_query_out_dir).mkdir(parents=True, exist_ok=True)
        
        llm_query_out_json = Path(
            llm_query_out_dir, f"{dataset_name}_llm_query_out.json")
        with open(llm_query_out_json, "w") as f:
            json.dump(llm_query_out, f, indent=4)

    process_dataset("SumMe", summe_nfs_json,
                    summe_frame_caption_dir, llm_query_out_dir)
    process_dataset("TVSum", tvsum_nfs_json,
                    tvsum_frame_caption_dir, llm_query_out_dir)


if __name__ == "__main__":

    data_dir = "/root/autodl-tmp/data"
    frame_dir_summe = Path(data_dir, "SumMe", "frames")
    frame_dir_tvsum = Path(data_dir, "TVSum", "frames")
    nfs_dir = "/root/tfnet/out"
    llm_query_out_dir = Path(nfs_dir, "llm_query_out")

    process_prompt = r"""Please help me evaluate the importance of a certain frame in a video summarization task.
    1. First, I will give you the descriptive text of the video in the following format.
    "Video Caption: a plane fly in sky......"
    2. Secondly, I will give you the Caption of the frame in the following format.
    "Frame 32: a person jumps from a tower."
    3. You need to evaluate the importance of the frame in the video summarization task, ranging from 0 to 1, with an interval of 0.1.
    4. Finally, you need to output and only output a number.
    Now start the task, your reference content is as follows:\n
    """

    # step01_nfs(
    #     frame_dir_summe=frame_dir_summe,
    #     frame_dir_tvsum=frame_dir_tvsum,
    #     nfs_dir=nfs_dir,
    #     batch_size=2048,
    #     frame_interval=5
    # )

    step02_query_llm(
        data_dir=data_dir,
        nfs_dir=nfs_dir,
        process_prompt=process_prompt,
        llm_query_out_dir=llm_query_out_dir,
    )

