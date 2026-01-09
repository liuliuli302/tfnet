from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
import json
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
sys.path.append("/root/tfnet/")


async def query_llm_text_async(model, query_type, prompt, frame_caption, video_summary):
    """
    异步查询 LLM
    """
    query_type_list = query_type.split("_")
    # 文本查询
    if query_type_list[1] == "ws":
        # 有摘要
        content = prompt.format(
            frame_caption=frame_caption,
            video_summary=video_summary,
        )
    else:
        # 无摘要
        content = prompt.format(
            frame_caption=frame_caption,
        )
    input_data = {
        "messages": [{"role": "user", "content": content}]
    }
    llm_out = await model.generate_async(input_data)
    return llm_out, content


async def process_video_text_async(model, query_type, prompt, frame_caption_file, video_caption_file, video_name, semaphore):
    """
    异步处理单个视频的文本查询
    """
    frame_caption = frame_caption_file[video_name]["captions"]
    frame_picks = frame_caption_file[video_name]["picks"]
    video_caption = video_caption_file[video_name]["summary"]

    async def process_single_frame(frame_idx, caption):
        async with semaphore:  # 限制并发数量
            result, content = await query_llm_text_async(
                model, query_type, prompt, caption, video_caption)
            return {
                "frame_idx": frame_idx,
                "query": content,
                "llm_output": result
            }

    # 创建任务列表
    tasks = [process_single_frame(frame_idx, caption)
             for frame_idx, caption in zip(frame_picks, frame_caption)]

    # 使用 tqdm.asyncio 显示异步任务进度
    llm_out_result = await atqdm.gather(*tasks, desc=f"Processing {video_name}")

    return {video_name: llm_out_result}


async def process_dataset_text_async(model, query_type, prompt, frame_caption_file, video_caption_file, max_concurrent=5):
    """
    异步处理数据集文本查询

    Args:
        max_concurrent: 最大并发数量，控制同时进行的请求数
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建视频处理任务
    tasks = []
    video_names = list(frame_caption_file.keys())

    for video_name in video_names:
        task = process_video_text_async(
            model, query_type, prompt, frame_caption_file,
            video_caption_file, video_name, semaphore)
        tasks.append(task)
        # break  # 保持原有逻辑，只处理第一个视频

    # 使用 tqdm.asyncio 显示数据集处理进度
    results = await atqdm.gather(*tasks, desc="Processing dataset")
    return results

# 保持原有同步函数不变，以保证兼容性


def query_llm_text(model, query_type, prompt, frame_caption, video_summary):
    """
    同步版本的查询函数（保持向后兼容）
    """
    query_type_list = query_type.split("_")
    # 文本查询
    if query_type_list[1] == "ws":
        # 有摘要
        content = prompt.format(
            frame_caption=frame_caption,
            video_summary=video_summary,
        )
    else:
        # 无摘要
        content = prompt.format(
            frame_caption=frame_caption,
        )
    input_data = {
        "messages": [{"role": "user", "content": content}]
    }
    llm_out = model.generate(input_data)
    return llm_out, content


def process_video_text(model, query_type, prompt, frame_caption_file, video_caption_file, video_name):
    """
    同步版本的视频处理函数（保持向后兼容）
    """
    # 处理视频文本查询
    llm_out_result = []

    frame_caption = frame_caption_file[video_name]["captions"]
    frame_picks = frame_caption_file[video_name]["picks"]

    video_caption = video_caption_file[video_name]["summary"]

    for frame_idx, caption in tqdm(zip(frame_picks, frame_caption), desc=f"Processing {video_name}", leave=False):
        # 对每个frame_caption进行查询
        result, content = query_llm_text(
            model, query_type, prompt, caption, video_caption)
        llm_out_result.append({
            "frame_idx": frame_idx,
            "query": content,
            "llm_output": result
        })

    return {video_name: llm_out_result}


def process_dataset_text(model, query_type, prompt, frame_caption_file, video_caption_file):
    """
    同步版本的数据集处理函数（保持向后兼容）
    """
    # 检查模型是否支持异步
    if hasattr(model, 'generate_async'):
        # 使用异步版本
        return asyncio.run(process_dataset_text_async(
            model, query_type, prompt, frame_caption_file, video_caption_file))
    else:
        # 使用原有同步版本
        llm_out_result = []
        for video_name, caption in frame_caption_file.items():
            result = process_video_text(
                model, query_type, prompt, frame_caption_file, video_caption_file, video_name)
            llm_out_result.append(result)
            break
        return llm_out_result


def save_result(results, output_path):
    """
    保存重要性分数结果到JSON文件

    Args:
        results: 重要性分数结果字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")


def test_module():
    """
    测试此模块中的函数.
    """
    # 1. 导入必要的模型
    # 注意：确保 llm.py 在 Python 路径中，或者使用相对导入
    from src.util.llm import moonshot

    # 2. 准备模拟数据
    mock_model = moonshot
    mock_query_type = "text_ws"  # 带摘要的文本查询
    mock_prompt = "基于以下视频摘要：'{video_summary}'，请判断以下帧描述：'{frame_caption}'，对于总结整个视频的重要性，并给出一个0-100之间的分数。请输出并仅仅输出一个分数"

    mock_frame_captions = {
        "video1": {
            "captions": ["一个人在切菜", "一个人在炒菜"],
            "picks": [10, 25]
        }
    }
    mock_video_captions = {
        "video1": {
            "summary": "这个视频展示了如何做一道家常菜。"
        }
    }

    print("--- 开始测试 process_dataset_text ---")

    # 3. 调用待测试的函数
    results = process_dataset_text(
        model=mock_model,
        query_type=mock_query_type,
        prompt=mock_prompt,
        frame_caption_file=mock_frame_captions,
        video_caption_file=mock_video_captions
    )

    # 4. 打印并验证结果
    print("\n--- 测试完成 ---")
    print("LLM返回结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    # 检查结果结构
    assert isinstance(results, list)
    assert "video1" in results[0]
    print("\n测试函数结构正确。")


if __name__ == "__main__":
    test_module()
