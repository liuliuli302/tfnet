import asyncio
import argparse
from dataclasses import dataclass
import json
import os
import time
import re
from typing import Dict, Any, List, Union
from tqdm import tqdm
from dotenv import load_dotenv
from src.config.config import BasicConfig
from src.utils.llm_client import OpenAIClient, OpenAIClientConfig


load_dotenv()


@dataclass
class SceneScoreQueryConfig(BasicConfig):
    model_name: str
    scene_score_prompt: Union[str, List[str]]
    llm_client_config_file_path: str
    max_retries: int
    retry_delay: int


class SceneScoreQuery:
    """
    Query LLM for scene importance score based on scene caption and video caption.
    """

    def __init__(self, config: SceneScoreQueryConfig):
        if not isinstance(config, SceneScoreQueryConfig):
            raise TypeError(
                "config must be SceneScoreQueryConfig")
        self.config = config
        self.llm_client_config = OpenAIClientConfig.load_config_from_file(
            config.llm_client_config_file_path
        )

        self.llm_client = OpenAIClient(self.llm_client_config)

    async def query_score_async(self, scene_caption: str, video_caption: str) -> str:
        prompt_template = self.config.scene_score_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        prompt = prompt_template.replace(
            "{scene_caption}", scene_caption).replace(
            "{video_caption}", video_caption)

        messages = [{"role": "user", "content": prompt}]

        response = await self.llm_client.generate_async(
            {"messages": messages},
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay
        )
        return response.strip()

    async def batch_query_scores_async(self, scene_captions: List[str], video_captions: List[str]) -> List[str]:
        """
        Batch version of query_score_async using llm_client's batch_generate.
        Lengths of scene_captions and video_captions must match.
        """
        if len(scene_captions) != len(video_captions):
            raise ValueError(
                f"scene_captions length {len(scene_captions)} != video_captions length {len(video_captions)}")

        prompt_template = self.config.scene_score_prompt
        if isinstance(prompt_template, list):
            prompt_template = "\n".join(prompt_template)

        prompts = []
        for scene_cap, video_cap in zip(scene_captions, video_captions):
            prompt = prompt_template.replace(
                "{scene_caption}", scene_cap).replace(
                "{video_caption}", video_cap)
            prompts.append(prompt)

        # Call the batch API (带重试&失败填空)
        results, failed = await self.llm_client.batch_generate(
            prompts,
            retries=self.config.max_retries,
            retry_delay=self.config.retry_delay,
            return_failed=True
        )
        if failed:
            print(f"batch_query_scores_async failed indices: {failed}")
        return [(res.strip() if res else "") for res in results]


def _parse_score_to_1_100(raw_text: str) -> str:
    if raw_text is None:
        return "0"
    text = str(raw_text).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return "0"
    try:
        value = float(match.group(0))
    except Exception:
        return "0"

    value = max(1.0, min(100.0, value))
    return str(int(round(value)))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _iter_scene_caption_entries(scene_payload: Any):
    if isinstance(scene_payload, dict):
        for scene_key, scene_info in scene_payload.items():
            if isinstance(scene_info, dict):
                scene_caption = scene_info.get("scene_caption", "")
                scene_idx = scene_info.get("scene_idx", scene_key)
                yield str(scene_idx), str(scene_caption)
    elif isinstance(scene_payload, list):
        for idx, item in enumerate(scene_payload):
            if isinstance(item, dict):
                scene_caption = item.get("scene_caption", "")
            else:
                scene_caption = str(item)
            yield str(idx), str(scene_caption)


async def _run_dataset_query(
    dataset_name: str,
    scene_caption_json: Dict[str, Any],
    video_caption_json: Dict[str, str],
    output_file: str,
    scene_score_query: SceneScoreQuery,
    batch_size: int = 64
):
    existing_output = _load_json(
        output_file) if os.path.exists(output_file) else {}
    output_scores = existing_output if isinstance(
        existing_output, dict) else {}

    pending_tasks = []
    skipped_no_video_caption = 0

    video_iter = tqdm(
        scene_caption_json.items(),
        total=len(scene_caption_json),
        desc=f"[{dataset_name}] Scan videos",
        leave=False,
        position=1
    )

    for video_name, scene_payload in video_iter:
        video_caption = video_caption_json.get(video_name, "")
        if not video_caption:
            skipped_no_video_caption += 1
            continue

        if video_name not in output_scores or not isinstance(output_scores[video_name], dict):
            output_scores[video_name] = {}

        for scene_idx, scene_caption in _iter_scene_caption_entries(scene_payload):
            if scene_idx in output_scores[video_name] and str(output_scores[video_name][scene_idx]).strip() != "":
                continue
            pending_tasks.append(
                (video_name, scene_idx, scene_caption, video_caption))

    print(
        f"[{dataset_name}] total_videos={len(scene_caption_json)}, "
        f"pending_scene_queries={len(pending_tasks)}, "
        f"skipped_no_video_caption={skipped_no_video_caption}"
    )

    if not pending_tasks:
        _save_json(output_file, output_scores)
        return {
            "dataset": dataset_name,
            "pending": 0,
            "saved_file": output_file,
            "skipped_no_video_caption": skipped_no_video_caption,
        }

    total_batches = (len(pending_tasks) + batch_size - 1) // batch_size
    batch_iter = tqdm(
        range(0, len(pending_tasks), batch_size),
        total=total_batches,
        desc=f"[{dataset_name}] Query batches",
        leave=False,
        position=2
    )

    for start in batch_iter:
        batch = pending_tasks[start:start + batch_size]
        scene_caps = [item[2] for item in batch]
        video_caps = [item[3] for item in batch]

        raw_scores = await scene_score_query.batch_query_scores_async(scene_caps, video_caps)

        for (video_name, scene_idx, _, _), raw_score in zip(batch, raw_scores):
            output_scores[video_name][scene_idx] = _parse_score_to_1_100(
                raw_score)

        _save_json(output_file, output_scores)

    return {
        "dataset": dataset_name,
        "pending": len(pending_tasks),
        "saved_file": output_file,
        "skipped_no_video_caption": skipped_no_video_caption,
    }


async def _main_async(args):
    base_cfg = SceneScoreQueryConfig.load_config_from_file(
        args.scene_score_config)

    run_cfg = SceneScoreQueryConfig(
        model_name=args.model_name,
        scene_score_prompt=base_cfg.scene_score_prompt,
        llm_client_config_file_path=args.llm_client_config,
        max_retries=base_cfg.max_retries,
        retry_delay=base_cfg.retry_delay,
    )

    query_engine = SceneScoreQuery(run_cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_list = [x.strip().lower()
                    for x in args.datasets.split(",") if x.strip()]

    all_stats = []
    start_time = time.time()

    dataset_iter = tqdm(
        dataset_list,
        total=len(dataset_list),
        desc="Datasets",
        position=0
    )

    for dataset_name in dataset_iter:
        dataset_iter.set_postfix_str(dataset_name)
        scene_caption_file = os.path.join(
            args.scene_caption_dir,
            f"{dataset_name}_scene_captions.json"
        )
        video_caption_file = os.path.join(
            args.video_caption_dir,
            f"{dataset_name}_video_captions.json"
        )
        output_file = os.path.join(
            args.output_dir,
            f"{dataset_name}_scene_scores.json"
        )

        if not os.path.exists(scene_caption_file):
            print(
                f"[{dataset_name}] skip, scene caption file not found: {scene_caption_file}")
            continue
        if not os.path.exists(video_caption_file):
            print(
                f"[{dataset_name}] skip, video caption file not found: {video_caption_file}")
            continue

        scene_caption_json = _load_json(scene_caption_file)
        video_caption_json = _load_json(video_caption_file)

        stats = await _run_dataset_query(
            dataset_name=dataset_name,
            scene_caption_json=scene_caption_json,
            video_caption_json=video_caption_json,
            output_file=output_file,
            scene_score_query=query_engine,
            batch_size=args.batch_size,
        )
        all_stats.append(stats)

    elapsed = time.time() - start_time
    summary = {
        "model_name": args.model_name,
        "llm_client_config": args.llm_client_config,
        "scene_caption_dir": args.scene_caption_dir,
        "video_caption_dir": args.video_caption_dir,
        "output_dir": args.output_dir,
        "datasets": dataset_list,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "stats": all_stats,
    }

    summary_file = os.path.join(args.output_dir, "run_summary.json")
    _save_json(summary_file, summary)
    print(f"Done. summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query scene scores using DeepSeek32 and save to data/scores/scene_score/deepseek32"
    )
    parser.add_argument("--model-name", type=str, default="deepseek32")
    parser.add_argument("--scene-score-config", type=str,
                        default="/root/tfnet/configs/model/scene_score_gpt5.yaml")
    parser.add_argument("--llm-client-config", type=str,
                        default="/root/tfnet/configs/llm_client/deepseek3_2.yaml")
    parser.add_argument("--scene-caption-dir", type=str,
                        default="/root/tfnet/data/captions/scene_caption/gpt5")
    parser.add_argument("--video-caption-dir", type=str,
                        default="/root/tfnet/data/captions/video_caption/gpt5")
    parser.add_argument("--output-dir", type=str,
                        default="/root/tfnet/data/scores/scene_score/deepseek32")
    parser.add_argument("--datasets", type=str, default="summe,tvsum")
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()
    asyncio.run(_main_async(args))
