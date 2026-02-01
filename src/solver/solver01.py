import argparse
import json
import os
from itertools import product
from typing import Dict, Any

from tqdm import tqdm

from src.config.config import BasicConfig
from src.models.frame_seclection import nfs_from_lvnet
from utils.llm_client import LLMQueryConfig, LLMQueryRunner, deepseek, moonshot, save_result
import yaml


class Exam01Config(BasicConfig):
    """Configuration holder for exam01 solver."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='/root/tfnet/configs/exam01_config.yaml',)
    return parser.parse_args()


class Exam01Solver:
    """
    Experiment runner encapsulating NFS sampling (optional) and LLM querying.
    Follows a class-based pattern for clarity and configurability.
    """

    def __init__(self, config: Exam01Config):
        if not isinstance(config, Exam01Config):
            raise TypeError("config must be Exam01Config")
        self.config = config
        self.model_mapping = {
            "moonshot-v1-128k": moonshot,
            "deepseek-r1": deepseek,
        }

    def run_nfs(self):
        """Optional NFS preprocessing; call if needed."""
        for dataset_name in ("summe", "tvsum"):
            ds_cfg = self.config.datasets[dataset_name]
            nfs_from_lvnet(
                frames_dir=ds_cfg["frames_dir"],
                out_dir=self.config.nfs["out_dir"],
                batch_size=self.config.nfs["batch_size"],
                dataset_name=dataset_name,
                frame_interval=self.config.nfs["frame_interval"],
                divlam=self.config.nfs["divlam"],
            )

    def _load_frame_captions(self, dataset_name: str):
        path = os.path.join(
            self.config.datasets[dataset_name]["frame_captions_dir"],
            "blip",
            f"{dataset_name}_captions.json",
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_video_captions(self, dataset_name: str, summary_source: str):
        path = os.path.join(
            self.config.datasets[dataset_name]["video_captions_dir"],
            summary_source,
            f"{dataset_name}_summary_{summary_source}.json",
        )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_runner(self, query_type: str, prompt: str, max_concurrent: int) -> LLMQueryRunner:
        cfg = LLMQueryConfig(
            query_type=query_type,
            prompt=prompt,
            max_concurrent=max_concurrent,
        )
        model = self.model_mapping[query_type.split('_')[0] if query_type.split(
            '_')[0] in self.model_mapping else list(self.model_mapping.keys())[0]]
        # The model is selected later per combination; here we just return runner factory style
        # but we keep method for symmetry; actual runner uses explicit model in loop
        return LLMQueryRunner(model, cfg)

    def run_llm_queries(self):
        datasets = ["summe", "tvsum"]
        summary_sources = ["moonshot", "deepseek"]

        llm_cfg = self.config.llmquery
        models = llm_cfg["models"]["text_modal"]
        query_types = llm_cfg["query_type"]
        prompts = llm_cfg["prompts"]
        max_concurrent = llm_cfg.get("max_concurrent", 20)

        output_base_dir = llm_cfg.get(
            "output_dir", "/root/tfnet/out/exam01/llm_query")
        os.makedirs(output_base_dir, exist_ok=True)

        print("开始LLM查询实验...")
        print(
            f"总共需要处理: {len(models)} 模型 × {len(query_types)} 查询类型 × {len(datasets)} 数据集 × {len(summary_sources)} 摘要来源")

        combinations = list(
            product(models, query_types, datasets, summary_sources))

        for model_name, query_type, dataset_name, summary_source in tqdm(combinations, leave=True, desc="Processing combinations"):
            needs_summary = "_ws_" in query_type
            if not needs_summary and summary_source != "moonshot":
                continue

            summary_suffix = f"_{summary_source}" if needs_summary else ""
            output_filename = f"{model_name}_{query_type}_{dataset_name}{summary_suffix}.json"
            output_path = os.path.join(output_base_dir, output_filename)

            if os.path.exists(output_path):
                print(f"文件已存在，跳过: {output_filename}")
                continue

            print(
                f"\n处理组合: {model_name} - {query_type} - {dataset_name} - {summary_source}")

            try:
                model = self.model_mapping[model_name]
                prompt = prompts[query_type]

                frame_captions = self._load_frame_captions(dataset_name)
                video_captions = {}
                if needs_summary:
                    video_captions = self._load_video_captions(
                        dataset_name, summary_source)

                runner_cfg = LLMQueryConfig(
                    query_type=query_type,
                    prompt=prompt,
                    max_concurrent=max_concurrent,
                )
                runner = LLMQueryRunner(model, runner_cfg)
                results = runner.process_dataset_text(
                    frame_captions, video_captions)

                save_result(results, output_path)

            except Exception as e:
                print(
                    f"处理组合 {model_name}-{query_type}-{dataset_name}-{summary_source} 时出错: {str(e)}")
                continue

        print("\nLLM查询实验完成！")

    def run(self):
        # self.run_nfs()  # enable if NFS preprocessing is desired
        self.run_llm_queries()


def demo_test_run():
    """Lightweight demo for quick verification (single combination)."""
    cfg_dict = load_yaml('/root/tfnet/configs/exam01_config.yaml')
    cfg = Exam01Config(**cfg_dict)
    llm_cfg = cfg.llmquery
    model = moonshot
    query_type = llm_cfg["query_type"][0]
    prompt = llm_cfg["prompts"][query_type]
    dataset_name = "summe"

    frame_caption_path = os.path.join(
        cfg.datasets[dataset_name]["frame_captions_dir"],
        "blip",
        f"{dataset_name}_captions.json",
    )
    with open(frame_caption_path, "r", encoding="utf-8") as f:
        frame_captions = json.load(f)

    video_captions = {}
    if "_ws_" in query_type:
        video_caption_path = os.path.join(
            cfg.datasets[dataset_name]["video_captions_dir"],
            "moonshot",
            f"{dataset_name}_summary_moonshot.json",
        )
        with open(video_caption_path, "r", encoding="utf-8") as f:
            video_captions = json.load(f)

    runner = LLMQueryRunner(model, LLMQueryConfig(
        query_type=query_type, prompt=prompt, max_concurrent=5))
    results = runner.process_dataset_text(frame_captions, video_captions)
    print("Demo run completed. Combinations processed: 1")
    print(results[:1] if isinstance(results, list) else results)


if __name__ == "__main__":
    args = arg_parser()
    cfg_dict = load_yaml(args.config)
    solver = Exam01Solver(Exam01Config(**cfg_dict))
    solver.run()
    # demo_test_run()  # Uncomment for a quick single-combo smoke test
