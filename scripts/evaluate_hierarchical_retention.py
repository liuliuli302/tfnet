#!/usr/bin/env python3
"""
层级语义聚合信息保留率评估实验

评估帧→场景 和 场景→视频 两个层级的信息保留率。
指标：ROUGE-1 Recall, BERTScore Recall
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "captions")
DEFAULT_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs", "hierarchical_retention")

FRAME_CAPTION_DIR = os.path.join(DATA_ROOT, "frame_caption", "llava")
SCENE_CAPTION_DIR = os.path.join(DATA_ROOT, "scene_caption", "gpt5")
VIDEO_CAPTION_DIR = os.path.join(DATA_ROOT, "video_caption", "gpt5")


class HierarchicalRetentionEvaluator:
    """层级语义保留率评估器"""

    def __init__(self, output_dir: str, bertscore_model: str = "microsoft/deberta-xlarge-mnli"):
        self.output_dir = output_dir
        self.bertscore_model = bertscore_model
        self.rouge_scorer = None
        os.makedirs(output_dir, exist_ok=True)

    # ---- 数据加载 ----

    def _load_json(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_data(self, dataset: str) -> dict:
        """加载三个层级的 caption JSON"""
        frame_path = os.path.join(
            FRAME_CAPTION_DIR, f"{dataset}_frame_captions.json")
        scene_path = os.path.join(
            SCENE_CAPTION_DIR, f"{dataset}_scene_captions.json")
        video_path = os.path.join(
            VIDEO_CAPTION_DIR, f"{dataset}_video_captions.json")

        for p, name in [(frame_path, "frame"), (scene_path, "scene"), (video_path, "video")]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{name} caption not found: {p}")

        return {
            "frame": self._load_json(frame_path),
            "scene": self._load_json(scene_path),
            "video": self._load_json(video_path),
        }

    # ---- 指标计算 ----

    def _init_rouge(self):
        if self.rouge_scorer is None:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ["rouge1"], use_stemmer=True)

    def compute_rouge1_recall(self, reference: str, candidate: str) -> float:
        """计算 ROUGE-1 Recall"""
        self._init_rouge()
        if not reference.strip() or not candidate.strip():
            return 0.0
        scores = self.rouge_scorer.score(reference, candidate)
        return scores["rouge1"].recall

    def compute_bertscore_batch(self, references: list, candidates: list) -> list:
        """批量计算 BERTScore Recall"""
        if not references or not candidates:
            return []
        # 过滤空文本对 & 截断过长文本（防止 DeBERTa tokenizer overflow）
        MAX_CHARS = 2000
        valid_pairs = []
        for r, c in zip(references, candidates):
            r = r.strip()[:MAX_CHARS]
            c = c.strip()[:MAX_CHARS]
            if r and c:
                valid_pairs.append((r, c))
        if not valid_pairs:
            return [0.0] * len(references)

        from bert_score import score
        refs, cands = zip(*valid_pairs)
        try:
            P, R, F1 = score(
                list(cands), list(refs),
                lang="en",
                model_type=self.bertscore_model,
                verbose=False,
                device="cuda" if self._has_gpu() else "cpu",
            )
            return R.tolist()
        except OverflowError:
            # DeBERTa tokenizer 溢出时的回退：使用 roberta-large
            print(
                "  [WARN] BERTScore with deberta failed (OverflowError), falling back to roberta-large")
            P, R, F1 = score(
                list(cands), list(refs),
                lang="en",
                model_type="roberta-large",
                verbose=False,
                device="cuda" if self._has_gpu() else "cpu",
            )
            return R.tolist()

    @staticmethod
    def _has_gpu() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---- 构建 pick→caption 映射 ----

    @staticmethod
    def _build_pick_to_caption(frame_data_entry: dict) -> dict:
        """构建 {pick_index: caption_string} 映射"""
        picks = frame_data_entry.get("picks", [])
        captions = frame_data_entry.get("captions", [])
        return {p: c for p, c in zip(picks, captions)}

    # ---- 帧→场景 评估 ----

    def evaluate_frame_to_scene(self, dataset: str) -> dict:
        """帧→场景 信息保留率"""
        data = self.load_data(dataset)
        frame_data = data["frame"]
        scene_data = data["scene"]

        per_scene_records = []
        per_video = defaultdict(list)

        # 收集所有 (reference, candidate) 对用于批量 BERTScore
        all_refs = []
        all_cands = []
        record_meta = []  # 每条记录的元数据

        print(f"\n{'='*60}")
        print(f"[Frame→Scene] {dataset.upper()}: 收集数据...")
        print(f"{'='*60}")

        for video_name in tqdm(sorted(scene_data.keys()), desc="收集帧→场景数据"):
            if video_name not in frame_data:
                print(f"  [WARN] {video_name}: 缺少帧级描述，跳过", file=sys.stderr)
                continue

            pick2cap = self._build_pick_to_caption(frame_data[video_name])
            scenes = scene_data[video_name]

            for scene_key in sorted(scenes.keys(), key=lambda x: int(x)):
                scene = scenes[scene_key]
                scene_picks = scene.get("picks", [])
                scene_caption = scene.get("scene_caption", "")

                # 按 picks 顺序拼接帧描述
                frame_captions = []
                for p in scene_picks:
                    cap = pick2cap.get(p, "")
                    if cap:
                        frame_captions.append(cap)
                    else:
                        print(f"  [WARN] {video_name} scene {scene_key}: pick {p} 缺失帧描述",
                              file=sys.stderr)

                reference_text = " ".join(frame_captions)

                if not reference_text.strip() or not scene_caption.strip():
                    print(f"  [WARN] {video_name} scene {scene_key}: 空文本，跳过",
                          file=sys.stderr)
                    continue

                all_refs.append(reference_text)
                all_cands.append(scene_caption)
                record_meta.append({
                    "video_name": video_name,
                    "scene_idx": scene.get("scene_idx", int(scene_key)),
                    "num_frames": len(frame_captions),
                    "reference_length": len(reference_text),
                    "candidate_length": len(scene_caption),
                })

        print(f"  有效场景数: {len(all_refs)}")

        # 批量计算 BERTScore
        print(f"\n  计算 BERTScore (model={self.bertscore_model})...")
        bertscore_recalls = self.compute_bertscore_batch(all_refs, all_cands)

        # 逐条计算 ROUGE
        print(f"  计算 ROUGE-1...")
        rouge_recalls = []
        for ref, cand in tqdm(zip(all_refs, all_cands), total=len(all_refs), desc="  ROUGE-1"):
            rouge_recalls.append(self.compute_rouge1_recall(ref, cand))

        # 组装 per_scene 记录
        for i, meta in enumerate(record_meta):
            meta["rouge1_recall"] = round(rouge_recalls[i], 4)
            meta["bertscore_recall"] = round(bertscore_recalls[i], 4)
            per_scene_records.append(meta)
            per_video[meta["video_name"]].append(meta)

        # 按视频聚合
        per_video_agg = []
        for vname in sorted(per_video.keys()):
            records = per_video[vname]
            per_video_agg.append({
                "video_name": vname,
                "num_scenes": len(records),
                "rouge1_recall_mean": round(np.mean([r["rouge1_recall"] for r in records]), 4),
                "bertscore_recall_mean": round(np.mean([r["bertscore_recall"] for r in records]), 4),
            })

        # 计算全局统计
        rouge_arr = np.array(rouge_recalls)
        bscore_arr = np.array(bertscore_recalls)
        comp_ratios = np.array([r["candidate_length"] / max(r["reference_length"], 1)
                                for r in per_scene_records])

        def stats(arr):
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        result = {
            "dataset": dataset,
            "level": "frame_to_scene",
            "bertscore_model": self.bertscore_model,
            "metrics": {
                "rouge1_recall": stats(rouge_arr),
                "bertscore_recall": stats(bscore_arr),
                "compression_ratio": {
                    **stats(comp_ratios),
                    "description": "candidate_length / reference_length (字符数)"
                },
            },
            "per_scene": per_scene_records,
            "per_video": per_video_agg,
        }
        return result

    # ---- 场景→视频 评估 ----

    def evaluate_scene_to_video(self, dataset: str) -> dict:
        """场景→视频 信息保留率"""
        data = self.load_data(dataset)
        scene_data = data["scene"]
        video_data = data["video"]

        all_refs = []
        all_cands = []
        record_meta = []

        print(f"\n{'='*60}")
        print(f"[Scene→Video] {dataset.upper()}: 收集数据...")
        print(f"{'='*60}")

        for video_name in tqdm(sorted(scene_data.keys()), desc="收集场景→视频数据"):
            if video_name not in video_data:
                print(f"  [WARN] {video_name}: 缺少视频级摘要，跳过", file=sys.stderr)
                continue

            scenes = scene_data[video_name]
            video_caption = video_data[video_name]

            # 按 scene_idx 升序拼接所有场景摘要
            sorted_scenes = sorted(
                scenes.values(), key=lambda s: s.get("scene_idx", 0))
            scene_captions = [s["scene_caption"] for s in sorted_scenes
                              if s.get("scene_caption", "").strip()]

            reference_text = " ".join(scene_captions)

            if not reference_text.strip() or not video_caption.strip():
                print(f"  [WARN] {video_name}: 空文本，跳过", file=sys.stderr)
                continue

            all_refs.append(reference_text)
            all_cands.append(video_caption)
            record_meta.append({
                "video_name": video_name,
                "num_scenes": len(scene_captions),
                "reference_length": len(reference_text),
                "candidate_length": len(video_caption),
            })

        print(f"  有效视频数: {len(all_refs)}")

        # 批量计算 BERTScore
        print(f"\n  计算 BERTScore (model={self.bertscore_model})...")
        bertscore_recalls = self.compute_bertscore_batch(all_refs, all_cands)

        # 逐条计算 ROUGE
        print(f"  计算 ROUGE-1...")
        rouge_recalls = []
        for ref, cand in tqdm(zip(all_refs, all_cands), total=len(all_refs), desc="  ROUGE-1"):
            rouge_recalls.append(self.compute_rouge1_recall(ref, cand))

        # 组装 per_video 记录
        per_video_records = []
        for i, meta in enumerate(record_meta):
            meta["rouge1_recall"] = round(rouge_recalls[i], 4)
            meta["bertscore_recall"] = round(bertscore_recalls[i], 4)
            per_video_records.append(meta)

        # 计算全局统计
        rouge_arr = np.array(rouge_recalls)
        bscore_arr = np.array(bertscore_recalls)
        comp_ratios = np.array([r["candidate_length"] / max(r["reference_length"], 1)
                                for r in per_video_records])

        def stats(arr):
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "median": round(float(np.median(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        result = {
            "dataset": dataset,
            "level": "scene_to_video",
            "bertscore_model": self.bertscore_model,
            "metrics": {
                "rouge1_recall": stats(rouge_arr),
                "bertscore_recall": stats(bscore_arr),
                "compression_ratio": {
                    **stats(comp_ratios),
                    "description": "candidate_length / reference_length (字符数)"
                },
            },
            "per_video": per_video_records,
        }
        return result

    # ---- 保存结果 ----

    def save_result(self, result: dict, filename: str):
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 已保存: {path}")

    def save_summary_csv(self, all_results: list):
        """生成 summary_table.csv"""
        rows = []
        for r in all_results:
            level_label = "Frame→Scene" if r["level"] == "frame_to_scene" else "Scene→Video"
            dataset_label = "SumMe" if r["dataset"] == "summe" else "TVSum"
            m = r["metrics"]
            rouge_str = f"{m['rouge1_recall']['mean']:.4f}±{m['rouge1_recall']['std']:.4f}"
            bscore_str = f"{m['bertscore_recall']['mean']:.4f}±{m['bertscore_recall']['std']:.4f}"
            comp_str = f"{m['compression_ratio']['mean']:.4f}"
            rows.append([level_label, dataset_label,
                        rouge_str, bscore_str, comp_str])

        path = os.path.join(self.output_dir, "summary_table.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "Level,Dataset,ROUGE-1 R (μ±σ),BERTScore R (μ±σ),Compression Ratio\n")
            for row in rows:
                f.write(",".join(row) + "\n")
        print(f"  ✓ 已保存: {path}")

        # 同时打印到终端
        print(f"\n{'='*80}")
        print("Summary Table")
        print(f"{'='*80}")
        header = f"{'Level':<16} {'Dataset':<8} {'ROUGE-1 R (μ±σ)':<22} {'BERTScore R (μ±σ)':<22} {'Compression':<14}"
        print(header)
        print("-" * 80)
        for row in rows:
            print(
                f"{row[0]:<16} {row[1]:<8} {row[2]:<22} {row[3]:<22} {row[4]:<14}")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="层级语义聚合信息保留率评估实验"
    )
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["tvsum", "summe", "all"],
                        help="数据集 (默认: all)")
    parser.add_argument("--level", type=str, default="all",
                        choices=["frame_to_scene", "scene_to_video", "all"],
                        help="评估层级 (默认: all)")
    parser.add_argument("--bertscore-model", type=str,
                        default="microsoft/deberta-xlarge-mnli",
                        help="BERTScore 模型 (默认: microsoft/deberta-xlarge-mnli)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (默认: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()

    datasets = ["tvsum", "summe"] if args.dataset == "all" else [args.dataset]
    levels = ["frame_to_scene",
              "scene_to_video"] if args.level == "all" else [args.level]

    print(f"数据集: {datasets}")
    print(f"评估层级: {levels}")
    print(f"BERTScore 模型: {args.bertscore_model}")
    print(f"输出目录: {args.output_dir}")

    evaluator = HierarchicalRetentionEvaluator(
        output_dir=args.output_dir,
        bertscore_model=args.bertscore_model,
    )

    all_results = []

    for dataset in datasets:
        for level in levels:
            if level == "frame_to_scene":
                result = evaluator.evaluate_frame_to_scene(dataset)
                evaluator.save_result(result, f"{dataset}_frame_to_scene.json")
                all_results.append(result)
            elif level == "scene_to_video":
                result = evaluator.evaluate_scene_to_video(dataset)
                evaluator.save_result(result, f"{dataset}_scene_to_video.json")
                all_results.append(result)

    if all_results:
        evaluator.save_summary_csv(all_results)

    print(f"\n✓ 实验完成! 结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
