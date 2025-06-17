"""
步骤2：相似度计算
使用相似度计算器计算视频帧与查询文本之间的相似度分数

输入：视频帧 + 查询文本
输出：帧相似度分数序列（0-1范围）

关键参数：
- 帧跳数(interval) M，表示对于视频，每隔M帧采样一次
- 批处理大小
- 查询文本
"""

import argparse
import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch

# 添加项目根目录到Python路径以支持绝对导入
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from pytorch_lightning.loggers import TensorBoardLogger
from src.util.similarity_calculator import SimilarityCalculator


# TensorBoard相关导入


class FrameSimilarityCalculator:
    """帧相似度计算器"""

    def __init__(
        self,
        dataset_paths: List[str],
        frame_interval: int = 15,
        output_dir: str = "./results",
        batch_size: int = 32,
        similarity_query: str = "video summary",
        device: str = None,
        logger: Optional[TensorBoardLogger] = None
    ):
        """
        初始化帧相似度计算器

        Args:
            dataset_paths: 数据集路径列表，每个路径包含frames文件夹
            frame_interval: 帧跳数M，表示每隔M帧采样一次
            output_dir: 输出结果的目录
            batch_size: 批处理大小
            similarity_query: 相似度计算时使用的查询文本
            device: 计算设备
            logger: TensorBoard logger对象
        """
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.similarity_query = similarity_query

        # TensorBoard logger配置
        self.logger = logger
        self.log_dir = None

        if self.logger:
            self.log_dir = Path(self.logger.log_dir)
            self.output_dir = self.log_dir / "results"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化相似度计算器
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.similarity_calculator = SimilarityCalculator(device=device)

    def log_hyperparameters(self):
        """记录超参数到TensorBoard"""
        if self.logger:
            hparams = {
                'frame_interval': self.frame_interval,
                'batch_size': self.batch_size,
                'similarity_query': self.similarity_query,
                'dataset_count': len(self.dataset_paths)
            }
            self.logger.log_hyperparams(hparams)

    def log_similarity_scores(self, video_name: str, similarities: List[float]):
        """记录视频的相似度分数到TensorBoard"""
        if self.logger:
            for i, similarity in enumerate(similarities):
                self.logger.log_metrics({
                    f'frame_similarity/{video_name}': similarity
                }, step=i)

            if similarities:
                stats = {
                    f'similarity_stats/{video_name}/mean': np.mean(similarities),
                    f'similarity_stats/{video_name}/max': np.max(similarities),
                    f'similarity_stats/{video_name}/min': np.min(similarities),
                    f'similarity_stats/{video_name}/std': np.std(similarities)
                }
                self.logger.log_metrics(stats)

    def log_dataset_summary(self, dataset_name: str, video_results: Dict[str, Dict]):
        """记录数据集汇总信息到TensorBoard"""
        if self.logger:
            total_videos = len(video_results)
            successful_videos = sum(1 for result in video_results.values()
                                    if result.get('status') == 'success')

            summary_metrics = {
                f'dataset_summary/{dataset_name}/total_videos': total_videos,
                f'dataset_summary/{dataset_name}/successful_videos': successful_videos
            }

            all_mean_similarities = []
            for result in video_results.values():
                if result.get('status') == 'success':
                    similarities = result.get('similarities', {})
                    if isinstance(similarities, dict) and 'mean' in similarities:
                        all_mean_similarities.append(similarities['mean'])

            if all_mean_similarities:
                summary_metrics[f'dataset_summary/{dataset_name}/overall_mean_similarity'] = np.mean(
                    all_mean_similarities)

            self.logger.log_metrics(summary_metrics)

    def get_video_frames(self, video_frames_dir: Path) -> List[str]:
        """
        获取视频目录下的所有帧文件路径

        Args:
            video_frames_dir: 视频帧文件夹路径

        Returns:
            排序后的帧文件路径列表
        """
        frame_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_files.extend(video_frames_dir.glob(ext))

        # 按帧编号排序（支持frame_xxxxxx.jpg格式）
        def extract_frame_number(file_path):
            filename = file_path.name
            # 匹配frame_xxxxxx.jpg格式
            match = re.search(r'frame_(\d+)', filename)
            if match:
                return int(match.group(1))
            # 如果不匹配，尝试提取任何数字
            match = re.search(r'(\d+)', filename)
            if match:
                return int(match.group(1))
            return 0

        frame_files.sort(key=extract_frame_number)
        return [str(f) for f in frame_files]

    def sample_frames(self, frame_paths: List[str]) -> List[int]:
        """
        按照帧跳数采样帧索引

        Args:
            frame_paths: 所有帧的文件路径列表

        Returns:
            采样的帧索引列表
        """
        total_frames = len(frame_paths)
        sampled_indices = []

        for i in range(0, total_frames, self.frame_interval):
            sampled_indices.append(i)

        return sampled_indices

    def get_video_caption(self, dataset_path: Path, video_name: str) -> str:
        """
        获取视频的Qwen caption
        
        Args:
            dataset_path: 数据集根路径
            video_name: 视频名称
            
        Returns:
            视频的caption文本，如果不存在则返回默认查询文本
        """
        caption_file = dataset_path / "captions" / "Qwen25VL" / f"{video_name}.txt"
        
        if caption_file.exists():
            try:
                with open(caption_file, 'r', encoding='utf-8') as f:
                    caption = f.read().strip()
                    return caption
            except Exception as e:
                print(f"Error reading caption file {caption_file}: {e}")
                return self.similarity_query
        else:
            print(f"Caption file not found: {caption_file}, using default query")
            return self.similarity_query

    def process_video_similarity(self, video_frames_dir: Path, video_name: str, dataset_path: Path) -> Dict[str, Any]:
        """处理单个视频的相似度计算"""
        try:
            # 获取视频帧路径
            frame_paths = self.get_video_frames(video_frames_dir)
            if not frame_paths:
                return {
                    "video_name": video_name,
                    "status": "no_frames",
                    "total_frames": 0,
                    "similarities": {}
                }

            # 采样帧
            sampled_indices = self.sample_frames(frame_paths)
            sampled_frame_paths = [frame_paths[idx] for idx in sampled_indices]

            # 加载图像
            images = []
            valid_indices = []
            for i, frame_path in enumerate(sampled_frame_paths):
                try:
                    img = Image.open(frame_path).convert('RGB')
                    images.append(img)
                    valid_indices.append(sampled_indices[i])
                except Exception as e:
                    print(f"Error loading image {frame_path}: {e}")
                    continue

            if not images:
                return {
                    "video_name": video_name,
                    "status": "no_valid_images",
                    "total_frames": len(frame_paths),
                    "similarities": {}
                }

            # 获取视频的Qwen caption作为查询文本
            video_caption = self.get_video_caption(dataset_path, video_name)
            print(f"Using caption for {video_name}: {video_caption[:100]}...")

            # 计算相似度 - 使用与calc_similarity.py相同的流程
            print(f"Computing similarity scores for {video_name}...")
            print("Extracting text features...")
            text_features = self.similarity_calculator.extract_text_features(
                video_caption)

            print("Extracting visual features...")
            visual_features = self.similarity_calculator.extract_visual_features_batch(
                images, batch_size=self.batch_size
            )

            print("Computing similarities from features...")
            similarities = self.similarity_calculator.compute_similarity_from_features(
                text_features, visual_features
            )

            # 确保相似度在0-1范围内
            similarities = np.clip(similarities, 0.0, 1.0)

            # 记录到TensorBoard
            self.log_similarity_scores(video_name, similarities.tolist())

            return {
                "video_name": video_name,
                "status": "success",
                "total_frames": len(frame_paths),
                "sampled_frames": len(images),
                "sampled_indices": valid_indices,
                "query_text": video_caption,  # 使用视频的caption作为查询文本
                "similarities": {
                    "values": similarities.tolist(),
                    "mean": float(np.mean(similarities)),
                    "max": float(np.max(similarities)),
                    "min": float(np.min(similarities)),
                    "std": float(np.std(similarities))
                }
            }

        except Exception as e:
            return {
                "video_name": video_name,
                "status": "error",
                "error": str(e),
                "similarities": {}
            }

    def process_dataset_similarity(self, dataset_path: Path) -> Dict[str, Any]:
        """处理单个数据集的相似度计算"""
        dataset_name = dataset_path.name
        frames_dir = dataset_path / "frames"

        if not frames_dir.exists():
            return {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path),
                "status": "no_frames_dir",
                "total_videos": 0,
                "videos": {}
            }

        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]

        dataset_results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "total_videos": len(video_dirs),
            "videos": {}
        }

        for video_dir in tqdm(video_dirs, desc=f"Computing similarities for {dataset_name}"):
            video_name = video_dir.name
            video_result = self.process_video_similarity(video_dir, video_name, dataset_path)
            dataset_results["videos"][video_name] = video_result

        self.log_dataset_summary(dataset_name, dataset_results["videos"])
        return dataset_results

    def save_results(self, results: Dict[str, Any], filename: str = "step2_similarity_scores.json"):
        """保存结果到JSON文件"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_path}")

    def close_logger(self):
        """关闭TensorBoard logger"""
        if self.logger:
            self.logger.finalize("success")

    def run(self):
        """执行相似度计算流程"""
        print("Loading BLIP2 model for similarity calculation...")
        try:
            # 确保相似度计算器已初始化
            if not hasattr(self.similarity_calculator, 'model') or self.similarity_calculator.model is None:
                self.similarity_calculator.load_model()
                print("BLIP2 model loaded successfully.")
        except Exception as e:
            print(f"Error loading BLIP2 model: {e}")
            return {
                "step": 2,
                "description": "Compute frame-text similarity scores using BLIP2",
                "status": "model_load_error",
                "error": str(e),
                "datasets": {}
            }

        self.log_hyperparameters()

        all_results = {
            "step": 2,
            "description": "Compute frame-text similarity scores using BLIP2 image-text matching",
            "parameters": {
                "frame_interval": self.frame_interval,
                "batch_size": self.batch_size,
                "query_text": self.similarity_query,
                "model_name": self.similarity_calculator.model_name,
                "model_type": self.similarity_calculator.model_type,
                "device": self.similarity_calculator.device
            },
            "timestamp": datetime.now().isoformat(),
            "datasets": {}
        }

        for dataset_path in self.dataset_paths:
            print(f"Processing dataset: {dataset_path}")
            dataset_result = self.process_dataset_similarity(dataset_path)
            dataset_name = dataset_result["dataset_name"]
            all_results["datasets"][dataset_name] = dataset_result

        self.save_results(all_results)
        self.close_logger()
        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="步骤2：帧相似度计算")
    parser.add_argument("--dataset_paths", nargs="+",
                        default=["/root/autodl-tmp/data/SumMe",
                                 "/root/autodl-tmp/data/TVSum"],
                        help="数据集路径列表")
    parser.add_argument("--frame_interval", type=int, default=15, help="帧跳数")
    parser.add_argument("--output_dir", type=str, default="./out", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--similarity_query", type=str,
                        default="video summary", help="相似度查询文本")
    parser.add_argument("--device", type=str, default=None, help="计算设备")

    args = parser.parse_args()

    calculator = FrameSimilarityCalculator(
        dataset_paths=args.dataset_paths,
        frame_interval=args.frame_interval,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        similarity_query=args.similarity_query,
        device=args.device
    )

    results = calculator.run()
    print(f"步骤2完成！处理了 {len(results['datasets'])} 个数据集")
