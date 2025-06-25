"""步骤2：相似度计算"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.util.similarity_calculator import SimilarityCalculator


class FrameSimilarityCalculator:
    def __init__(self, dataset_paths: List[str], frame_interval: int = 15, 
                 output_dir: str = "./results", batch_size: int = 32,
                 device: str = None):
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_calculator = SimilarityCalculator(device=device)

    def get_video_frames(self, video_frames_dir: Path) -> List[str]:
        frame_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_files.extend(video_frames_dir.glob(ext))

        def extract_frame_number(file_path):
            filename = file_path.name
            match = re.search(r'frame_(\d+)', filename)
            if match:
                return int(match.group(1))
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0

        frame_files.sort(key=extract_frame_number)
        return [str(f) for f in frame_files]

    def sample_frames(self, frame_paths: List[str]) -> List[int]:
        return list(range(0, len(frame_paths), self.frame_interval))

    def get_video_caption(self, dataset_path: Path, video_name: str) -> str:
        caption_file = dataset_path / "captions" / "Qwen25VL" / f"{video_name}.txt"
        return caption_file.read_text(encoding='utf-8').strip()

    def process_video_similarity(self, video_frames_dir: Path, video_name: str, dataset_path: Path) -> Dict[str, Any]:
        frame_paths = self.get_video_frames(video_frames_dir)
        sampled_indices = self.sample_frames(frame_paths)
        sampled_frame_paths = [frame_paths[idx] for idx in sampled_indices]
        
        images = [Image.open(frame_path).convert('RGB') for frame_path in sampled_frame_paths]
        video_caption = self.get_video_caption(dataset_path, video_name)
        
        text_features = self.similarity_calculator.extract_text_features(video_caption)
        visual_features = self.similarity_calculator.extract_visual_features_batch(images, batch_size=self.batch_size)
        similarities = self.similarity_calculator.compute_similarity_from_features(text_features, visual_features)
        similarities = np.clip(similarities, 0.0, 1.0)

        return {
            "video_name": video_name,
            "total_frames": len(frame_paths),
            "sampled_frames": len(images),
            "sampled_indices": sampled_indices,
            "query_text": video_caption,
            "similarities": {
                "values": similarities.tolist(),
                "mean": float(np.mean(similarities)),
                "max": float(np.max(similarities)),
                "min": float(np.min(similarities)),
                "std": float(np.std(similarities))
            }
        }

    def process_dataset_similarity(self, dataset_path: Path) -> Dict[str, Any]:
        frames_dir = dataset_path / "frames"
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        
        videos = {}
        for video_dir in tqdm(video_dirs, desc=f"Computing similarities for {dataset_path.name}"):
            videos[video_dir.name] = self.process_video_similarity(video_dir, video_dir.name, dataset_path)
        
        return {
            "dataset_name": dataset_path.name,
            "dataset_path": str(dataset_path),
            "total_videos": len(video_dirs),
            "videos": videos
        }

    def save_results(self, results: Dict[str, Any], filename: str = "step2_similarity_scores.json"):
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Results saved to: {output_path}")

    def run(self):
        self.similarity_calculator.load_model()
        
        results = {
            "step": 2,
            "description": "Compute frame-text similarity scores using BLIP2 image-text matching",
            "parameters": {
                "frame_interval": self.frame_interval,
                "batch_size": self.batch_size,
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
            results["datasets"][dataset_result["dataset_name"]] = dataset_result

        self.save_results(results)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="步骤2：帧相似度计算")
    parser.add_argument("--dataset_paths", nargs="+", 
                        default=["/root/autodl-tmp/data/SumMe", "/root/autodl-tmp/data/TVSum"])
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    calculator = FrameSimilarityCalculator(**vars(args))
    results = calculator.run()
    print(f"步骤2完成！处理了 {len(results['datasets'])} 个数据集")
