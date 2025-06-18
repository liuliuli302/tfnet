"""步骤2改进版：分段相似度计算"""

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


class RefinedFrameSimilarityCalculator:
    def __init__(self, dataset_paths: List[str], frame_interval: int = 15, 
                 output_dir: str = "./results", batch_size: int = 32,
                 device: str = None, segment_counts: List[int] = None):
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.frame_interval = frame_interval
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_counts = segment_counts or [2, 4, 8]  # 简化为3个分段数
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_calculator = SimilarityCalculator(device=device)

    def get_video_frames(self, video_frames_dir: Path) -> List[str]:
        """获取视频帧路径列表"""
        frame_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            frame_files.extend(video_frames_dir.glob(ext))

        def extract_frame_number(file_path):
            filename = file_path.name
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0

        frame_files.sort(key=extract_frame_number)
        return [str(f) for f in frame_files]

    def get_video_caption(self, dataset_path: Path, video_name: str) -> str:
        """获取视频字幕"""
        caption_file = dataset_path / "captions" / "Qwen25VL" / f"{video_name}.txt"
        if caption_file.exists():
            return caption_file.read_text(encoding='utf-8').strip()
        return "video summary"  # 默认查询文本

    def segment_data(self, items: List, num_segments: int) -> List[List]:
        """将列表分割成指定数量的段落"""
        if len(items) <= num_segments:
            return [[item] for item in items]
        
        segment_size = len(items) // num_segments
        segments = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = len(items) if i == num_segments - 1 else (i + 1) * segment_size
            segments.append(items[start_idx:end_idx])
        
        return segments

    def process_video_similarity(self, video_frames_dir: Path, video_name: str, dataset_path: Path) -> Dict[str, Any]:
        """处理单个视频的相似度计算"""
        # 获取帧和采样
        frame_paths = self.get_video_frames(video_frames_dir)
        sampled_indices = list(range(0, len(frame_paths), self.frame_interval))
        sampled_frame_paths = [frame_paths[idx] for idx in sampled_indices]
        
        video_caption = self.get_video_caption(dataset_path, video_name)
        
        # 为每个分段数计算相似度
        segment_results = {}
        
        # 添加更详细的进度条
        pbar_segments = tqdm(self.segment_counts, desc=f"Processing {video_name}", leave=False)
        
        for num_segments in pbar_segments:
            pbar_segments.set_postfix({"segments": num_segments})
            
            # 分割文本和帧
            text_words = video_caption.split()
            text_segments = self.segment_data(text_words, num_segments)
            text_segments = [' '.join(words) for words in text_segments]
            
            frame_segments = self.segment_data(sampled_indices, num_segments)
            
            # 计算每个段落的相似度
            all_similarities = []
            segment_details = []
            
            for seg_idx, (frame_seg, text_seg) in enumerate(zip(frame_segments, text_segments)):
                # 加载图像
                segment_images = []
                for idx in frame_seg:
                    try:
                        img = Image.open(frame_paths[idx]).convert('RGB')
                        segment_images.append(img)
                    except Exception as e:
                        print(f"Error loading frame {idx}: {e}")
                        continue
                
                if not segment_images:
                    continue
                
                # 计算相似度
                try:
                    similarities = self.similarity_calculator.compute_similarity(
                        frames=segment_images,
                        text=text_seg,
                        batch_size=self.batch_size
                    )
                    similarities = np.clip(similarities, 0.0, 1.0)
                    all_similarities.extend(similarities.tolist())
                    
                    segment_details.append({
                        "segment_index": seg_idx,
                        "frame_indices": frame_seg,
                        "text_segment": text_seg,
                        "frame_count": len(segment_images),
                        "mean_similarity": float(np.mean(similarities))
                    })
                    
                except Exception as e:
                    print(f"Error computing similarity for segment {seg_idx}: {e}")
                    continue
            
            segment_results[f"segments_{num_segments}"] = {
                "num_segments": num_segments,
                "similarities": all_similarities,
                "overall_mean": float(np.mean(all_similarities)) if all_similarities else 0.0,
                "segment_details": segment_details
            }
        
        pbar_segments.close()
        
        return {
            "video_name": video_name,
            "total_frames": len(frame_paths),
            "sampled_frames": len(sampled_indices),
            "sampled_indices": sampled_indices,
            "query_text": video_caption,
            "segment_results": segment_results
        }

    def process_dataset_similarity(self, dataset_path: Path) -> Dict[str, Any]:
        """处理单个数据集的相似度计算"""
        frames_dir = dataset_path / "frames"
        if not frames_dir.exists():
            print(f"Warning: frames directory not found: {frames_dir}")
            return {
                "dataset_name": dataset_path.name,
                "dataset_path": str(dataset_path),
                "total_videos": 0,
                "videos": {}
            }
        
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        videos = {}
        
        # 主进度条显示视频处理进度
        for video_dir in tqdm(video_dirs, desc=f"Processing {dataset_path.name}"):
            try:
                videos[video_dir.name] = self.process_video_similarity(
                    video_dir, video_dir.name, dataset_path)
            except Exception as e:
                print(f"Error processing video {video_dir.name}: {e}")
                continue
        
        return {
            "dataset_name": dataset_path.name,
            "dataset_path": str(dataset_path),
            "total_videos": len(video_dirs),
            "videos": videos
        }

    def save_results(self, results: Dict[str, Any], filename: str = "step2_refined_similarity_scores.json"):
        """保存结果"""
        output_path = self.output_dir / filename
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Results saved to: {output_path}")

    def run(self):
        """运行计算流程"""
        print("Loading BLIP2 model...")
        self.similarity_calculator.load_model()
        print("Model loaded successfully.")
        
        results = {
            "step": 2,
            "description": "Compute refined frame-text similarity scores using segmented approach",
            "parameters": {
                "frame_interval": self.frame_interval,
                "batch_size": self.batch_size,
                "segment_counts": self.segment_counts,
                "model_name": getattr(self.similarity_calculator, 'model_name', 'blip2_image_text_matching'),
                "device": self.similarity_calculator.device
            },
            "timestamp": datetime.now().isoformat(),
            "datasets": {}
        }

        for dataset_path in self.dataset_paths:
            print(f"\nProcessing dataset: {dataset_path}")
            dataset_result = self.process_dataset_similarity(dataset_path)
            results["datasets"][dataset_result["dataset_name"]] = dataset_result

        self.save_results(results)
        print(f"步骤2改进版完成！处理了 {len(results['datasets'])} 个数据集")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="步骤2改进版：分段帧相似度计算")
    parser.add_argument("--dataset_paths", nargs="+", 
                        default=["/root/autodl-tmp/data/SumMe", "/root/autodl-tmp/data/TVSum"])
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--output_dir", type=str, default="./out")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--segment_counts", nargs="+", type=int, default=[2, 4, 8])

    args = parser.parse_args()
    calculator = RefinedFrameSimilarityCalculator(**vars(args))
    results = calculator.run()