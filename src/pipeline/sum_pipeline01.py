"""
视频摘要流程 - 第一步实现
免训练的视频摘要流程，通过多模态大模型查询帧重要性
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径以支持绝对导入
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.handler.llm_client import Qwen25VLClient
except ImportError:
    # 如果绝对导入失败，尝试相对导入
    from ..handler.llm_client import Qwen25VLClient

# TensorBoard相关导入
from pytorch_lightning.loggers import TensorBoardLogger


class VideoSummaryPipeline:
    """视频摘要流程类 - 第一步：查询多模态大模型获取帧重要性分数"""
    
    def __init__(
        self,
        dataset_paths: List[str],
        frame_context_window: int = 3,
        frame_interval: int = 15,
        output_dir: str = "./results",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        importance_prompt: str = None,
        logger: Optional[TensorBoardLogger] = None
    ):
        """
        初始化视频摘要流程
        
        Args:
            dataset_paths: 数据集路径列表，每个路径包含frames文件夹
            frame_context_window: 帧上下文窗口长度N
            frame_interval: 帧跳数M，表示每隔M帧采样一次
            output_dir: 输出结果的目录
            model_name: 多模态模型名称
            device_map: 设备映射
            importance_prompt: 查询帧重要性的提示词
            logger: TensorBoard logger对象
        """
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.frame_context_window = frame_context_window
        self.frame_interval = frame_interval
        
        # TensorBoard logger配置
        self.tb_writer = logger
        self.log_dir = None
        
        if self.tb_writer:
            self.log_dir = Path(self.tb_writer.log_dir)
            self.output_dir = self.log_dir / "results"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化多模态大模型客户端
        self.llm_client = Qwen25VLClient(
            model_name=model_name,
            device_map=device_map
        )
        
        self.importance_prompt = importance_prompt
    
    def log_hyperparameters(self):
        """记录超参数到TensorBoard"""
        if self.tb_writer:
            hparams = {
                'frame_context_window': self.frame_context_window,
                'frame_interval': self.frame_interval,
                'dataset_count': len(self.dataset_paths)
            }
            self.tb_writer.log_hyperparams(hparams)
    
    def log_video_scores(self, video_name: str, scores: List[float], step_offset: int = 0):
        """记录视频的重要性分数到TensorBoard"""
        if self.tb_writer:
            metrics = {}
            for i, score in enumerate(scores):
                metrics[f'frame_importance/{video_name}'] = score
                self.tb_writer.log_metrics(metrics, step=step_offset + i)
            
            if scores:
                stats = {
                    f'video_stats/{video_name}/mean_score': np.mean(scores),
                    f'video_stats/{video_name}/max_score': np.max(scores),
                    f'video_stats/{video_name}/min_score': np.min(scores),
                    f'video_stats/{video_name}/std_score': np.std(scores)
                }
                self.tb_writer.log_metrics(stats)
    
    def log_dataset_summary(self, dataset_name: str, video_results: Dict[str, Dict]):
        """记录数据集汇总信息到TensorBoard"""
        if self.tb_writer:
            total_videos = len(video_results)
            total_frames = sum(result.get('total_frames', 0) for result in video_results.values())
            total_sampled = sum(result.get('sampled_frames', 0) for result in video_results.values())
            
            summary_metrics = {
                f'dataset_summary/{dataset_name}/total_videos': total_videos,
                f'dataset_summary/{dataset_name}/total_frames': total_frames,
                f'dataset_summary/{dataset_name}/total_sampled_frames': total_sampled
            }
            
            all_mean_scores = []
            for result in video_results.values():
                scores = result.get('scores', [])
                if scores:
                    all_mean_scores.append(np.mean(scores))
            
            if all_mean_scores:
                summary_metrics[f'dataset_summary/{dataset_name}/overall_mean_score'] = np.mean(all_mean_scores)
            
            self.tb_writer.log_metrics(summary_metrics)
    
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
            import re
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
    
    def get_context_frames(self, current_idx: int, sampled_indices: List[int]) -> List[int]:
        """
        获取当前帧的上下文帧索引
        
        Args:
            current_idx: 当前帧在采样序列中的位置
            sampled_indices: 采样的帧索引列表
            
        Returns:
            上下文帧索引列表（包含当前帧）
        """
        context_frames = []
        
        # 向前查找上下文帧
        start_idx = max(0, current_idx - self.frame_context_window + 1)
        end_idx = min(len(sampled_indices), current_idx + 1)
        
        for i in range(start_idx, end_idx):
            context_frames.append(sampled_indices[i])
            
        return context_frames
    
    def query_frame_importance(self, frame_paths: List[str]) -> float:
        """查询帧的重要性分数"""
        conversation_content = []
        
        for frame_path in frame_paths:
            conversation_content.append({
                "type": "image",
                "path": frame_path
            })
        
        conversation_content.append({
            "type": "text", 
            "text": self.importance_prompt
        })
        
        conversation = [{
            "role": "user",
            "content": conversation_content
        }]
        
        response = self.llm_client.generate_response(
            conversation=conversation,
            max_new_tokens=10
        )
        
        response_text = response[0].strip()
        score = float(response_text)
        return max(0.0, min(1.0, score))
    
    def process_video(self, video_frames_dir: Path, video_name: str) -> Dict[str, Any]:
        """处理单个视频，获取其重要性分数序列"""
        frame_paths = self.get_video_frames(video_frames_dir)
        total_frames = len(frame_paths)
        
        if total_frames == 0:
            return {
                "video_name": video_name,
                "total_frames": 0,
                "sampled_frames": 0,
                "scores": []
            }
        
        sampled_indices = self.sample_frames(frame_paths)
        sampled_count = len(sampled_indices)
        
        scores = []
        for i, frame_idx in enumerate(tqdm(sampled_indices, desc=f"Processing {video_name}")):
            context_indices = self.get_context_frames(i, sampled_indices)
            context_frame_paths = [frame_paths[idx] for idx in context_indices]
            score = self.query_frame_importance(context_frame_paths)
            scores.append(score)
        
        self.log_video_scores(video_name, scores)
        
        return {
            "video_name": video_name,
            "total_frames": total_frames,
            "sampled_frames": sampled_count,
            "sampled_indices": sampled_indices,
            "scores": scores,
            "frame_interval": self.frame_interval,
            "context_window": self.frame_context_window
        }
    
    def process_dataset(self, dataset_path: Path) -> Dict[str, Any]:
        """处理单个数据集"""
        frames_dir = dataset_path / "frames"
        if not frames_dir.exists():
            return {"dataset_path": str(dataset_path), "videos": {}}
        
        dataset_name = dataset_path.name
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        
        dataset_results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "total_videos": len(video_dirs),
            "videos": {}
        }
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            video_result = self.process_video(video_dir, video_name)
            dataset_results["videos"][video_name] = video_result
        
        self.log_dataset_summary(dataset_name, dataset_results["videos"])
        return dataset_results
    
    def step1_query_multimodal_model(self) -> Dict[str, Any]:
        """第一步：查询多模态大模型获取帧重要性分数"""
        self.log_hyperparameters()
        
        all_results = {
            "step": 1,
            "description": "Query multimodal model for frame importance scores",
            "parameters": {
                "frame_context_window": self.frame_context_window,
                "frame_interval": self.frame_interval
            },
            "datasets": {}
        }
        
        for dataset_path in self.dataset_paths:
            dataset_result = self.process_dataset(dataset_path)
            dataset_name = dataset_result["dataset_name"]
            all_results["datasets"][dataset_name] = dataset_result
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], filename: str = "step1_results.json"):
        """保存结果到JSON文件"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    def close_logger(self):
        """关闭TensorBoard logger"""
        if self.tb_writer:
            self.tb_writer.finalize("success")
    
    def run(self):
        """执行流程的run方法，调用第一步流程"""
        results = self.step1_query_multimodal_model()
        self.save_results(results)
        self.close_logger()
        return results


if __name__ == "__main__":
    # 示例用法
    dataset_paths = [
        "/path/to/dataset1",
        "/path/to/dataset2"
    ]
    
    pipeline = VideoSummaryPipeline(
        dataset_paths=dataset_paths,
        frame_context_window=3,
        frame_interval=15,
        output_dir="./results"
    )
    
    results = pipeline.run()