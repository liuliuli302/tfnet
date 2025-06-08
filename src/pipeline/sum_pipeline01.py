"""
视频摘要流程 - 第一步实现
免训练的视频摘要流程，通过多模态大模型查询帧重要性

免训练的视频摘要流程

1 查询多模态大模型

输入：
帧+上下文帧

关键参数包括
查询该帧在视频摘要维度重要性程度的Prompt
帧上下文窗口长度N
帧跳数(inteval) M，表示对于视频，每隔15帧采样一次


例子：
帧上下文窗口长度为3，帧跳数为15时
当前帧为0时，输入到模型的帧为[0 15]
当前帧为15时，输入模型的帧为[0 15 30]
当前帧为采样帧序列的最后一帧时候，输入模型的帧为[last-15 last]

这个阶段需要完成的任务如下
已知且已准备好的资源如下
有两个数据集
每个数据集有一个frames文件夹，里面有若干以视频名字标注的子文件夹
子文件内有该视频的所有帧

你需要，首先统计帧数，之后按照帧跳数采样
比如帧跳数为15，则采样[0 15 30 ...]帧，计帧数为N1

之后针对采样的每帧，以帧上下文长度N+Prompt送入多模态大模型，查询以视频摘要为目的的该帧的重要性分数（分数范围为0到1）

最终经历过N1次查询，你得到一个长度为N1的得分序列，作为该视频的的最初的得分序列
你需要将所有数据集的所有视频的得分序列都拿到

并保存其结果json到指定文件夹下

目前已完成第一阶段

2 计算视觉文本相似度
~/autodl-tmp/data/SumMe/
以及
~/autodl-tmp/data/TVSum/
文件夹下包含着文本和视觉特征文件

你需要完成计算一个视频的视频帧特征与视频文本摘要形成的文本特征的相似度计算
得到一个帧相似度分数序列，范围从0到1

帧特征：(N, 32, 2560)
文本特征：(M, 2560)



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
import torch
import torch.nn.functional as F

# 添加项目根目录到Python路径以支持绝对导入
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.handler.llm_client import Qwen25VLClient

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
        logger: Optional[TensorBoardLogger] = None,
        skip_step1: bool = False,
        skip_step2: bool = False
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
            skip_step1: 是否跳过第一步
            skip_step2: 是否跳过第二步
        """
        self.dataset_paths = [Path(path) for path in dataset_paths]
        self.frame_context_window = frame_context_window
        self.frame_interval = frame_interval
        self.skip_step1 = skip_step1
        self.skip_step2 = skip_step2
        self.frame_context_window = frame_context_window
        self.frame_interval = frame_interval
        
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
        
        # 初始化多模态大模型客户端
        self.llm_client = Qwen25VLClient(
            model_name=model_name,
            device_map=device_map
        )
        
        self.importance_prompt = importance_prompt
    
    def log_hyperparameters(self):
        """记录超参数到TensorBoard"""
        if self.logger:
            hparams = {
                'frame_context_window': self.frame_context_window,
                'frame_interval': self.frame_interval,
                'dataset_count': len(self.dataset_paths)
            }
            self.logger.log_hyperparams(hparams)
    
    def log_video_scores(self, video_name: str, scores: List[float], step_offset: int = 0):
        """记录视频的重要性分数到TensorBoard"""
        if self.logger:
            metrics = {}
            for i, score in enumerate(scores):
                metrics[f'frame_importance/{video_name}'] = score
                self.logger.log_metrics(metrics, step=step_offset + i)
            
            if scores:
                stats = {
                    f'video_stats/{video_name}/mean_score': np.mean(scores),
                    f'video_stats/{video_name}/max_score': np.max(scores),
                    f'video_stats/{video_name}/min_score': np.min(scores),
                    f'video_stats/{video_name}/std_score': np.std(scores)
                }
                self.logger.log_metrics(stats)
    
    def log_dataset_summary(self, dataset_name: str, video_results: Dict[str, Dict]):
        """记录数据集汇总信息到TensorBoard"""
        if self.logger:
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
        dataset_name = dataset_path.name
        frames_dir = dataset_path / "frames"
        if not frames_dir.exists():
            print(f"Warning: frames directory not found for dataset {dataset_name}: {frames_dir}")
            return {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_path), 
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
        
        for video_dir in video_dirs:
            video_name = video_dir.name
            video_result = self.process_video(video_dir, video_name)
            dataset_results["videos"][video_name] = video_result
        
        self.log_dataset_summary(dataset_name, dataset_results["videos"])
        return dataset_results
    
    def load_features(self, dataset_path: Path, video_name: str) -> Dict[str, Any]:
        """加载视频的视觉和文本特征"""
        features = {}
        
        # 查找视觉特征文件
        visual_feature_paths = [
            dataset_path / "visual_features" / "BLIP2" / f"{video_name}.npy",
            dataset_path / "visual_features" / f"{video_name}.npy",
            dataset_path / f"visual_features_{video_name}.npy",
            dataset_path / f"{video_name}_visual_features.npy"
        ]
        
        for visual_path in visual_feature_paths:
            if visual_path.exists():
                try:
                    features['visual'] = np.load(visual_path)
                    break
                except Exception as e:
                    print(f"Error loading visual features from {visual_path}: {e}")
                    continue
        
        # 查找文本特征文件  
        text_feature_paths = [
            dataset_path / "text_features" / "BLIP2" / f"{video_name}.npy",
            dataset_path / "text_features" / f"{video_name}.npy",
            dataset_path / f"text_features_{video_name}.npy",
            dataset_path / f"{video_name}_text_features.npy"
        ]
        
        for text_path in text_feature_paths:
            if text_path.exists():
                try:
                    features['text'] = np.load(text_path)
                    break
                except Exception as e:
                    print(f"Error loading text features from {text_path}: {e}")
                    continue
                
        return features
    
    def compute_visual_text_similarity(self, visual_features: np.ndarray, text_features: np.ndarray, method: str = "mean") -> List[float]:
        """
        计算视觉特征和文本特征的相似度
        
        Args:
            visual_features: (N, 32, 2560) 帧特征
            text_features: (M, 2560) 文本特征
            method: 相似度计算方法，"mean" 或 "max"
            
        Returns:
            相似度序列 (N,) 每帧对应一个相似度分数
        """
        
        # 确保输入维度正确
        if visual_features.ndim != 3:
            raise ValueError(f"Expected visual features shape (N, 32, 2560), got {visual_features.shape}")
        if visual_features.shape[1] != 32 or visual_features.shape[2] != 2560:
            raise ValueError(f"Expected visual features shape (N, 32, 2560), got {visual_features.shape}")
            
        # 处理文本特征
        if text_features.ndim == 1:
            text_features = text_features.reshape(1, -1)
        elif text_features.ndim != 2:
            raise ValueError(f"Unexpected text features shape: {text_features.shape}")
        if text_features.shape[1] != 2560:
            raise ValueError(f"Expected text features shape (M, 2560), got {text_features.shape}")
            
        # 转换为torch张量
        visual_tensor = torch.tensor(visual_features, dtype=torch.float32)  # (N, 32, 2560)
        text_tensor = torch.tensor(text_features, dtype=torch.float32)      # (M, 2560)
        
        # L2归一化特征
        visual_tensor_norm = F.normalize(visual_tensor, dim=2)  # (N, 32, 2560)
        text_tensor_norm = F.normalize(text_tensor, dim=1)      # (M, 2560)
        
        # 计算特征矩阵 (N, 32, M)
        # 对于每帧，计算32个patch与M个文本特征的相似度
        N, patches, feat_dim = visual_tensor_norm.shape
        M = text_tensor_norm.shape[0]
        
        similarity_matrix = torch.zeros(N, patches, M, dtype=torch.float32)
        
        for i in range(N):
            # 计算第i帧的32个patch与所有文本特征的相似度矩阵 (32, M)
            frame_patches = visual_tensor_norm[i]  # (32, 2560)
            patch_text_sim = torch.matmul(frame_patches, text_tensor_norm.T)  # (32, M)
            similarity_matrix[i] = patch_text_sim
        
        # 对每帧的相似度矩阵进行聚合
        if method == "mean":
            # 方法1：求平均 - 对32个patch和M个文本取平均
            frame_similarities = similarity_matrix.mean(dim=(1, 2))  # (N,)
        elif method == "max":
            # 方法2：求最大 - 先对M个文本取最大，再对32个patch取最大
            max_over_text = similarity_matrix.max(dim=2)[0]  # (N, 32) - 每个patch的最大文本相似度
            frame_similarities = max_over_text.max(dim=1)[0]  # (N,) - 每帧的最大patch相似度
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'mean' or 'max'")
        
        # 确保相似度在[0,1]范围内
        frame_similarities = torch.clamp(frame_similarities, 0.0, 1.0)
        
        return frame_similarities.cpu().numpy().tolist()
    
    def process_video_step2(self, dataset_path: Path, video_name: str) -> Dict[str, Any]:
        """处理单个视频的视觉文本相似度计算"""
        
        # 加载特征
        features = self.load_features(dataset_path, video_name)
        
        if 'visual' not in features or 'text' not in features:
            missing = []
            if 'visual' not in features:
                missing.append('visual')
            if 'text' not in features:
                missing.append('text')
                
            return {
                "video_name": video_name,
                "status": "missing_features",
                "missing_features": missing,
                "similarities": []
            }
        
        # 计算相似度 - 使用两种方法
        visual_features = features['visual']
        text_features = features['text']
        
        try:
            # 方法1：求平均
            similarities_mean = self.compute_visual_text_similarity(visual_features, text_features, method="mean")
            
            # 方法2：求最大
            similarities_max = self.compute_visual_text_similarity(visual_features, text_features, method="max")
            
            # 记录到TensorBoard
            if self.logger:
                for i, (sim_mean, sim_max) in enumerate(zip(similarities_mean, similarities_max)):
                    self.logger.log_metrics({
                        f'visual_text_similarity_mean/{video_name}': sim_mean,
                        f'visual_text_similarity_max/{video_name}': sim_max
                    }, step=i)
                    
                if similarities_mean and similarities_max:
                    stats = {
                        f'similarity_stats_mean/{video_name}/mean': np.mean(similarities_mean),
                        f'similarity_stats_mean/{video_name}/max': np.max(similarities_mean),
                        f'similarity_stats_mean/{video_name}/min': np.min(similarities_mean),
                        f'similarity_stats_mean/{video_name}/std': np.std(similarities_mean),
                        f'similarity_stats_max/{video_name}/mean': np.mean(similarities_max),
                        f'similarity_stats_max/{video_name}/max': np.max(similarities_max),
                        f'similarity_stats_max/{video_name}/min': np.min(similarities_max),
                        f'similarity_stats_max/{video_name}/std': np.std(similarities_max)
                    }
                    self.logger.log_metrics(stats)
            
            return {
                "video_name": video_name,
                "status": "success",
                "visual_feature_shape": list(visual_features.shape),
                "text_feature_shape": list(text_features.shape),
                "num_similarities": len(similarities_mean),
                "similarities_mean": {
                    "values": similarities_mean,
                    "mean": float(np.mean(similarities_mean)) if similarities_mean else 0.0,
                    "max": float(np.max(similarities_mean)) if similarities_mean else 0.0,
                    "min": float(np.min(similarities_mean)) if similarities_mean else 0.0,
                    "std": float(np.std(similarities_mean)) if similarities_mean else 0.0
                },
                "similarities_max": {
                    "values": similarities_max,
                    "mean": float(np.mean(similarities_max)) if similarities_max else 0.0,
                    "max": float(np.max(similarities_max)) if similarities_max else 0.0,
                    "min": float(np.min(similarities_max)) if similarities_max else 0.0,
                    "std": float(np.std(similarities_max)) if similarities_max else 0.0
                }
            }
            
        except Exception as e:
            return {
                "video_name": video_name,
                "status": "computation_error",
                "error": str(e),
                "visual_feature_shape": list(visual_features.shape),
                "text_feature_shape": list(text_features.shape),
                "similarities_mean": {"values": []},
                "similarities_max": {"values": []}
            }
    
    def process_dataset_step2(self, dataset_path: Path) -> Dict[str, Any]:
        """处理单个数据集的视觉文本相似度计算"""
        dataset_name = dataset_path.name
        
        # 查找视频目录
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
        video_names = [d.name for d in video_dirs]
        
        dataset_results = {
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "total_videos": len(video_names),
            "videos": {}
        }
        
        for video_name in tqdm(video_names, desc=f"Processing {dataset_name} similarities"):
            video_result = self.process_video_step2(dataset_path, video_name)
            dataset_results["videos"][video_name] = video_result
            
        return dataset_results
    
    def step2_compute_visual_text_similarity(self, skip: bool = None) -> Dict[str, Any]:
        """第二步：计算视觉文本相似度"""
        if skip is None:
            skip = self.skip_step2
            
        if skip:
            return {
                "step": 2,
                "description": "Compute visual-text similarity scores (SKIPPED)",
                "status": "skipped",
                "datasets": {}
            }
        
        all_results = {
            "step": 2,
            "description": "Compute visual-text similarity scores",
            "parameters": {
                "frame_interval": self.frame_interval
            },
            "datasets": {}
        }
        
        for dataset_path in self.dataset_paths:
            dataset_result = self.process_dataset_step2(dataset_path)
            dataset_name = dataset_result["dataset_name"]
            all_results["datasets"][dataset_name] = dataset_result
            
        return all_results
    
    def step1_query_multimodal_model(self, skip: bool = None) -> Dict[str, Any]:
        """第一步：查询多模态大模型获取帧重要性分数"""
        if skip is None:
            skip = self.skip_step1
            
        if skip:
            return {
                "step": 1,
                "description": "Query multimodal model for frame importance scores (SKIPPED)",
                "status": "skipped",
                "datasets": {}
            }
            
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
        if self.logger:
            self.logger.finalize("success")
    
    def run(self):
        """执行流程的run方法，调用第一步和第二步流程"""
        results = {}
        
        # 执行第一步 - 强制跳过
        print("Step 1 skipped (forced)")
        step1_results = self.step1_query_multimodal_model(skip=True)
        self.save_results(step1_results, "step1_results.json")
        results['step1'] = step1_results
            
        # 执行第二步
        if not self.skip_step2:
            print("Running Step 2: Compute visual-text similarity...")
            step2_results = self.step2_compute_visual_text_similarity()
            self.save_results(step2_results, "step2_results.json")
            results['step2'] = step2_results
        else:
            print("Step 2 skipped")
            
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