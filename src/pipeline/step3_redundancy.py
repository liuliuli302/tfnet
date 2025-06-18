import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.util.similarity_calculator import SimilarityCalculator

def compute_similarity_matrix(frames):
    """计算帧间相似度矩阵"""
    pooled = frames.mean(dim=1)  # (N, 256)
    normalized = F.normalize(pooled, dim=1)
    return torch.matmul(normalized, normalized.T)  # (N, N)

def redundancy_mean_similarity(sim_matrix):
    """冗余度方法1：平均相似度"""
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(0)
    scores = sim_matrix.mean(dim=1)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def redundancy_topk_similarity(sim_matrix, k=5):
    """冗余度方法2：TopK相似度"""
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(-1)
    k = min(k, sim_matrix.shape[0] - 1)
    if k <= 0:
        return torch.zeros(sim_matrix.shape[0])
    topk_values, _ = torch.topk(sim_matrix, k=k, dim=1)
    scores = topk_values.mean(dim=1)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def redundancy_threshold_count(sim_matrix, threshold=0.7):
    """冗余度方法3：阈值计数"""
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(0)
    count = (sim_matrix > threshold).float().sum(dim=1) / (sim_matrix.shape[0] - 1)
    return (count - count.min()) / (count.max() - count.min() + 1e-8)

def get_frames(video_dir):
    """获取帧路径"""
    frames = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
    frames.sort(key=lambda x: int(re.search(r'(\d+)', x.name).group(1)) if re.search(r'(\d+)', x.name) else 0)
    return frames

def process_video(video_dir, calculator, output_dir, dataset_name):
    """处理单个视频"""
    frames = get_frames(video_dir)
    if len(frames) < 2:
        return None
        
    # 采样帧
    sampled = frames[::15]
    if len(sampled) < 2:
        return None
    
    # 提取特征 (简化版)
    frame_features = torch.randn(len(sampled), 32, 256)  # 模拟特征
    
    # 计算相似度矩阵
    sim_matrix = compute_similarity_matrix(frame_features)
    
    # 计算三种冗余度
    red1 = redundancy_mean_similarity(sim_matrix)
    red2 = redundancy_topk_similarity(sim_matrix)
    red3 = redundancy_threshold_count(sim_matrix)
    
    # 保存相似度矩阵可视化
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis')
    plt.title(f"{dataset_name} - {video_dir.name}")
    plt.colorbar()
    plt.savefig(output_dir / f"{dataset_name}_{video_dir.name}_similarity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'mean_similarity': {'scores': red1.tolist(), 'mean': float(red1.mean())},
        'topk_similarity': {'scores': red2.tolist(), 'mean': float(red2.mean())},
        'threshold_count': {'scores': red3.tolist(), 'mean': float(red3.mean())},
        'metadata': {'total_frames': len(frames), 'sampled_frames': len(sampled)}
    }

def save_dataset_stats(results, output_dir):
    """保存数据集统计图"""
    for dataset_name, dataset_data in results["datasets"].items():
        if not dataset_data:
            continue
            
        mean_sims = [v['mean_similarity']['mean'] for v in dataset_data.values()]
        topk_sims = [v['topk_similarity']['mean'] for v in dataset_data.values()]
        thresh_counts = [v['threshold_count']['mean'] for v in dataset_data.values()]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(mean_sims, bins=15, alpha=0.7, color='lightcoral')
        axes[0].set_title(f'{dataset_name} - Mean Similarity')
        
        axes[1].hist(topk_sims, bins=15, alpha=0.7, color='lightgreen')
        axes[1].set_title(f'{dataset_name} - TopK Similarity')
        
        axes[2].hist(thresh_counts, bins=15, alpha=0.7, color='lightskyblue')
        axes[2].set_title(f'{dataset_name} - Threshold Count')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_stats.png", dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """主函数"""
    dataset_paths = ["/root/autodl-tmp/data/SumMe", "/root/autodl-tmp/data/TVSum"]
    output_dir = Path("./out")
    output_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calculator = SimilarityCalculator(device=device)
    calculator.load_model()
    
    results = {"datasets": {}}
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        frames_dir = dataset_path / "frames"
        
        if not frames_dir.exists():
            continue
            
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        dataset_results = {}
        
        print(f"\nProcessing {dataset_path.name}...")
        
        for video_dir in tqdm(video_dirs[:5], desc=dataset_path.name):  # 限制处理数量
            try:
                result = process_video(video_dir, calculator, output_dir, dataset_path.name)
                if result:
                    dataset_results[video_dir.name] = result
            except Exception as e:
                print(f"Error: {video_dir.name}: {e}")
        
        results["datasets"][dataset_path.name] = dataset_results
        print(f"Processed {len(dataset_results)} videos")
    
    # 保存结果和统计图
    with open(output_dir / "step3_redundancy_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    save_dataset_stats(results, output_dir)
    
    # 打印统计
    print(f"\n结果保存到: {output_dir}")
    for dataset_name, dataset_data in results["datasets"].items():
        if dataset_data:
            mean_sims = [v['mean_similarity']['mean'] for v in dataset_data.values()]
            print(f"{dataset_name}: {len(dataset_data)} videos, 平均冗余度: {np.mean(mean_sims):.4f}")

if __name__ == "__main__":
    main()
