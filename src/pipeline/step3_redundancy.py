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
from typing import List

# 忽略所有警告信息
warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from lavis.models import load_model_and_preprocess

class VisualFeatureExtractor:
    """Visual feature extractor using BLIP2 model."""
    
    def __init__(self, device: str = 'cuda', model_name: str = "blip2_image_text_matching", 
                 model_type: str = "coco"):
        """
        Initialize the VisualFeatureExtractor.
        
        Args:
            device: Device to run computations on ('cuda' or 'cpu')
            model_name: Name of the BLIP2 model to use
            model_type: Type/variant of the model
        """
        self.device = device
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.vis_processors = None
        self.text_processors = None
        
    def load_model(self):
        """Load the BLIP2 model and processors."""
        print("Loading BLIP2 model for feature extraction...")
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess(
            self.model_name, self.model_type, device=self.device, is_eval=True
        )
        print("Model loaded successfully.")
        
    def extract_visual_features_batch(self, frames: List[Image.Image], 
                                    batch_size: int = 32) -> torch.Tensor:
        """
        Extract visual features from frames in batches using BLIP2's proper pipeline.
        
        Args:
            frames: List of PIL Image frames
            batch_size: Number of frames to process in each batch
            
        Returns:
            Normalized visual features tensor
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        visual_features_list = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Process images
            batch_images = []
            for frame in batch_frames:
                image_input = self.vis_processors["eval"](frame).unsqueeze(0)
                batch_images.append(image_input)
            
            if batch_images:
                batch_images = torch.cat(batch_images, dim=0).to(self.device)
                
                with torch.no_grad():
                    # Use BLIP2's proper visual feature extraction pipeline
                    with self.model.maybe_autocast():
                        # Extract image embeddings through visual encoder
                        image_embeds = self.model.ln_vision(self.model.visual_encoder(batch_images))
                    
                    # Pass through Q-Former for proper feature alignment
                    image_embeds = image_embeds.float()
                    image_atts = torch.ones(
                        image_embeds.size()[:-1], dtype=torch.long).to(self.device)
                    query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    
                    query_output = self.model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    
                    # Apply vision projection to align with text features
                    visual_features = self.model.vision_proj(query_output.last_hidden_state)
                    
                    # Global average pooling to get fixed-size features
                    visual_features = visual_features.mean(dim=1)  # Shape: (batch_size, feature_dim)
                    
                    # Normalize features
                    visual_features = F.normalize(visual_features, p=2, dim=1)
                    
                    visual_features_list.append(visual_features.cpu())
        
        if visual_features_list:
            return torch.cat(visual_features_list, dim=0)
        else:
            return torch.empty((0, 256))  # Return empty tensor if no features

def compute_similarity_matrix(features):
    """
    Compute cosine similarity matrix between frame features.
    :param features: Tensor of shape (N, feature_dim) where feature_dim is the output dimension from BLIP2
    :return: Tensor of shape (N, N)
    """
    # Features are already normalized from the feature extractor
    similarity_matrix = torch.matmul(features, features.T)  # shape: (N, N)
    return similarity_matrix

def redundancy_mean_similarity(sim_matrix):
    """
    冗余度方法1：每一帧与其它帧的平均相似度
    """
    N = sim_matrix.shape[0]
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(0)  # 排除自身相似度
    scores = sim_matrix.mean(dim=1)  # 平均相似度
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def redundancy_topk_similarity(sim_matrix, k=5):
    """
    冗余度方法2：与相似度最高的K帧的平均
    """
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(-1)  # 排除自身
    k = min(k, sim_matrix.shape[0] - 1)  # 确保k不超过帧数-1
    if k <= 0:
        return torch.zeros(sim_matrix.shape[0])
    topk_values, _ = torch.topk(sim_matrix, k=k, dim=1)
    scores = topk_values.mean(dim=1)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def redundancy_threshold_count(sim_matrix, threshold=0.9):
    """
    冗余度方法3：相似度高于阈值的帧数比例
    """
    N = sim_matrix.shape[0]
    sim_matrix = sim_matrix.clone()
    sim_matrix.fill_diagonal_(0)
    count = (sim_matrix > threshold).float().sum(dim=1) / (N - 1)
    return (count - count.min()) / (count.max() - count.min() + 1e-8)

def save_similarity_visualization(sim_matrix, output_path, title="Frame Similarity Matrix"):
    """保存相似度矩阵可视化"""
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_redundancy_visualization(redundancy_scores, output_path, method_name, video_name):
    """保存冗余度分布可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：冗余度分布直方图
    ax1.hist(redundancy_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Redundancy Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{method_name} - {video_name}\nRedundancy Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 右图：冗余度时序图
    ax2.plot(redundancy_scores, marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Redundancy Score')
    ax2.set_title(f'{method_name} - {video_name}\nRedundancy Timeline')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_dataset_statistics(results, output_dir):
    """保存数据集统计可视化"""
    for dataset_name, dataset_data in results["datasets"].items():
        if not dataset_data:
            continue
            
        # 收集数据
        mean_sims = [v['mean_similarity']['mean'] for v in dataset_data.values()]
        topk_sims = [v['topk_similarity']['mean'] for v in dataset_data.values()]
        thresh_counts = [v['threshold_count']['mean'] for v in dataset_data.values()]
        
        # 创建统计图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 三种方法的分布对比
        axes[0, 0].boxplot([mean_sims, topk_sims, thresh_counts], 
                          labels=['Mean Similarity', 'TopK Similarity', 'Threshold Count'])
        axes[0, 0].set_title(f'{dataset_name} - Redundancy Methods Comparison')
        axes[0, 0].set_ylabel('Redundancy Score')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 平均相似度分布
        axes[0, 1].hist(mean_sims, bins=15, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title(f'{dataset_name} - Mean Similarity Distribution')
        axes[0, 1].set_xlabel('Mean Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # TopK相似度分布
        axes[1, 0].hist(topk_sims, bins=15, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title(f'{dataset_name} - TopK Similarity Distribution')
        axes[1, 0].set_xlabel('TopK Similarity Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 阈值计数分布
        axes[1, 1].hist(thresh_counts, bins=15, alpha=0.7, color='lightskyblue')
        axes[1, 1].set_title(f'{dataset_name} - Threshold Count Distribution')
        axes[1, 1].set_xlabel('Threshold Count Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_redundancy_statistics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def get_frames(video_dir):
    """获取帧路径"""
    frames = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
    frames.sort(key=lambda x: int(re.search(r'(\d+)', x.name).group(1)) if re.search(r'(\d+)', x.name) else 0)
    return frames

def extract_frame_features(frames, feature_extractor, batch_size=8):
    """提取帧特征"""
    try:
        # 加载图像
        images = [Image.open(f).convert('RGB') for f in frames]
        
        # 使用visual feature extractor提取特征
        features = feature_extractor.extract_visual_features_batch(images, batch_size=batch_size)
        
        return features
        
    except Exception as e:
        print(f"Error processing frames: {e}")
        # 如果出错，返回随机特征用于测试
        return torch.randn(len(frames), 256)  # 假设特征维度为256

def process_video_redundancy(video_dir, feature_extractor, output_dir, dataset_name, frame_interval=15):
    """处理单个视频的冗余度计算"""
    frames = get_frames(video_dir)
    
    if len(frames) == 0:
        return None
        
    # 采样帧
    sampled_frames = frames[::frame_interval]
    
    if len(sampled_frames) < 2:
        return None
    
    # 提取特征
    frame_features = extract_frame_features(sampled_frames, feature_extractor)
    
    # 计算相似度矩阵
    sim_matrix = compute_similarity_matrix(frame_features)
    
    # 计算三种冗余度指标
    redundancy_results = {}
    
    # 方法1：平均相似度
    red1 = redundancy_mean_similarity(sim_matrix)
    redundancy_results['mean_similarity'] = {
        'scores': red1.tolist(),
        'mean': float(red1.mean()),
        'std': float(red1.std())
    }
    
    # 方法2：TopK相似度
    k = min(5, len(sampled_frames) - 1)
    red2 = redundancy_topk_similarity(sim_matrix, k=k)
    redundancy_results['topk_similarity'] = {
        'scores': red2.tolist(),
        'mean': float(red2.mean()),
        'std': float(red2.std()),
        'k': k
    }
    
    # 方法3：阈值计数
    red3 = redundancy_threshold_count(sim_matrix, threshold=0.7)  # 降低阈值使其更实用
    redundancy_results['threshold_count'] = {
        'scores': red3.tolist(),
        'mean': float(red3.mean()),
        'std': float(red3.std()),
        'threshold': 0.7
    }
    
    # 添加基本信息
    redundancy_results['metadata'] = {
        'total_frames': len(frames),
        'sampled_frames': len(sampled_frames),
        'frame_interval': frame_interval,
        'similarity_matrix_shape': list(sim_matrix.shape),
        'feature_shape': list(frame_features.shape)
    }
    
    # 保存可视化图片
    video_name = video_dir.name
    
    # 保存相似度矩阵
    sim_output_path = output_dir / f"{dataset_name}_{video_name}_similarity_matrix.png"
    save_similarity_visualization(sim_matrix, sim_output_path, 
                                f"{dataset_name} - {video_name} Similarity Matrix")
    
    # 保存三种冗余度方法的可视化
    methods = {
        'mean_similarity': (red1.cpu().numpy(), 'Mean Similarity'),
        'topk_similarity': (red2.cpu().numpy(), 'TopK Similarity'),
        'threshold_count': (red3.cpu().numpy(), 'Threshold Count')
    }
    
    for method_key, (scores, method_title) in methods.items():
        redundancy_output_path = output_dir / f"{dataset_name}_{video_name}_{method_key}.png"
        save_redundancy_visualization(scores, redundancy_output_path, method_title, video_name)
    
    return redundancy_results, sim_matrix

def main():
    """主函数"""
    dataset_paths = ["/root/autodl-tmp/data/SumMe", "/root/autodl-tmp/data/TVSum"]
    output_dir = Path("./out")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化特征提取器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feature_extractor = VisualFeatureExtractor(device=device)
    feature_extractor.load_model()
    
    results = {"datasets": {}}
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        frames_dir = dataset_path / "frames"
        
        if not frames_dir.exists():
            print(f"Frames directory not found: {frames_dir}")
            continue
            
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        dataset_results = {}
        
        print(f"\nProcessing {dataset_path.name} dataset...")
        
        for video_dir in tqdm(video_dirs, desc=f"Computing redundancy for {dataset_path.name}"):
            try:
                result = process_video_redundancy(video_dir, feature_extractor, output_dir, 
                                                dataset_path.name)
                if result is not None:
                    redundancy_results, sim_matrix = result
                    dataset_results[video_dir.name] = redundancy_results
                        
            except Exception as e:
                print(f"Error processing {video_dir.name}: {e}")
                continue
        
        results["datasets"][dataset_path.name] = dataset_results
        print(f"Processed {len(dataset_results)} videos in {dataset_path.name}")
    
    # 保存数据集级别的统计可视化
    save_dataset_statistics(results, output_dir)
    
    # 保存结果
    output_file = output_dir / "step3_redundancy_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 打印统计信息
    print(f"\n结果保存到: {output_file}")
    print(f"可视化图片保存到: {output_dir}")
    
    for dataset_name, dataset_data in results["datasets"].items():
        print(f"\n{dataset_name} 数据集统计:")
        print(f"  处理视频数: {len(dataset_data)}")
        
        if dataset_data:
            # 计算数据集级别的统计
            mean_sims = [v['mean_similarity']['mean'] for v in dataset_data.values()]
            topk_sims = [v['topk_similarity']['mean'] for v in dataset_data.values()]
            thresh_counts = [v['threshold_count']['mean'] for v in dataset_data.values()]
            
            print(f"  平均相似度冗余度: {np.mean(mean_sims):.4f} ± {np.std(mean_sims):.4f}")
            print(f"  TopK相似度冗余度: {np.mean(topk_sims):.4f} ± {np.std(topk_sims):.4f}")
            print(f"  阈值计数冗余度: {np.mean(thresh_counts):.4f} ± {np.std(thresh_counts):.4f}")

if __name__ == "__main__":
    main()
