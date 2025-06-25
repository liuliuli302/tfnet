# 测试ImageBind模型进行视觉-文本相似度计算

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置缓存目录
os.environ['XDG_CACHE_HOME'] = '/root/autodl-tmp/.cache'

class ImageBindSimilarityCalculator:
    """ImageBind相似度计算器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """加载ImageBind模型"""
        try:
            # 尝试导入ImageBind
            import imagebind.models.imagebind_model as imagebind_model
            from imagebind.models import imagebind_model
            from imagebind import data
            
            self.imagebind_model = imagebind_model
            self.data = data
            
            # 加载预训练模型
            self.model = imagebind_model.imagebind_huge(pretrained=True)
            self.model.eval()
            self.model.to(self.device)
            print("ImageBind模型加载成功")
            
        except ImportError:
            print("ImageBind未安装，正在安装...")
            self.install_imagebind()
            self.load_model()
    
    def install_imagebind(self):
        """安装ImageBind"""
        import subprocess
        import sys
        
        # 克隆ImageBind仓库
        subprocess.run([
            "git", "clone", "https://github.com/facebookresearch/ImageBind.git", 
            "/root/autodl-tmp/ImageBind"
        ], check=True)
        
        # 安装依赖
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "/root/autodl-tmp/ImageBind"
        ], check=True)
        
        # 添加到Python路径
        sys.path.append("/root/autodl-tmp/ImageBind")
        
        print("ImageBind安装完成")
    
    def load_video_frames(self, video_path: str, frame_interval: int = 1) -> List[np.ndarray]:
        """
        加载视频帧
        Args:
            video_path: 视频文件路径
            frame_interval: 帧间隔
        Returns:
            帧序列列表
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def extract_visual_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        提取视觉特征
        Args:
            frames: 帧序列
        Returns:
            视觉特征 (N, feature_dim)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 转换帧为PIL图像
        pil_images = [Image.fromarray(frame) for frame in frames]
        
        # 预处理图像数据
        image_data = self.data.load_and_transform_vision_data(pil_images, self.device)
        
        # 提取特征
        with torch.no_grad():
            embeddings = self.model({
                self.data.ModalityType.VISION: image_data
            })
            visual_features = embeddings[self.data.ModalityType.VISION]
        
        return visual_features
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        提取文本特征
        Args:
            texts: 文本列表
        Returns:
            文本特征 (M, feature_dim)
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 预处理文本数据
        text_data = self.data.load_and_transform_text(texts, self.device)
        
        # 提取特征
        with torch.no_grad():
            embeddings = self.model({
                self.data.ModalityType.TEXT: text_data
            })
            text_features = embeddings[self.data.ModalityType.TEXT]
        
        return text_features
    
    def calculate_similarity(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        计算相似度
        Args:
            visual_features: 视觉特征 (N, feature_dim)
            text_features: 文本特征 (M, feature_dim)
        Returns:
            相似度矩阵 (N, M)
        """
        # 标准化特征
        visual_norm = F.normalize(visual_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        
        # 计算余弦相似度
        similarity_matrix = torch.matmul(visual_norm, text_norm.T)
        
        return similarity_matrix
    
    def compute_frame_text_similarity(self, video_path: str, text: str, frame_interval: int = 1) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        计算视频帧与文本的相似度
        Args:
            video_path: 视频路径
            text: 文本描述
            frame_interval: 帧间隔
        Returns:
            相似度分数序列, 帧序列
        """
        # 加载视频帧
        frames = self.load_video_frames(video_path, frame_interval)
        print(f"加载了 {len(frames)} 帧")
        
        # 提取视觉特征
        visual_features = self.extract_visual_features(frames)
        print(f"视觉特征形状: {visual_features.shape}")
        
        # 提取文本特征
        text_features = self.extract_text_features([text])
        print(f"文本特征形状: {text_features.shape}")
        
        # 计算相似度
        similarity_matrix = self.calculate_similarity(visual_features, text_features)
        similarity_scores = similarity_matrix.squeeze(-1)  # (N,)
        
        print(f"相似度分数形状: {similarity_scores.shape}")
        print(f"相似度范围: [{similarity_scores.min().item():.4f}, {similarity_scores.max().item():.4f}]")
        print(f"平均相似度: {similarity_scores.mean().item():.4f}")
        
        return similarity_scores.cpu().numpy(), frames

def load_visual_features_from_npy(visual_path: str) -> torch.Tensor:
    """
    从npy文件加载预提取的视觉特征
    Args:
        visual_path: 视觉特征文件路径
    Returns:
        视觉特征tensor
    """
    features = np.load(visual_path)
    return torch.from_numpy(features).cuda().float()

def calc_similarity_with_imagebind_features(visual_features: torch.Tensor, text: str, calculator: ImageBindSimilarityCalculator) -> torch.Tensor:
    """
    使用预提取的视觉特征和ImageBind文本特征计算相似度
    注意：这种方法混合了不同模型的特征，仅用于实验对比
    Args:
        visual_features: 预提取的视觉特征 (N, 32, 256) 或其他形状
        text: 文本描述
        calculator: ImageBind计算器
    Returns:
        相似度分数 (N,)
    """
    # 提取ImageBind文本特征
    text_features = calculator.extract_text_features([text])
    
    # 如果视觉特征是3D的，需要进行降维处理
    if visual_features.dim() == 3:
        # 假设形状是 (N, 32, 256)，取平均或最大值
        visual_features_reduced = visual_features.mean(dim=1)  # (N, 256)
    else:
        visual_features_reduced = visual_features
    
    # 调整特征维度以匹配ImageBind的特征维度
    if visual_features_reduced.shape[-1] != text_features.shape[-1]:
        # 使用线性变换调整维度（这是一个简化的方法）
        feature_dim = text_features.shape[-1]
        projection = torch.nn.Linear(visual_features_reduced.shape[-1], feature_dim).to(visual_features_reduced.device)
        visual_features_reduced = projection(visual_features_reduced)
    
    # 计算相似度
    similarity_matrix = calculator.calculate_similarity(visual_features_reduced, text_features)
    similarity_scores = similarity_matrix.squeeze(-1)
    
    return similarity_scores

def plot_imagebind_similarity_scores(scores: np.ndarray, text: str, video_name: str, save_path: Optional[str] = None):
    """
    绘制ImageBind相似度分数图
    Args:
        scores: 相似度分数序列
        text: 文本描述
        video_name: 视频名称
        save_path: 保存路径
    """
    plt.figure(figsize=(14, 8))
    
    frame_indices = range(len(scores))
    
    # 主要的相似度曲线
    plt.plot(frame_indices, scores, 'g-', linewidth=2.5, alpha=0.8, label='ImageBind Similarity')
    plt.fill_between(frame_indices, scores, alpha=0.2, color='green')
    
    # 添加峰值标记
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + std_score
    
    peaks = np.where(scores > threshold)[0]
    if len(peaks) > 0:
        plt.scatter(peaks, scores[peaks], color='red', s=50, alpha=0.7, 
                   label=f'High Similarity Frames (>{threshold:.3f})')
    
    plt.title(f'ImageBind Visual-Text Similarity - {video_name}\nText: "{text}"', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    stats_text = f'Statistics:\nMean: {mean_score:.4f}\nStd: {std_score:.4f}\nMax: {max_score:.4f}\nMin: {min_score:.4f}\nPeaks: {len(peaks)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 添加阈值线
    plt.axhline(y=mean_score, color='orange', linestyle='--', alpha=0.7, label=f'Mean ({mean_score:.3f})')
    plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold:.3f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ImageBind相似度图保存到: {save_path}")
    
    plt.show()

def plot_comparison_imagebind_vs_blip(imagebind_scores: np.ndarray, blip_scores: np.ndarray, 
                                      text: str, video_name: str, save_path: Optional[str] = None):
    """
    比较ImageBind和BLIP2的相似度分数
    Args:
        imagebind_scores: ImageBind相似度分数
        blip_scores: BLIP2相似度分数
        text: 文本描述
        video_name: 视频名称
        save_path: 保存路径
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    frame_indices = range(len(imagebind_scores))
    
    # 子图1: ImageBind结果
    ax1.plot(frame_indices, imagebind_scores, 'g-', linewidth=2, alpha=0.8, label='ImageBind')
    ax1.fill_between(frame_indices, imagebind_scores, alpha=0.3, color='green')
    ax1.set_title(f'ImageBind Similarity - {video_name}', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Similarity Score')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 统计信息
    ib_mean, ib_std = np.mean(imagebind_scores), np.std(imagebind_scores)
    ax1.text(0.02, 0.98, f'Mean: {ib_mean:.4f}\nStd: {ib_std:.4f}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 子图2: BLIP2结果
    ax2.plot(frame_indices, blip_scores, 'b-', linewidth=2, alpha=0.8, label='BLIP2')
    ax2.fill_between(frame_indices, blip_scores, alpha=0.3, color='blue')
    ax2.set_title('BLIP2 Similarity', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Similarity Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 统计信息
    blip_mean, blip_std = np.mean(blip_scores), np.std(blip_scores)
    ax2.text(0.02, 0.98, f'Mean: {blip_mean:.4f}\nStd: {blip_std:.4f}',
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 子图3: 对比图
    ax3.plot(frame_indices, imagebind_scores, 'g-', linewidth=2, alpha=0.8, label='ImageBind')
    ax3.plot(frame_indices, blip_scores, 'b-', linewidth=2, alpha=0.8, label='BLIP2')
    ax3.set_title(f'Comparison: ImageBind vs BLIP2\nText: "{text}"', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Similarity Score')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 相关性分析
    correlation = np.corrcoef(imagebind_scores, blip_scores)[0, 1]
    diff_mean = np.mean(imagebind_scores - blip_scores)
    ax3.text(0.02, 0.98, f'Correlation: {correlation:.4f}\nMean Diff (IB-BLIP): {diff_mean:.4f}',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图保存到: {save_path}")
    
    plt.show()

def test_imagebind_with_video(video_path: str, text: str, output_dir: str):
    """
    测试ImageBind模型处理视频
    Args:
        video_path: 视频文件路径
        text: 测试文本
        output_dir: 输出目录
    """
    print(f"测试视频: {video_path}")
    print(f"测试文本: {text}")
    
    # 创建ImageBind计算器
    calculator = ImageBindSimilarityCalculator()
    
    # 计算相似度
    scores, frames = calculator.compute_frame_text_similarity(video_path, text, frame_interval=5)
    
    # 获取视频名称
    video_name = Path(video_path).stem
    
    # 保存结果图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{video_name}_imagebind_similarity.png")
    
    # 绘制结果
    plot_imagebind_similarity_scores(scores, text, video_name, save_path)
    
    return scores, frames

def test_imagebind_with_preextracted_features(visual_root: str, video_name: str, text: str, output_dir: str):
    """
    测试ImageBind使用预提取的特征（混合模式）
    Args:
        visual_root: 视觉特征根目录
        video_name: 视频名称
        text: 测试文本
        output_dir: 输出目录
    """
    visual_path = os.path.join(visual_root, f"{video_name}.npy")
    
    if not os.path.exists(visual_path):
        print(f"视觉特征文件不存在: {visual_path}")
        return None
    
    print(f"加载视觉特征: {visual_path}")
    print(f"测试文本: {text}")
    
    # 创建ImageBind计算器
    calculator = ImageBindSimilarityCalculator()
    
    # 加载预提取的视觉特征
    visual_features = load_visual_features_from_npy(visual_path)
    print(f"视觉特征形状: {visual_features.shape}")
    
    # 计算相似度（混合模式）
    scores = calc_similarity_with_imagebind_features(visual_features, text, calculator)
    
    print(f"相似度分数形状: {scores.shape}")
    print(f"相似度范围: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"平均相似度: {scores.mean().item():.4f}")
    
    # 保存结果图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{video_name}_imagebind_mixed_similarity.png")
    
    # 绘制结果
    plot_imagebind_similarity_scores(scores.cpu().numpy(), text, video_name, save_path)
    
    return scores.cpu().numpy()

def main():
    """主函数"""
    print("=" * 60)
    print("ImageBind视觉-文本相似度测试")
    print("=" * 60)
    
    # 设置路径
    visual_root = "/root/autodl-tmp/data/SumMe/features/visual"
    output_dir = "/root/tfnet/test/imagebind_plots"
    
    # 测试文本
    test_text = "A person's face"
    
    # 检查是否有预提取的特征
    if os.path.exists(visual_root):
        print("\n1. 使用预提取特征进行测试（混合模式）")
        visual_files = glob.glob(os.path.join(visual_root, "*.npy"))
        
        if visual_files:
            # 选择第一个视频进行测试
            target_video = os.path.basename(visual_files[0]).replace('.npy', '')
            print(f"测试视频: {target_video}")
            
            # 测试混合模式
            imagebind_scores = test_imagebind_with_preextracted_features(
                visual_root, target_video, test_text, output_dir
            )
            
            if imagebind_scores is not None:
                print(f"ImageBind混合模式平均相似度: {np.mean(imagebind_scores):.4f}")
    
    # 如果有实际视频文件，也可以进行端到端测试
    video_dir = "/root/autodl-tmp/data/SumMe/videos"
    if os.path.exists(video_dir):
        print("\n2. 使用原始视频进行端到端测试")
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        
        if video_files:
            # 选择第一个视频
            test_video = video_files[0]
            print(f"测试视频: {test_video}")
            
            try:
                scores, frames = test_imagebind_with_video(test_video, test_text, output_dir)
                print(f"ImageBind端到端平均相似度: {np.mean(scores):.4f}")
                print(f"处理了 {len(frames)} 帧")
            except Exception as e:
                print(f"端到端测试失败: {e}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
