"""极简版分段相似度计算"""
import json
import re
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import sys
import warnings

# 忽略所有警告信息
warnings.filterwarnings("ignore")
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.util.similarity_calculator import SimilarityCalculator


def get_frames(video_dir):
    """获取帧路径"""
    frames = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
    frames.sort(key=lambda x: int(re.search(r'(\d+)', x.name).group(1)) if re.search(r'(\d+)', x.name) else 0)
    return frames

def get_caption(dataset_path, video_name):
    """获取视频描述"""
    caption_file = dataset_path / "captions" / "Qwen25VL" / f"{video_name}.txt"
    return caption_file.read_text().strip() if caption_file.exists() else "video summary"

def segment_list(items, n_segments):
    """分割列表"""
    if len(items) <= n_segments:
        return [[item] for item in items]
    size = len(items) // n_segments
    return [items[i*size:(i+1)*size if i < n_segments-1 else len(items)] for i in range(n_segments)]

def process_video(video_dir, dataset_path, calculator, frame_interval=15):
    """处理单个视频"""
    frames = get_frames(video_dir)
    sampled = frames[::frame_interval]
    caption = get_caption(dataset_path, video_dir.name)
    
    results = {}
    for n_seg in [2, 4, 6, 8, 16, 32]:  # 只用2个分段数
        # 分割文本和帧
        words = caption.split()
        text_segs = [' '.join(seg) for seg in segment_list(words, n_seg)]
        frame_segs = segment_list(sampled, n_seg)
        
        similarities = []
        for frame_seg, text_seg in zip(frame_segs, text_segs):
            try:
                images = [Image.open(f).convert('RGB') for f in frame_seg]
                sims = calculator.compute_similarity(frames=images, text=text_seg, batch_size=1)
                similarities.extend(sims.tolist())
            except:
                continue
        
        results[f"seg_{n_seg}"] = {
            "similarities": similarities,
            "mean": float(np.mean(similarities)) if similarities else 0.0
        }
    
    return results

def main():
    """主函数"""
    dataset_paths = ["/root/autodl-tmp/data/SumMe", "/root/autodl-tmp/data/TVSum"]
    output_dir = Path("./out")
    output_dir.mkdir(exist_ok=True)
    
    # 初始化计算器
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
        
        for video_dir in tqdm(video_dirs, desc=f"Processing {dataset_path.name}"):
            try:
                dataset_results[video_dir.name] = process_video(video_dir, dataset_path, calculator)
            except Exception as e:
                print(f"Error: {video_dir.name}: {e}")
                continue
        
        results["datasets"][dataset_path.name] = dataset_results
    
    # 保存结果
    output_file = output_dir / "step2_simple_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"完成！结果保存到: {output_file}")

if __name__ == "__main__":
    main()
