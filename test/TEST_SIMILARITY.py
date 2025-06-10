import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from torchvision import transforms
from decord import VideoReader, cpu

# Set cache directory
os.environ['XDG_CACHE_HOME'] = '/root/autodl-tmp/.cache'

class VideoTextSimilarityCalculator:
    def __init__(self, device='cuda'):
        self.device = device
        # Load BLIP-2 model
        self.model, vis_processors, text_processors = load_model_and_preprocess(
            "blip2_image_text_matching", "coco", device=device, is_eval=True
        )
        # Remove ToTensor transform as we'll handle it manually
        self.vis_processors = transforms.Compose([
            t for t in vis_processors['eval'].transform.transforms 
            if not isinstance(t, transforms.ToTensor)
        ])
    
    def load_video(self, video_path, fps=3, stride=None, max_duration=None):
        """Load video frames with specified sampling rate"""
        vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        duration = len(vr) / vr.get_avg_fps()
        
        if fps is not None:
            num_sampled_frames = round(duration * fps)
            all_index = np.linspace(0, len(vr)-1, num=num_sampled_frames).round().astype(np.int32)
            if max_duration is not None:
                all_index = all_index[:round(max_duration * fps)]
        else:
            assert stride is not None
            all_index = np.arange(0, len(vr), stride, dtype=np.int32)
            if max_duration is not None:
                all_index = all_index[:round(max_duration * all_index.shape[0] / duration)]
        
        vr.seek(0)
        buffer = vr.get_batch(all_index).permute(0, 3, 1, 2) / 255.
        
        # Calculate frame timestamps
        timestamps = all_index / vr.get_avg_fps()
        
        return buffer, timestamps, duration
    
    @torch.no_grad()
    def extract_visual_features(self, video_frames, batch_size=32):
        """Extract visual features from video frames"""
        img = self.vis_processors(video_frames)
        features = []
        
        for bid in range(0, img.size(0), batch_size):
            batch_img = img[bid:bid+batch_size].to(self.device)
            with self.model.maybe_autocast():
                image_embeds = self.model.ln_vision(self.model.visual_encoder(batch_img))
            image_embeds = image_embeds.float()
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_feats = self.model.vision_proj(query_output.last_hidden_state)
            features.append(image_feats.cpu())
        
        return torch.cat(features, dim=0)
    
    @torch.no_grad()
    def extract_text_features(self, text):
        """Extract text features"""
        text_input = self.model.tokenizer(
            text, padding='max_length', truncation=True, 
            max_length=35, return_tensors="pt"
        ).to(self.device)
        
        text_output = self.model.Qformer.bert(
            text_input.input_ids, 
            attention_mask=text_input.attention_mask, 
            return_dict=True
        )
        text_feat = self.model.text_proj(text_output.last_hidden_state[:, 0, :])
        return text_feat.cpu()
    
    def calculate_similarities(self, video_features, text_features):
        """Calculate similarity between video frames and text"""
        # Normalize features
        video_features = F.normalize(video_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Calculate similarities for each frame
        similarities = []
        for frame_idx in range(video_features.shape[0]):
            frame_feat = video_features[frame_idx]  # Shape: [num_queries, feat_dim]
            # Calculate similarity between text and each query, then take max
            sim = torch.einsum('d,qd->q', text_features.squeeze(), frame_feat)
            similarities.append(sim.mean().item())
        
        return np.array(similarities)
    
    def plot_similarities(self, timestamps, similarities, text, video_path, save_path=None):
        """Plot similarity scores over time"""
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, similarities, 'b-', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Similarity Score')
        plt.title(f'Video-Text Similarity Over Time\nVideo: {Path(video_path).name}\nText: "{text}"')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def process_video(self, video_path, text, fps=3, stride=None, max_duration=None, 
                     batch_size=32, save_plot=True, output_dir="./results"):
        """Main processing function"""
        print(f"Processing video: {video_path}")
        print(f"Text query: {text}")
        
        # Load video
        video_frames, timestamps, duration = self.load_video(
            video_path, fps=fps, stride=stride, max_duration=max_duration
        )
        print(f"Loaded {len(video_frames)} frames from {duration:.2f}s video")
        
        # Extract features
        print("Extracting visual features...")
        video_features = self.extract_visual_features(video_frames, batch_size)
        
        print("Extracting text features...")
        text_features = self.extract_text_features(text)
        
        # Calculate similarities
        print("Calculating similarities...")
        similarities = self.calculate_similarities(video_features, text_features)
        
        # Plot results
        if save_plot:
            os.makedirs(output_dir, exist_ok=True)
            video_name = Path(video_path).stem
            save_path = os.path.join(output_dir, f"{video_name}_similarity.png")
        else:
            save_path = None
        
        self.plot_similarities(timestamps, similarities, text, video_path, save_path)
        
        # Return results
        return {
            'timestamps': timestamps,
            'similarities': similarities,
            'video_duration': duration,
            'num_frames': len(video_frames)
        }

def main():
    # Hardcoded parameters - modify these as needed
    video_path = '/root/autodl-tmp/data/SumMe/videos/Bus_in_Rock_Tunnel.mp4'  # Change this to your video path
    text = "a man in shirt"  # Change this to your text query
    fps = None
    stride = 15
    max_duration = None
    batch_size = 32
    output_dir = '/root/tfnet/test/similarity_plots'
    device = 'cuda'
    
    # Validate inputs
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please modify the video_path variable in the main() function to point to an existing video file.")
        return
    
    # Initialize calculator
    calculator = VideoTextSimilarityCalculator(device=device)
    
    # Process video
    results = calculator.process_video(
        video_path=video_path,
        text=text,
        fps=fps,
        stride=stride,
        max_duration=max_duration,
        batch_size=batch_size,
        output_dir=output_dir
    )
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Video: {video_path}")
    print(f"Text: {text}")
    print(f"Duration: {results['video_duration']:.2f}s")
    print(f"Frames processed: {results['num_frames']}")
    print(f"Max similarity: {results['similarities'].max():.4f}")
    print(f"Min similarity: {results['similarities'].min():.4f}")
    print(f"Mean similarity: {results['similarities'].mean():.4f}")
    
    # Find peak similarity moments
    peak_indices = np.argsort(results['similarities'])[-3:][::-1]
    print(f"\nTop 3 similarity peaks:")
    for i, idx in enumerate(peak_indices, 1):
        print(f"{i}. Time: {results['timestamps'][idx]:.2f}s, Score: {results['similarities'][idx]:.4f}")

if __name__ == '__main__':
    main()