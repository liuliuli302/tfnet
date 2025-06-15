import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lavis.models import load_model_and_preprocess
import torch.nn.functional as F
import os

def extract_frames(video_path, fps=None, stride=15, max_duration=None):
    """Extract frames from video with specified parameters."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling rate
    if fps is None:
        frame_interval = stride
    else:
        frame_interval = max(1, int(original_fps / fps))
    
    frames = []
    frame_indices = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(frame_count)
            
            # Check max duration
            if max_duration and len(frames) >= max_duration * (fps or original_fps / frame_interval):
                break
                
        frame_count += 1
    
    cap.release()
    return frames, frame_indices, original_fps

def extract_text_features(model, text_processor, text, device):
    """Extract text features using the BLIP2 model."""
    text_input = text_processor["eval"](text)
    
    with torch.no_grad():
        # Use BLIP2's proper text feature extraction pipeline
        text_tokens = model.tokenizer(
            text,
            padding='max_length', 
            truncation=True, 
            max_length=35, 
            return_tensors="pt"
        ).to(device)
        
        # Extract text features through Q-Former
        with model.maybe_autocast():
            text_output = model.Qformer.bert(
                text_tokens.input_ids, 
                attention_mask=text_tokens.attention_mask, 
                return_dict=True
            )
            # Use vision projection to align with visual features
            text_features = model.text_proj(text_output.last_hidden_state[:, 0, :])
        
        # Normalize features
        text_features = F.normalize(text_features, p=2, dim=1)
        
    return text_features

def extract_visual_features_batch(model, vis_processor, frames, batch_size, device):
    """Extract visual features from frames in batches using BLIP2's proper pipeline."""
    visual_features_list = []
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i+batch_size]
        
        # Process images
        batch_images = []
        for frame in batch_frames:
            image_input = vis_processor["eval"](frame).unsqueeze(0)
            batch_images.append(image_input)
        
        if batch_images:
            batch_images = torch.cat(batch_images, dim=0).to(device)
            
            with torch.no_grad():
                # Use BLIP2's proper visual feature extraction pipeline
                with model.maybe_autocast():
                    # Extract image embeddings through visual encoder
                    image_embeds = model.ln_vision(model.visual_encoder(batch_images))
                
                # Pass through Q-Former for proper feature alignment
                image_embeds = image_embeds.float()
                image_atts = torch.ones(
                    image_embeds.size()[:-1], dtype=torch.long).to(device)
                query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                
                query_output = model.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                
                # Apply vision projection to align with text features
                visual_features = model.vision_proj(query_output.last_hidden_state)
                
                # Global average pooling to get fixed-size features
                visual_features = visual_features.mean(dim=1)  # Shape: (batch_size, feature_dim)
                
                # Normalize features
                visual_features = F.normalize(visual_features, p=2, dim=1)
                
                visual_features_list.append(visual_features.cpu())
    
    if visual_features_list:
        return torch.cat(visual_features_list, dim=0)
    else:
        return torch.empty((0, 256))  # Return empty tensor if no features

def compute_similarity_from_features(text_features, visual_features):
    """Compute cosine similarity between text and visual features."""
    # Ensure both features are on the same device and data type
    if text_features.device != visual_features.device:
        visual_features = visual_features.to(text_features.device)
    
    # Ensure both features have the same data type (convert to float32)
    text_features = text_features.float()
    visual_features = visual_features.float()
    
    # Compute cosine similarity
    similarities = torch.mm(visual_features, text_features.t()).squeeze()
    
    return similarities.cpu().numpy()

def compute_similarity_batch(model, vis_processor, text_processor, frames, text, batch_size, device):
    """Compute similarity scores between frames and text using feature extraction."""
    print("Extracting text features...")
    text_features = extract_text_features(model, text_processor, text, device)
    
    print("Extracting visual features...")
    visual_features = extract_visual_features_batch(model, vis_processor, frames, batch_size, device)
    
    print("Computing similarities from features...")
    similarities = compute_similarity_from_features(text_features, visual_features)
    
    return similarities

def plot_similarity_timeline(similarities, frame_indices, fps, text, output_dir, video_name):
    """Plot similarity scores over time."""
    # Ensure similarities is 1D
    similarities = similarities.flatten()
    
    # Convert frame indices to time
    time_stamps = np.array(frame_indices) / fps
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_stamps, similarities, linewidth=2, color='blue', alpha=0.7)
    plt.fill_between(time_stamps, similarities, alpha=0.3, color='blue')
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Similarity Score', fontsize=12)
    plt.title(f'Text-Video Similarity Timeline\nQuery: "{text}"', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    max_sim = np.max(similarities)
    max_time = time_stamps[np.argmax(similarities)]
    min_sim = np.min(similarities)
    
    plt.axhline(y=max_sim, color='red', linestyle='--', alpha=0.5)
    plt.text(0.02, 0.98, f'Max: {max_sim:.3f} at {max_time:.1f}s\nMin: {min_sim:.3f}\nMean: {np.mean(similarities):.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{video_name}_similarity_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_path}")
    return plot_path

def main():
    # Parameters
    video_path = '/root/autodl-tmp/data/SumMe/videos/Base_jumping.mp4'
    text = "Road"
    fps = None
    stride = 15
    max_duration = None
    batch_size = 32
    output_dir = '/root/tfnet/test/similarity_plots'
    device = 'cuda'
    
    print("Loading BLIP2 model...")
    model, vis_processors, text_processors = load_model_and_preprocess(
        "blip2_image_text_matching", "coco", device=device, is_eval=True
    )
    
    print("Extracting frames from video...")
    frames, frame_indices, original_fps = extract_frames(
        video_path, fps=fps, stride=stride, max_duration=max_duration
    )
    
    print(f"Extracted {len(frames)} frames from video")
    print(f"Original FPS: {original_fps}")
    
    print("Computing similarity scores...")
    similarities = compute_similarity_batch(
        model, vis_processors, text_processors, frames, text, batch_size, device
    )
    
    print(f"Computed similarities for {len(similarities)} frames")
    print(f"Similarity shape: {similarities.shape}")
    print(f"Similarity range: {similarities.min():.3f} - {similarities.max():.3f}")
    
    # Get video name for saving
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    print("Creating similarity plot...")
    plot_path = plot_similarity_timeline(
        similarities, frame_indices, original_fps, text, output_dir, video_name
    )
    
    # Save raw data
    data_path = os.path.join(output_dir, f'{video_name}_similarity_data.npz')
    np.savez(data_path, 
             similarities=similarities, 
             frame_indices=frame_indices, 
             fps=original_fps,
             text=text)
    print(f"Raw data saved to: {data_path}")

if __name__ == "__main__":
    main()