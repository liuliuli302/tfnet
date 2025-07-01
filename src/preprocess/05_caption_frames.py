#!/usr/bin/env python3
"""
Script to caption frames for SumMe and TVSum datasets using ImageCaptioner
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path to import util modules
sys.path.append(str(Path(__file__).parent.parent))

from util.image_captioner import ImageCaptioner


def main():
    """Caption frames for both SumMe and TVSum datasets"""
    
    # Dataset paths
    summe_frames_dir = os.path.expanduser("~/autodl-tmp/data/SumMe/frames")
    tvsum_frames_dir = os.path.expanduser("~/autodl-tmp/data/TVSum/frames")
    
    # Initialize ImageCaptioner with default parameters
    captioner = ImageCaptioner(
        batch_size=128,  # Process 4 images at once for better efficiency
        frame_interval=1,  # Will be overridden by caption_frames method
        imagefile_template="frame_{:06d}.jpg",  # Standard frame naming format
        pretrained_model_name="Salesforce/blip-image-captioning-base",  # Smaller model for faster processing
        dtype_str="float16",  # Use half precision for memory efficiency
        output_dir="/tmp",  # Will be overridden by caption_frames method
    )
    
    datasets = [
        ("SumMe", summe_frames_dir),
        ("TVSum", tvsum_frames_dir)
    ]
    
    for dataset_name, frames_dir in datasets:
        if os.path.exists(frames_dir):
            print(f"\nProcessing {dataset_name} dataset...")
            try:
                captioner.caption_frames(
                    frames_dir=frames_dir,
                    frame_interval=1,  # Caption every 15th frame
                    batch_size=1024,  # Override batch size for captioning
                    resume=True,  # Skip already processed videos
                    pathname="*.json"
                )
                print(f"{dataset_name} processing completed!")
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
        else:
            print(f"Warning: {dataset_name} frames directory not found: {frames_dir}")
    
    print("\nAll tasks completed!")
    print("Caption files saved in:")
    print("  SumMe: ~/autodl-tmp/data/SumMe/captions/Blip2/")
    print("  TVSum: ~/autodl-tmp/data/TVSum/captions/Blip2/")


if __name__ == "__main__":
    main()