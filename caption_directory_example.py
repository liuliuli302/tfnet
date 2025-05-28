#!/usr/bin/env python3
"""
Example script showing how to use the caption_videos_from_directory method.
Usage: python caption_directory_example.py
"""

import os
import sys

# Add the src directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from util.captioner import GPT4oCaptioner, Gemini25ProCaptioner, LLaVAVideoCaptioner

def main():
    """Demonstrate using caption_videos_from_directory method."""
    
    # Configuration
    video_dir = "data/videos"  # Directory containing videos
    output_base = "out/captions_batch"  # Base output directory
    
    # Example 1: Using GPT-4o (requires OPENAI_API_KEY environment variable)
    print("=== GPT-4o Batch Captioning ===")
    try:
        gpt_captioner = GPT4oCaptioner()
        gpt_output_dir = os.path.join(output_base, "gpt4o")
        
        results = gpt_captioner.caption_videos_from_directory(video_dir, gpt_output_dir)
        
        if results:
            successful = [r for r in results if r["status"] == "success"]
            errors = [r for r in results if r["status"] == "error"]
            print(f"GPT-4o: {len(successful)} successful, {len(errors)} errors")
        
    except Exception as e:
        print(f"GPT-4o captioner failed to initialize: {e}")
    
    # Example 2: Using Gemini (requires GEMINI_API_KEY environment variable)
    print("\n=== Gemini Batch Captioning ===")
    try:
        gemini_captioner = Gemini25ProCaptioner()
        gemini_output_dir = os.path.join(output_base, "gemini")
        
        results = gemini_captioner.caption_videos_from_directory(video_dir, gemini_output_dir)
        
        if results:
            successful = [r for r in results if r["status"] == "success"]
            errors = [r for r in results if r["status"] == "error"]
            print(f"Gemini: {len(successful)} successful, {len(errors)} errors")
        
    except Exception as e:
        print(f"Gemini captioner failed to initialize: {e}")
    
    # Example 3: Using LLaVA (simplified implementation)
    print("\n=== LLaVA Batch Captioning ===")
    try:
        llava_captioner = LLaVAVideoCaptioner()
        llava_output_dir = os.path.join(output_base, "llava")
        
        results = llava_captioner.caption_videos_from_directory(video_dir, llava_output_dir)
        
        if results:
            successful = [r for r in results if r["status"] == "success"]
            errors = [r for r in results if r["status"] == "error"]
            print(f"LLaVA: {len(successful)} successful, {len(errors)} errors")
        
    except Exception as e:
        print(f"LLaVA captioner failed to initialize: {e}")

    print(f"\nAll captions saved to: {output_base}")
    print("Directory structure:")
    print(f"  {output_base}/")
    print(f"    ├── gpt4o/          # GPT-4o captions")
    print(f"    ├── gemini25pro/    # Gemini captions") 
    print(f"    └── llavavideo/     # LLaVA captions")
    print("Each subdirectory contains .txt files named after the original video files.")

if __name__ == '__main__':
    main()
