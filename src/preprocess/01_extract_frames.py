#!/usr/bin/env python3
"""
Frame extraction script using utility functions.
Usage: python 01_extract_frames.py
"""

import os
import sys

# Add the parent directory to sys.path to import util modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.video_utils import extract_frames_from_directory

def main():
    """Main function to extract frames from all videos in a directory."""
    extract_frames_from_directory(
        video_dir='/root/autodl-tmp/data/SumMe/videos',
        output_folder='/root/autodl-tmp/data/SumMe/frames',
        frame_interval=1
    )
    
    extract_frames_from_directory(
        video_dir='/root/autodl-tmp/data/TVSum/videos',
        output_folder='/root/autodl-tmp/data/TVSum/frames',
        frame_interval=1
    )
    

if __name__ == '__main__':
    main()