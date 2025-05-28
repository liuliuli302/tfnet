#!/usr/bin/env python3
"""
Video captioning script using utility functions.
Usage: python 02_caption_videos.py
"""

import os
import sys

# Add the parent directory to sys.path to import util modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from util.captioner import caption_videos

def main():
    """Main function to caption videos in a directory."""
    # Caption SumMe with GPT-4o
    caption_videos(
        input_path='/root/autodl-tmp/data/SumMe/videos',
        model_name='llavavideo',
        output_dir='/root/autodl-tmp/data/SumMe/captions',
        # api_key='YOUR_OPENAI'
    )
    
    # Caption TVSum with GPT-4o
    caption_videos(
        input_path='/root/autodl-tmp/data/SumMe/videos',
        model_name='llavavideo',
        output_dir='/root/autodl-tmp/data/SumMe/captions',
        # api_key='YOUR_OPENAI'
    )

if __name__ == '__main__':
    main()