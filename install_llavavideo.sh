#!/bin/bash
# Installation script for LLaVA-Video dependencies

echo "Installing LLaVA-Video dependencies..."

# Install basic requirements
pip install torch>=2.0.0 transformers>=4.30.0 accelerate>=0.20.0 decord>=0.6.0

# Install LLaVA-Video from GitHub
echo "Installing LLaVA-Video from GitHub..."
pip install git+https://github.com/PKU-YuanGroup/LLaVA-Video.git

echo "LLaVA-Video installation complete!"
echo ""
echo "Note: Make sure you have CUDA available if you want to use GPU acceleration."
echo "You may also need to download model weights separately."
echo ""
echo "Example usage:"
echo "python src/preprocess/02_caption_videos.py"
