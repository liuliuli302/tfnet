#!/usr/bin/env python3
"""
Test script for simplified pipeline with TensorBoardLogger
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from pytorch_lightning.loggers import TensorBoardLogger
from src.pipeline.sum_pipeline01 import VideoSummaryPipeline

def test_tensorboard_logger():
    """Test TensorBoardLogger integration"""
    # Create TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="./test_logs",
        name="test_experiment",
        version="v1"
    )
    
    print(f"TensorBoard log directory: {tb_logger.log_dir}")
    
    # Test logging methods
    tb_logger.log_hyperparams({
        "frame_context_window": 3,
        "frame_interval": 15,
        "dataset_count": 1
    })
    
    tb_logger.log_metrics({
        "test_metric": 0.5,
        "another_metric": 0.8
    })
    
    # Finalize logger
    tb_logger.finalize("success")
    
    print("TensorBoard logger test completed successfully!")

def test_simplified_pipeline():
    """Test simplified pipeline without actual video data"""
    tb_logger = TensorBoardLogger(
        save_dir="./test_logs",
        name="pipeline_test",
        version="v1"
    )
    
    # Create pipeline with mock dataset paths
    pipeline = VideoSummaryPipeline(
        dataset_paths=["/nonexistent/path"],  # This will be handled gracefully
        frame_context_window=3,
        frame_interval=15,
        output_dir="./test_results",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        device_map="auto",
        logger=tb_logger
    )
    
    # Test that pipeline initializes correctly
    assert pipeline.tb_writer is not None
    assert pipeline.log_dir is not None
    assert pipeline.output_dir.exists()
    
    print("Simplified pipeline test completed successfully!")

if __name__ == "__main__":
    print("Testing TensorBoardLogger integration...")
    test_tensorboard_logger()
    
    print("\nTesting simplified pipeline...")
    test_simplified_pipeline()
    
    print("\nAll tests passed!")
