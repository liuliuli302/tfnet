#!/usr/bin/env python3
"""
Video Summarization Pipeline Main Entry Point
视频摘要流程主入口点
"""

from src.pipeline.sum_pipeline01 import VideoSummaryPipeline
import sys
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# TensorBoard相关导入
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def load_config(config_path: str) -> dict:
    """从YAML文件加载配置"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def create_tensorboard_logger(logging_config: dict) -> TensorBoardLogger:
    """创建TensorBoard logger"""
    if not logging_config.get("enabled", True):
        return None
    
    experiment_name = logging_config.get("experiment_name", None)
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"video_summary_step1_{timestamp}"
    
    tb_logger = TensorBoardLogger(
        save_dir=logging_config.get("log_dir", "./tensorboard_logs"),
        name=experiment_name,
        version=logging_config.get("version", None),
        default_hp_metric=False
    )
    
    return tb_logger


def main():
    """主函数：运行视频摘要流程"""
    parser = argparse.ArgumentParser(description="Video Summarization Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to YAML configuration file (default: config.yaml)")
    args = parser.parse_args()

    config = load_config(args.config)
    logging_config = config.get("logging", {})
    tb_logger = create_tensorboard_logger(logging_config)
    
    pipeline = VideoSummaryPipeline(
        dataset_paths=config["dataset_paths"],
        frame_context_window=config["frame_context_window"],
        frame_interval=config["frame_interval"],
        output_dir=config["output"],
        model_name=config["model_name"],
        device_map=config["device_map"],
        importance_prompt=config["importance_prompt"],
        logger=tb_logger,
        skip_step1=config.get("skip_step1", False),
        skip_step2=config.get("skip_step2", False),
        batch_size=config.get("batch_size", 32),
    )

    results = pipeline.run()
    return True


if __name__ == "__main__":
    main()
