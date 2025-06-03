#!/usr/bin/env python3
"""
TensorBoard Integration Test
测试TensorBoard集成功能的简单脚本
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_tensorboard_integration():
    """测试TensorBoard集成功能"""
    print("Testing TensorBoard Integration...")
    
    try:
        from src.pipeline.sum_pipeline01 import VideoSummaryPipeline
        print("✅ Successfully imported VideoSummaryPipeline")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # 创建临时目录用于测试
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建模拟的数据集结构
        dataset_path = temp_path / "test_dataset"
        frames_dir = dataset_path / "frames" / "video1"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建一些模拟的帧文件（空文件）
        for i in range(5):
            frame_file = frames_dir / f"frame_{i:06d}.jpg"
            frame_file.touch()
        
        log_dir = temp_path / "test_logs"
        
        try:
            # 创建TensorBoard logger
            from torch.utils.tensorboard import SummaryWriter
            tb_logger = SummaryWriter(log_dir=str(log_dir / "test_experiment"))
            
            # 测试pipeline初始化（仅测试TensorBoard相关功能）
            pipeline = VideoSummaryPipeline(
                dataset_paths=[str(dataset_path)],
                frame_context_window=2,
                frame_interval=1,
                output_dir=str(temp_path / "results"),
                logger=tb_logger
            )
            
            print("✅ Pipeline initialization successful")
            print(f"✅ TensorBoard logger provided: {pipeline.tb_writer is not None}")
            print(f"✅ Log directory: {pipeline.log_dir}")
            
            # 测试日志记录功能
            if pipeline.tb_writer:
                # 记录一些测试数据
                pipeline.log_hyperparameters()
                pipeline.log_video_scores("test_video", [0.1, 0.5, 0.8, 0.3])
                pipeline.log_dataset_summary("test_dataset", {
                    "test_video": {"total_frames": 10, "sampled_frames": 4, "scores": [0.1, 0.5, 0.8, 0.3]}
                })
                print("✅ TensorBoard logging functions work correctly")
            
            # 清理资源
            pipeline.close_logger()
            print("✅ Logger cleanup successful")
            
            # 检查是否生成了TensorBoard文件
            log_files = list(log_dir.rglob("events.out.tfevents.*"))
            if log_files:
                print(f"✅ TensorBoard event files created: {len(log_files)} files")
            else:
                print("⚠️  No TensorBoard event files found")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline test failed: {e}")
            return False

def test_config_loading():
    """测试配置文件加载"""
    print("\nTesting config loading...")
    
    try:
        import yaml
        from main import load_config
        
        config_path = "config/config.yaml"
        if not Path(config_path).exists():
            print(f"❌ Config file not found: {config_path}")
            return False
        
        config = load_config(config_path)
        
        # 检查TensorBoard相关配置
        if "logging" in config:
            logging_config = config["logging"]
            print(f"✅ Logging config found: {logging_config}")
            
            required_keys = ["enabled", "log_dir", "experiment_name", "comment"]
            for key in required_keys:
                if key in logging_config:
                    print(f"✅ {key}: {logging_config[key]}")
                else:
                    print(f"❌ Missing key: {key}")
                    return False
        else:
            print("❌ No logging configuration found in config file")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TensorBoard Integration Test Suite")
    print("=" * 60)
    
    # 测试配置加载
    config_test = test_config_loading()
    
    # 测试TensorBoard集成
    tensorboard_test = test_tensorboard_integration()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Config Loading: {'✅ PASS' if config_test else '❌ FAIL'}")
    print(f"TensorBoard Integration: {'✅ PASS' if tensorboard_test else '❌ FAIL'}")
    
    if config_test and tensorboard_test:
        print("\n🎉 All tests passed! TensorBoard integration is ready to use.")
        print("\nTo start using:")
        print("1. Run: python main.py --config config/config.yaml")
        print("2. Start TensorBoard: tensorboard --logdir=./tensorboard_logs")
        print("3. Open browser: http://localhost:6006")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    print("=" * 60)
