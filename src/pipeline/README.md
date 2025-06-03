# 视频摘要流程 - 第一步实现

## 概述

本模块实现了免训练的视频摘要流程的第一步：通过多模态大模型查询视频帧的重要性分数。

## 功能特性

- **帧采样**: 按照指定的帧跳数对视频帧进行采样
- **上下文窗口**: 为每个查询帧提供上下文帧信息
- **重要性评分**: 使用多模态大模型为每帧评估0-1之间的重要性分数
- **批量处理**: 支持同时处理多个数据集
- **结果保存**: 将评分结果保存为JSON格式

## 核心参数

### VideoSummaryPipeline 类参数

- **dataset_paths**: 数据集路径列表，每个路径应包含`frames`文件夹
- **frame_context_window**: 帧上下文窗口长度N（默认：3）
- **frame_interval**: 帧跳数M，表示每隔M帧采样一次（默认：15）
- **output_dir**: 输出结果的目录（默认：`./results`）
- **model_name**: 多模态模型名称（默认：`Qwen/Qwen2.5-VL-7B-Instruct`）
- **device_map**: 设备映射（默认：`auto`）
- **importance_prompt**: 自定义重要性查询提示词

## 使用方法

### 1. 基本使用

```python
from src.pipeline.sum_pipeline01 import VideoSummaryPipeline

# 配置数据集路径
dataset_paths = [
    "/path/to/your/dataset1",
    "/path/to/your/dataset2"
]

# 创建流程实例
pipeline = VideoSummaryPipeline(
    dataset_paths=dataset_paths,
    frame_context_window=3,
    frame_interval=15,
    output_dir="./results"
)

# 运行流程
results = pipeline.run()
```

### 2. 自定义提示词

```python
custom_prompt = """
请评估这一帧在视频摘要中的重要性程度。
只需要回答一个0到1之间的数字分数。
"""

pipeline = VideoSummaryPipeline(
    dataset_paths=dataset_paths,
    importance_prompt=custom_prompt
)
```

### 3. 运行示例脚本

```bash
cd /root/tfnet
python example_run_pipeline.py
```

## 数据集结构要求

您的数据集应该具有以下结构：

```
dataset_name/
├── frames/
│   ├── video1_name/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   ├── video2_name/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── ...
```

## 采样逻辑

### 帧采样示例
- 帧跳数为15时，采样帧序列为：[0, 15, 30, 45, ...]
- 如果视频总帧数为100，最终採样帧为：[0, 15, 30, 45, 60, 75, 90]

### 上下文窗口示例
假设帧上下文窗口长度为3，帧跳数为15：

- 当前帧为0时，输入到模型的帧为：[0, 15]（如果第15帧存在）
- 当前帧为15时，输入模型的帧为：[0, 15, 30]
- 当前帧为采样序列最后一帧时，输入模型的帧为：[倒数第二帧, 最后帧]

## 输出格式

结果将保存为JSON文件，格式如下：

```json
{
  "step": 1,
  "description": "Query multimodal model for frame importance scores",
  "parameters": {
    "frame_context_window": 3,
    "frame_interval": 15,
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct"
  },
  "datasets": {
    "dataset_name": {
      "dataset_name": "dataset_name",
      "dataset_path": "/path/to/dataset",
      "total_videos": 10,
      "videos": {
        "video_name": {
          "video_name": "video_name",
          "total_frames": 300,
          "sampled_frames": 20,
          "sampled_indices": [0, 15, 30, 45, ...],
          "scores": [0.8, 0.3, 0.9, 0.1, ...],
          "frame_interval": 15,
          "context_window": 3
        }
      }
    }
  }
}
```

## 依赖要求

请确保安装了以下依赖：

```bash
pip install torch transformers accelerate
pip install opencv-python pillow numpy tqdm
pip install qwen-vl-utils  # 如果需要
```

## 注意事项

1. **GPU内存**: 多模态大模型需要较大的GPU内存，建议使用至少8GB显存的GPU
2. **处理时间**: 每个视频的处理时间取决于采样帧数和模型推理速度
3. **错误处理**: 如果模型返回无法解析的分数，系统会默认分配0.5（中等重要性）
4. **路径配置**: 请确保数据集路径正确，且包含预期的frames文件夹结构

## 后续步骤

这是视频摘要流程的第一步，后续步骤将在接下来的开发中实现。每个步骤的结果将作为下一步的输入。
