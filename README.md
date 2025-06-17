resource文件夹中有两个结果分数文件。首先设定一个可变参数a：[0, 0.1, 0.2, ... 1]
将这两个结果文件对应的分数以a求加权和。之后将这个分数进行评测。
然后找出两个数据集最高的f1分数。（两个数据集的a可以不一样）
=== step1_results.json 详细结构 ===
顶层键:
  - step
  - description
  - parameters
  - datasets

parameters 键下的内容:
  - frame_context_window
  - frame_interval

datasets 键下的内容:
  - SumMe
    数据集属性:
      - dataset_name
      - dataset_path
      - total_videos
      - videos
        视频示例 (Air_Force_One) 的属性:
          - video_name
          - total_frames
          - sampled_frames
          - sampled_indices
          - scores
          - frame_interval
          - context_window

=== step2_similarity_scores.json 详细结构 ===
顶层键:
  - step
  - description
  - parameters
  - timestamp
  - datasets

parameters 键下的内容:
  - frame_interval
  - batch_size
  - model_name
  - model_type
  - device

datasets 键下的内容:
  - SumMe
    数据集属性:
      - dataset_name
      - dataset_path
      - total_videos
      - videos
        视频示例 (Air_Force_One) 的属性:
          - video_name
          - total_frames
          - sampled_frames
          - sampled_indices
          - query_text
          - similarities
# 0 视频摘要框架的基础支持
## 0.1 视频帧提取 ✅
将视频帧提取到输出文件夹

关键参数：

video_dir：视频文件夹

out_dir：输出文件夹

输出：

视频帧文件

### 具体实现 01

简单的帧提取实现

提取到的路径：frames/

代码路径：src/util/video_utils.py

## 0.2 视频帧特征提取

将视频帧的特征提取到指定文件夹下

关键参数：

model_name：采用的模型

frames_dir：帧文件夹

输出：

视频特征文件

### 具体实现 01

利用Blip2提取经过vision_model，q_former，lang_proj层后的视觉特征

代码路径：src/util/feature_extractor.py

### 具体实现 02

纯粹复用文献TFVTG的代码

提取到的路径：features/visual

形状：(N, 32, 256)

代码路径：src/util/0_2_2_feature_extraction.py

## 0.3 视频帧Caption

生成视频帧字幕

关键参数：

model_name：采用的模型

frames_dir：帧文件夹

输出：

帧级别的字幕

### 具体实现 01

利用Blip2模型提取帧级别字幕

提取到的路径：captions/Blip2

代码路径：src/util/image_captioner.py

## 0.4 视频级别Caption/Summarizaiton

生成视频级别的文本摘要

关键参数：

video_dir：视频文件夹

输出：

视频级别的文本Caption/Summarization

### 技术路径 1

将视频+摘要所需prompt直接送入Video-LLM，生成视频Caption/Summarization

### 技术路径 2

以视频帧Caption为原始输出，利用LLM实现摘要/聚合

实现：TODO

### 具体实现 01

采用技术路径 1，利用Qwen2.5VL多模态大模型提取视频的Caption，采样帧数8，均匀采样

提取到的路径：captions/Blip2

代码路径：src/util/video_captioner.py

## 0.5 文本特征提取

提取文本的特征

关键参数：

model_name：采用的模型

text_dir：帧Caption/视频Caption等文本信息存在位置

输出：

文本特征文件

### 具体实现 01

自己编写的利用Blip2模型提取文本特征

提取路径：text_feature

代码路径：src/util/feature_extractor.py

### 具体实现 02

纯粹复用的TFVTG的特征提取方法

提取路径：features/text

代码路径：src/util/0_5_2_text_feature_extraction.py

# 1 预处理

## 数据清洗

## 噪声去除

## 冗余去除





