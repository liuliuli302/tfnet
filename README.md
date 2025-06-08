# 0 视频摘要框架的基础支持
## 0.1 视频帧提取
将视频帧提取到输出文件夹

关键参数：

video_dir：视频文件夹

out_dir：输出文件夹

输出：

视频帧文件

### 具体实现 01

简单的帧提取实现

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

## 0.3 视频帧Caption

生成视频帧字幕

关键参数：

model_name：采用的模型

frames_dir：帧文件夹

输出：

帧级别的字幕

### 具体实现 01

利用Blip2模型提取帧级别字幕

代码路径：src/util/image_captioner.py

## 0.4 视频级别Caption/Summarizaiton

生成视频级别的文本摘要

关键参数：

video_dir：视频文件夹

输出：

视频级别的文本Caption/Summarization

### 技术路径 1

将视频+摘要所需prompt直接送入Video-LLM，生成视频Caption/Summarization

实现：TODO

### 技术路径 2

以视频帧Caption为原始输出，利用LLM实现摘要/聚合

实现：TODO

### 具体实现 01

采用技术路径 1，利用Qwen2.5VL多模态大模型提取视频的Caption，采样帧数8，均匀采样

代码路径：src/util/video_captioner.py

## 0.5 文本特征提取

提取文本的特征

关键参数：

model_name：采用的模型

text_dir：帧Caption/视频Caption等文本信息存在位置

输出：

文本特征文件

### 具体实现 01

利用Blip2模型提取文本特征

代码路径：src/util/feature_extractor.py

# 1 预处理

## 数据清洗

## 噪声去除

## 冗余去除

