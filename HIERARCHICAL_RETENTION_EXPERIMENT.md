# 层级语义聚合信息保留率评估实验

## 1. 实验目标

定量评估论文 3.2.2 节层级语义聚合过程中的信息保留情况：

| 聚合层级      | Reference（源文本）    | Candidate（聚合文本） | 含义                         |
| ------------- | ---------------------- | --------------------- | ---------------------------- |
| **帧→场景**   | 场景内所有帧描述拼接   | 场景级聚合摘要        | 场景聚合保留了多少帧级细节   |
| **场景→视频** | 视频内所有场景摘要拼接 | 视频级全局摘要        | 视频聚合保留了多少场景级信息 |

### 评价指标

| 指标            | 全称             | 测度                                 | 出处                |
| --------------- | ---------------- | ------------------------------------ | ------------------- |
| **ROUGE-1 R**   | ROUGE-1 Recall   | unigram 词面保留率                   | Lin (2004)          |
| **BERTScore R** | BERTScore Recall | 语义级保留率（contextual embedding） | Zhang et al. (2020) |

---

## 2. 环境依赖

```bash
pip install rouge-score bert-score torch numpy
```

版本参考：
- `rouge-score >= 0.1.2`
- `bert-score >= 0.3.13`
- `torch`（BERTScore 后端依赖）

---

## 3. 数据源

实验**无需重新运行 VLM/LLM**，直接使用已存储的 caption 文件。

### 3.1 文件路径

```
Codes/tfnet/data/captions/
├── frame_caption/llava/
│   ├── tvsum_frame_captions.json      # 帧级描述
│   └── summe_frame_captions.json
├── scene_caption/gpt5/
│   ├── tvsum_scene_captions.json      # 场景级摘要
│   └── summe_scene_captions.json
└── video_caption/gpt5/
    ├── tvsum_video_captions.json       # 视频级摘要
    └── summe_video_captions.json
```

### 3.2 数据格式

**帧级描述** (`tvsum_frame_captions.json`)：
```json
{
  "video_name": {
    "picks": [0, 15, 30, ...],
    "frame_caption": ["A group of people walk...", "A skateboarder...", ...]
  }
}
```

**场景级摘要** (`tvsum_scene_captions.json`)：
```json
{
  "video_name": {
    "0": {
      "scene_idx": 0,
      "picks": [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150],
      "scene_caption": "On a palm-lined street, a group of people walk...",
      "change_points": [0, 160]
    },
    "1": { ... }
  }
}
```

**视频级摘要** (`tvsum_video_captions.json`)：
```json
{
  "video_name": "The video follows a long, loosely organized procession..."
}
```

---

## 4. 计算流程

### 4.1 帧→场景 (Frame → Scene)

对每个视频的**每个场景**逐一计算：

```
输入：
  - reference_text = 该场景内所有帧描述按 pick 顺序拼接
  - candidate_text = 该场景的 scene_caption

处理：
  1. 从 scene_captions.json 遍历每个 video → 每个 scene
  2. 用 scene["picks"] 从 frame_captions.json 定位对应帧描述
  3. 按 picks 顺序拼接帧描述作为 reference
  4. 用 scene["scene_caption"] 作为 candidate
  5. 计算 ROUGE-1 Recall 和 BERTScore Recall
  6. 记录 scene_idx, num_frames, reference_length, candidate_length

聚合：
  - 按视频平均 → per-video scene-level retention
  - 按数据集平均 → overall frame→scene retention
```

### 4.2 场景→视频 (Scene → Video)

对每个视频计算：

```
输入：
  - reference_text = 该视频所有场景摘要按 scene_idx 顺序拼接
  - candidate_text = 该视频的 video_caption

处理：
  1. 从 scene_captions.json 遍历每个 video
  2. 按 scene_idx 升序拼接所有 scene_caption 作为 reference
  3. 从 video_captions.json 取 video_caption 作为 candidate
  4. 计算 ROUGE-1 Recall 和 BERTScore Recall
  5. 记录 num_scenes, reference_length, candidate_length

聚合：
  - 按数据集平均 → overall scene→video retention
```

---

## 5. 指标计算细节

### 5.1 ROUGE-1 Recall

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
scores = scorer.score(reference_text, candidate_text)
rouge1_recall = scores['rouge1'].recall  # float, [0, 1]
```

- `use_stemmer=True`：启用 Porter Stemmer，使 "walking" 和 "walks" 匹配
- ROUGE-1 Recall = (reference 与 candidate 共有的 unigram 数) / (reference 的 unigram 总数)
- **直觉**：reference 里的词有多少在 candidate 中保留

### 5.2 BERTScore Recall

```python
from bert_score import score

P, R, F1 = score(
    [candidate_text], 
    [reference_text], 
    lang='en',
    model_type='microsoft/deberta-xlarge-mnli',  # 推荐，与原始论文一致
    verbose=False
)
bertscore_recall = R[0].item()  # float, [0, 1]
```

- 使用 `microsoft/deberta-xlarge-mnli`（BERTScore 官方推荐模型，比 bert-base 更准确）
- 如果显存不足，可降级为 `bert-base-uncased`
- BERTScore Recall = candidate 中每个 token 与 reference 中最相似 token 的余弦相似度均值
- **直觉**：candidate 的语义内容有多少在 reference 中有对应（同义词也能匹配）

### 5.3 重要注意事项

1. **文本预处理**：不需要额外清洗。ROUGE scorer 内置 tokenization；BERTScore 使用模型自带 tokenizer。
2. **空文本处理**：如果 candidate 或 reference 为空字符串，跳过该条记录并输出 warning。
3. **BERTScore 输出精度**：保留 4 位小数。
4. **长度记录**：同时记录 reference 和 candidate 的字符数/词数，便于计算压缩率。

---

## 6. 期望输出

### 6.1 输出目录

```
Codes/tfnet/figure/results/hierarchical_retention/
```

### 6.2 输出文件

#### (a) `tvsum_frame_to_scene.json` / `summe_frame_to_scene.json`

```json
{
  "dataset": "tvsum",
  "level": "frame_to_scene",
  "metrics": {
    "rouge1_recall": {
      "mean": 0.723,
      "std": 0.142,
      "median": 0.751,
      "min": 0.312,
      "max": 0.945
    },
    "bertscore_recall": {
      "mean": 0.851,
      "std": 0.098,
      "median": 0.867,
      "min": 0.523,
      "max": 0.972
    },
    "compression_ratio": {
      "mean": 0.234,
      "description": "candidate_length / reference_length (字符数)"
    }
  },
  "per_scene": [
    {
      "video_name": "z_6gVvQb2d0",
      "scene_idx": 0,
      "num_frames": 11,
      "reference_length": 1523,
      "candidate_length": 367,
      "rouge1_recall": 0.451,
      "bertscore_recall": 0.723
    },
    ...
  ],
  "per_video": [
    {
      "video_name": "z_6gVvQb2d0",
      "num_scenes": 5,
      "rouge1_recall_mean": 0.512,
      "bertscore_recall_mean": 0.756
    },
    ...
  ]
}
```

#### (b) `tvsum_scene_to_video.json` / `summe_scene_to_video.json`

```json
{
  "dataset": "tvsum",
  "level": "scene_to_video",
  "metrics": {
    "rouge1_recall": {
      "mean": 0.548,
      "std": 0.167,
      "median": 0.569,
      "min": 0.201,
      "max": 0.821
    },
    "bertscore_recall": {
      "mean": 0.792,
      "std": 0.113,
      "median": 0.808,
      "min": 0.412,
      "max": 0.938
    },
    "compression_ratio": {
      "mean": 0.312,
      "description": "candidate_length / reference_length (字符数)"
    }
  },
  "per_video": [
    {
      "video_name": "z_6gVvQb2d0",
      "num_scenes": 5,
      "reference_length": 2847,
      "candidate_length": 912,
      "rouge1_recall": 0.548,
      "bertscore_recall": 0.792
    },
    ...
  ]
}
```

#### (c) `summary_table.csv`

论文表格的 CSV 格式（可直接导入 LaTeX）：

```csv
Level,Dataset,ROUGE-1 R (μ±σ),BERTScore R (μ±σ),Compression Ratio
Frame→Scene,SumMe,0.xxx±0.xxx,0.xxx±0.xxx,0.xxx
Frame→Scene,TVSum,0.xxx±0.xxx,0.xxx±0.xxx,0.xxx
Scene→Video,SumMe,0.xxx±0.xxx,0.xxx±0.xxx,0.xxx
Scene→Video,TVSum,0.xxx±0.xxx,0.xxx±0.xxx,0.xxx
```

---

## 7. 实现要求

### 7.1 脚本位置

```
Codes/tfnet/figure/evaluate_hierarchical_retention.py
```

### 7.2 命令行接口

```bash
python figure/evaluate_hierarchical_retention.py \
    --dataset tvsum \          # tvsum | summe | all（默认 all）
    --level frame_to_scene \   # frame_to_scene | scene_to_video | all（默认 all）
    --bertscore-model microsoft/deberta-xlarge-mnli \  # 可选，默认值
    --output-dir figure/results/hierarchical_retention  # 可选，默认值
```

### 7.3 代码结构建议

```
class HierarchicalRetentionEvaluator:
    def __init__(self, data_root, output_dir, bertscore_model):
        ...
    
    def load_data(self, dataset: str):
        """加载三个层级的 caption JSON"""
        ...
    
    def evaluate_frame_to_scene(self, dataset: str) -> dict:
        """帧→场景 保留率"""
        ...
    
    def evaluate_scene_to_video(self, dataset: str) -> dict:
        """场景→视频 保留率"""
        ...
    
    def compute_metrics(self, reference: str, candidate: str) -> dict:
        """计算 ROUGE-1 R 和 BERTScore R"""
        ...
    
    def aggregate_and_save(self, results, dataset, level):
        """汇总统计并写 JSON/CSV"""
        ...
```

### 7.4 关键实现注意点

1. **BERTScore 批处理**：`bert_score.score()` 支持一次传入多个 candidate-reference 对，应尽可能批量计算（建议 batch size = 64），大幅加速。
2. **GPU 利用**：BERTScore 的 DeBERTa 模型约 1.6GB 显存，建议在 GPU 上运行；CPU 亦可但较慢。
3. **进度条**：使用 `tqdm` 显示处理进度。
4. **错误处理**：对空文本、缺失 key 等情况用 try-except 包裹，记录到 stderr。
5. **可复现性**：在输出 JSON 中记录 `bertscore_model` 和 `rouge_library_version`。

---

## 8. 预期结果参考范围

基于类似视频摘要层级聚合任务的文献经验：

| Level     | ROUGE-1 R 预期 | BERTScore R 预期 | 说明                           |
| --------- | -------------- | ---------------- | ------------------------------ |
| 帧→场景   | 0.55–0.75      | 0.78–0.90        | BERTScore > ROUGE 因为同义聚合 |
| 场景→视频 | 0.40–0.60      | 0.70–0.85        | 视频级压缩率更高，保留率下降   |

如果 BERTScore R 显著高于 ROUGE-1 R（差值 > 0.15），说明 LLM 进行了大量同义改写（paraphrasing）而非简单删除——这正是"语义聚合"而非"机械截断"的证据，对论文论证有利。

---

## 9. 论文写作要点

结果写入 3.2.2 节时，可配以下分析文字（模板）：

> 为定量评估层级语义聚合中的信息保留率，我们分别计算了帧→场景和场景→视频两个层级的 ROUGE-1 Recall 和 BERTScore Recall。ROUGE-1 Recall 测量 unigram 级别的词面重叠，反映表层信息的保留比率；BERTScore Recall 基于上下文语义嵌入的余弦相似度，能够容忍同义词替换，更适合评估语义级保留。
>
> 如表 X 所示，帧→场景聚合在 TVSum 上 ROUGE-1 R 为 X.XX，BERTScore R 为 X.XX，SumMe 上分别为 X.XX 和 X.XX。BERTScore R 显著高于 ROUGE-1 R（差值约 X.XX），说明 LLM 在聚合中进行了大量语义层面的重述而非简单删除，验证了层级聚合机制的有效性。场景→视频层级的保留率适度下降（BERTScore R 约为 X.XX），符合从细粒度到粗粒度摘要的预期信息压缩行为。
