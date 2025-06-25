# 1 自相似度矩阵
对于一个视频帧特征序列（300，2580）

构建一个自相似度计算矩阵

计算两两帧之间的相似度

如帧A与其他所有帧都计算余弦相似度
这样，得到一个按照帧顺序的相似度分数序列

如何操作这种方法
以达到视频视频中相似内容很多的冗余帧

定义一种冗余分数，如果某帧的冗余分数比较高
则说明该帧附近具有很多与他具有较高语义相似度的帧


具体的实现方案如下


示例输入
视频帧特征序列形状为 (300, 2580)，即 300 帧，每帧 2580 维的特征向量（比如是提取的ViT、I3D等模型输出）。

## 第一步：计算自相似度矩阵
构建一个 (300, 300) 的余弦相似度矩阵 S，其中：
$$
S_{i,j} = \text{cosine\_similarity}(f_i, f_j) = \frac{f_i \cdot f_j}{\|f_i\| \cdot \|f_j\|}
$$

这表示帧 i 和帧 j 的语义相似度。

## 第二步：定义冗余分数（Redundancy Score）
对于第 
𝑖
i 帧，我们定义冗余分数为它与附近帧的平均相似度（排除自身）：

$$
R_i = \frac{1}{2w} \sum_{\substack{j=i-w \\ j \ne i}}^{i+w} S_{i,j}
$$

w 为窗口大小，控制“附近”的帧范围，建议设置为 w=10。

可以对 R_i 做归一化处理，用于后续筛选冗余帧。

## 作用

高冗余分数：帧在当前时间段有很多相似帧（语义冗余）。

低冗余分数：帧较为独特，可保留。

你可以用该分数作为视频摘要或压缩中的帧筛选依据。

示例实现代码

```
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_self_similarity_matrix(features):
    # features: (num_frames, feature_dim)
    return cosine_similarity(features)  # returns (num_frames, num_frames)

def compute_redundancy_score(sim_matrix, window_size=10):
    num_frames = sim_matrix.shape[0]
    redundancy_scores = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = max(0, i - window_size)
        end = min(num_frames, i + window_size + 1)
        # Exclude the frame itself
        indices = [j for j in range(start, end) if j != i]
        redundancy_scores[i] = sim_matrix[i, indices].mean()
    
    return redundancy_scores

# 示例
features = np.random.rand(300, 2580)  # 你的特征输入
sim_matrix = compute_self_similarity_matrix(features)
redundancy_scores = compute_redundancy_score(sim_matrix, window_size=10)

# 可视化或使用该分数选取非冗余帧
import matplotlib.pyplot as plt
plt.plot(redundancy_scores)
plt.title("Frame Redundancy Score")
plt.xlabel("Frame Index")
plt.ylabel("Redundancy")
plt.show()
```


# 2 文本自相似度矩阵
和视觉的差不多

不过模态换成了文本版

这可能就需要你先提取每一帧的caption
之后再提取特征

之后再计算相似度


