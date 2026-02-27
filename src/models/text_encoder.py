from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

# 1. 加载模型 (会自动从 HuggingFace 下载)
# 如果本地有显存压力，可以添加 device='cuda' 或 'cpu'
model = SentenceTransformer('Qwen/Qwen3-Embedding-8B', trust_remote_code=True)


def calculate_similarity(text1, text2):
    # 2. 将文本转换为向量 (Embeddings)
    # Qwen3 建议在检索任务中给 Query 加上特定指令，但纯匹配可直接输入
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # 3. 计算余弦相似度
    # cosine_similarity = (A · B) / (||A|| * ||B||)
    sim = F.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0))

    return sim.item()


# --- 测试示例 ---
text_a = "人工智能正在改变医疗诊断的准确率。"
text_b = "AI技术在医学影像分析中显著提升了判断精度。"
text_c = "今天晚饭我想吃红烧肉。"

score_ab = calculate_similarity(text_a, text_b)
score_ac = calculate_similarity(text_a, text_c)

print(f"相关文本匹配度: {score_ab:.4f}")  # 预期：数值较高（如 0.8+）
print(f"无关文本匹配度: {score_ac:.4f}")  # 预期：数值较低
