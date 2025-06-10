import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from lavis.models import load_model_and_preprocess

os.environ['XDG_CACHE_HOME'] = '/root/autodl-tmp/.cache'

model, _, _ = load_model_and_preprocess("blip2_image_text_matching", "coco", device='cuda', is_eval=True)

def calc_scores_with_custom_text_compare(visual_features, sentences, model):
    with torch.no_grad():
        if isinstance(sentences, str):
            sentences = [sentences]
        
        text = model.tokenizer(sentences, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to('cuda')
        text_output = model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
        text_feat = model.text_proj(text_output.last_hidden_state[:,0,:])
        
        v1 = F.normalize(text_feat, dim=-1)
        if isinstance(visual_features, torch.Tensor):
            v2 = F.normalize(visual_features.to(device='cuda', dtype=v1.dtype), dim=-1)
        else:
            v2 = F.normalize(torch.from_numpy(visual_features).to(device='cuda', dtype=v1.dtype), dim=-1)
        
        scores = torch.einsum('md,npd->mnp', v1, v2)
        max_scores, _ = scores.max(dim=-1)
        max_scores = max_scores.squeeze(0)
        mean_scores = scores.mean(dim=-1)
        mean_scores = mean_scores.squeeze(0)
        
        return max_scores, mean_scores

def plot_comparison_scores(max_scores, mean_scores, custom_text, video_name, save_path=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    frame_indices = range(len(max_scores))
    
    ax1.plot(frame_indices, max_scores, 'r-', linewidth=2, alpha=0.8)
    ax1.fill_between(frame_indices, max_scores, alpha=0.3, color='red')
    ax1.set_title(f'Max Aggregation - {video_name}\nText: "{custom_text}"')
    ax1.set_ylabel('Similarity Score')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(frame_indices, mean_scores, 'b-', linewidth=2, alpha=0.8)
    ax2.fill_between(frame_indices, mean_scores, alpha=0.3, color='blue')
    ax2.set_title('Mean Aggregation')
    ax2.set_ylabel('Similarity Score')
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(frame_indices, max_scores, 'r-', linewidth=2, alpha=0.8, label='Max')
    ax3.plot(frame_indices, mean_scores, 'b-', linewidth=2, alpha=0.8, label='Mean')
    ax3.set_title('Comparison')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Similarity Score')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def test_custom_text_similarity_compare(video_name, visual_root, custom_text, output_dir=None):
    visual_path = os.path.join(visual_root, f"{video_name}.npy")
    if not os.path.exists(visual_path):
        return None
    
    visual_features = np.load(visual_path)
    max_scores, mean_scores = calc_scores_with_custom_text_compare(visual_features, custom_text, model)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{video_name}_comparison_scores.png")
        plot_comparison_scores(max_scores.cpu().numpy(), mean_scores.cpu().numpy(), 
                             custom_text, video_name, save_path)
    
    return max_scores.cpu().numpy(), mean_scores.cpu().numpy()

def main():
    visual_root = "/root/autodl-tmp/data/SumMe/features/visual"
    output_dir = "/root/tfnet/test/similarity_plots"
    
    video_name = "Air_Force_One"
    custom_text = "A person's face"
    
    max_scores, mean_scores = test_custom_text_similarity_compare(video_name, visual_root, custom_text, output_dir)
    
    if max_scores is not None:
        print(f"Max aggregation - Mean: {np.mean(max_scores):.4f}, Range: [{np.min(max_scores):.4f}, {np.max(max_scores):.4f}]")
        print(f"Mean aggregation - Mean: {np.mean(mean_scores):.4f}, Range: [{np.min(mean_scores):.4f}, {np.max(mean_scores):.4f}]")

if __name__ == "__main__":
    main()
